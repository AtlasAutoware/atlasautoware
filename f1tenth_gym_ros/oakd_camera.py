"""
OAK-D Pro driver — RGB frames + onboard IMU for the real car.
=============================================================

Publishes from a Luxonis OAK-D Pro over the `depthai` Python API (pure pip,
works on Foxy — no depthai-ros packages needed on the Jetson):

  /oakd/rgb          sensor_msgs/Image  (bgr8)   -> camera_perception YOLO
  /oakd/camera_info  sensor_msgs/CameraInfo      -> pinhole back-projection
  /oakd/imu          sensor_msgs/Imu (accel+gyro)-> raceline_mpc traction governor

Design notes:
  - The RGB preview stream is configured interleaved-BGR at the requested size
    so frames go straight into an Image message as raw bytes — no cv_bridge,
    no per-frame colour conversion on the Jetson CPU.
  - Intrinsics come from the device's factory calibration (readCalibration),
    scaled to the published resolution, so camera_perception gets a correct
    fx/cx without hand-tuning.
  - IMU runs at `imu_hz` (accel + gyro, BMI270/BNO086 depending on unit) and is
    drained on a fast timer; orientation is not estimated (covariance[0] = -1).
    Mount note: the governor downstream uses |gyro z|, so any flat (z-up or
    z-down) mounting works without sign config.
  - Sensor-data QoS (best-effort, depth 1): a late frame is a useless frame.

Run:
    ros2 run f1tenth_gym_ros oakd_camera --ros-args --params-file config/hardware.yaml
"""

try:
    import depthai as dai
except Exception:                       # pragma: no cover — not on dev machines
    dai = None


def build_pipeline(width, height, fps, imu_hz):
    """depthai pipeline: RGB preview (interleaved BGR) + raw accel/gyro IMU."""
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setPreviewSize(int(width), int(height))
    cam.setInterleaved(True)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(float(fps))
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName('rgb')
    cam.preview.link(xout_rgb.input)

    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(
        [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW],
        int(imu_hz))
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(20)
    xout_imu = pipeline.create(dai.node.XLinkOut)
    xout_imu.setStreamName('imu')
    imu.out.link(xout_imu.input)
    return pipeline


def _make_node():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo, Imu

    class OakDCamera(Node):
        def __init__(self):
            super().__init__('oakd_camera')
            self.declare_parameter('width', 640)
            self.declare_parameter('height', 360)
            self.declare_parameter('fps', 30.0)
            self.declare_parameter('imu_hz', 200)
            self.declare_parameter('rgb_topic', '/oakd/rgb')
            self.declare_parameter('info_topic', '/oakd/camera_info')
            self.declare_parameter('imu_topic', '/oakd/imu')
            self.declare_parameter('camera_frame', 'oakd_rgb')
            self.declare_parameter('imu_frame', 'oakd_imu')
            p = lambda n: self.get_parameter(n).value   # noqa: E731
            self.w, self.h = int(p('width')), int(p('height'))
            self.cam_frame = p('camera_frame')
            self.imu_frame = p('imu_frame')

            if dai is None:
                raise RuntimeError('depthai not installed — pip3 install depthai')

            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.pub_rgb = self.create_publisher(Image, p('rgb_topic'), qos)
            self.pub_info = self.create_publisher(CameraInfo, p('info_topic'), qos)
            self.pub_imu = self.create_publisher(Imu, p('imu_topic'),
                                                 QoSProfile(depth=50))

            self.device = dai.Device(
                build_pipeline(self.w, self.h, p('fps'), p('imu_hz')))
            self.q_rgb = self.device.getOutputQueue('rgb', maxSize=2, blocking=False)
            self.q_imu = self.device.getOutputQueue('imu', maxSize=50, blocking=False)
            self.info = self._camera_info()
            self.create_timer(1.0 / (2.0 * float(p('fps'))), self._poll_rgb)
            self.create_timer(0.002, self._poll_imu)
            self.get_logger().info(
                f'oakd_camera ready — rgb {self.w}x{self.h}@{p("fps"):.0f} '
                f'imu @{p("imu_hz")}Hz ({self.device.getMxId()})')

        def _camera_info(self):
            info = CameraInfo()
            info.header.frame_id = self.cam_frame
            info.width, info.height = self.w, self.h
            try:
                calib = self.device.readCalibration()
                K = calib.getCameraIntrinsics(
                    dai.CameraBoardSocket.RGB, self.w, self.h)
                info.k = [float(v) for row in K for v in row]
                info.p = [info.k[0], info.k[1], info.k[2], 0.0,
                          info.k[3], info.k[4], info.k[5], 0.0,
                          info.k[6], info.k[7], info.k[8], 0.0]
                info.distortion_model = 'plumb_bob'
                d = calib.getDistortionCoefficients(dai.CameraBoardSocket.RGB)
                info.d = [float(v) for v in d[:5]]
            except Exception as e:                      # uncalibrated dev unit
                self.get_logger().warning(f'no factory calibration: {e}')
            return info

        def _poll_rgb(self):
            frame = self.q_rgb.tryGet()
            if frame is None:
                return
            stamp = self.get_clock().now().to_msg()
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = self.cam_frame
            msg.height, msg.width = self.h, self.w
            msg.encoding = 'bgr8'
            msg.is_bigendian = 0
            msg.step = self.w * 3
            msg.data = frame.getData().tobytes()
            self.pub_rgb.publish(msg)
            self.info.header.stamp = stamp
            self.pub_info.publish(self.info)

        def _poll_imu(self):
            data = self.q_imu.tryGet()
            if data is None:
                return
            for pkt in data.packets:
                msg = Imu()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.imu_frame
                acc, gyr = pkt.acceleroMeter, pkt.gyroscope
                msg.linear_acceleration.x = float(acc.x)
                msg.linear_acceleration.y = float(acc.y)
                msg.linear_acceleration.z = float(acc.z)
                msg.angular_velocity.x = float(gyr.x)
                msg.angular_velocity.y = float(gyr.y)
                msg.angular_velocity.z = float(gyr.z)
                msg.orientation_covariance[0] = -1.0    # orientation not provided
                self.pub_imu.publish(msg)

        def shutdown(self):
            try:
                self.device.close()
            except Exception:
                pass

    return rclpy, OakDCamera


def main(args=None):
    rclpy, NodeCls = _make_node()
    rclpy.init(args=args)
    node = None
    try:
        node = NodeCls()
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        if node is not None:
            node.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
