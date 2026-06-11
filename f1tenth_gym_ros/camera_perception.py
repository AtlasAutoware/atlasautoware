"""
Camera perception — detect other cars with a trained YOLO model and feed the
same race brain the lidar does.
=============================================================================

This runs on the **real car** (the f1tenth_gym sim has no camera).  It is
deliberately dependency-light for deployment: inference uses a YOLOv8 model
exported to **ONNX**, run through OpenCV's `cv2.dnn` — so the car only needs
OpenCV, not PyTorch.  (Training is done separately on a GPU machine — see
`tools/train_car_detector.py`.)

Pipeline:  image  ->  YOLO car boxes  ->  back-project each box to a position
relative to the car (pinhole + known car width)  ->  `race_brain.Opponent`
objects, in the **same format the lidar detector emits**, so they drop straight
into `RaceStrategist` or fuse with the lidar tracks.

The output is sensor-agnostic on purpose: camera gives "what + bearing", lidar
gives precise range; fusing them (camera class/bearing + lidar range) is the
robust setup, and both already speak the `Opponent` type.

Standalone (no ROS) test of the detector + geometry:
    from camera_perception import CarDetector, box_to_relative
"""

import math
import os

import numpy as np

try:
    import cv2
except Exception:                      # pragma: no cover
    cv2 = None


# ─────────────────────────────────────────────────────────────────────────────
# Detector — YOLOv8 ONNX via cv2.dnn (CPU/GPU, no torch needed)
# ─────────────────────────────────────────────────────────────────────────────

class CarDetector:
    def __init__(self, model_path, img_size=640, conf=0.35, nms=0.45,
                 car_class=0, use_cuda=False):
        if cv2 is None:
            raise RuntimeError('OpenCV not available')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'model not found: {model_path} — train one with '
                f'tools/train_car_detector.py and export to ONNX')
        self.net = cv2.dnn.readNetFromONNX(model_path)
        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.sz, self.conf, self.nms, self.car_class = img_size, conf, nms, car_class

    def detect(self, img):
        """BGR image -> list of (x, y, w, h, confidence) car boxes in image px."""
        h0, w0 = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.sz, self.sz),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()                       # (1, 4+nc, N)
        return self._parse(out, w0, h0)

    def _parse(self, out, w0, h0):
        out = np.squeeze(out)                           # (4+nc, N)
        if out.ndim != 2:
            return []
        if out.shape[0] < out.shape[1]:                 # (4+nc, N) -> (N, 4+nc)
            out = out.T
        cls = out[:, 4:]
        cids = np.argmax(cls, axis=1)
        confs = cls[np.arange(len(cls)), cids]
        keep = (confs > self.conf) & (cids == self.car_class)
        if not keep.any():
            return []
        rows, confs = out[keep], confs[keep]
        sx, sy = w0 / self.sz, h0 / self.sz
        cx, cy, bw, bh = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        boxes = np.stack([(cx - bw/2)*sx, (cy - bh/2)*sy, bw*sx, bh*sy], 1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), confs.tolist(), self.conf, self.nms)
        if len(idx) == 0:
            return []
        idx = np.array(idx).flatten()
        return [(*boxes[i], float(confs[i])) for i in idx]


# ─────────────────────────────────────────────────────────────────────────────
# Geometry — image box -> position relative to the car
# ─────────────────────────────────────────────────────────────────────────────

def box_to_relative(box, fx, cx_img, car_width=0.30):
    """
    Pinhole back-projection with a known object width.
    box = (x, y, w, h) px.  Returns (x_forward, y_left, range) in metres, in the
    camera/base frame (ROS: +X forward, +Y left).
    """
    x, y, w, h = box[:4]
    u_center = x + w / 2.0
    depth = car_width * fx / max(w, 1.0)               # along camera optical axis
    y_left = -(u_center - cx_img) * depth / fx         # right of centre -> -Y
    rng = math.hypot(depth, y_left)
    return depth, y_left, rng


def relative_to_world(x_fwd, y_left, ego):
    ex, ey, eyaw = ego
    wx = ex + x_fwd * math.cos(eyaw) - y_left * math.sin(eyaw)
    wy = ey + x_fwd * math.sin(eyaw) + y_left * math.cos(eyaw)
    return wx, wy


# ─────────────────────────────────────────────────────────────────────────────
# ROS node (only imported/used on the car)
# ─────────────────────────────────────────────────────────────────────────────

def _make_node():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from nav_msgs.msg import Odometry
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import PoseArray, Pose
    from std_msgs.msg import ColorRGBA
    from transforms3d.euler import quat2euler

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from race_brain import Opponent, OpponentDetector  # noqa: reuse the tracker

    class CameraPerception(Node):
        def __init__(self):
            super().__init__('camera_perception')
            self.declare_parameter('image_topic', '/camera/color/image_raw')
            self.declare_parameter('model_path', os.path.join(
                os.path.dirname(__file__), '..', 'models', 'car_yolov8.onnx'))
            self.declare_parameter('fx', 600.0)        # camera intrinsics
            self.declare_parameter('cx', 320.0)
            self.declare_parameter('car_width', 0.30)
            self.declare_parameter('conf', 0.35)
            self.declare_parameter('odom_topic', '/ego_racecar/odom')
            topic = self.get_parameter('image_topic').value
            self.fx = float(self.get_parameter('fx').value)
            self.cx = float(self.get_parameter('cx').value)
            self.car_w = float(self.get_parameter('car_width').value)

            try:
                self.detector = CarDetector(self.get_parameter('model_path').value,
                                            conf=float(self.get_parameter('conf').value))
                self.get_logger().info('YOLO car detector loaded (ONNX/cv2.dnn)')
            except Exception as e:
                self.detector = None
                self.get_logger().error(f'No detector ({e}); node idle until a model exists')

            # Reuse the lidar tracker purely for smoothing camera detections.
            self.tracker = OpponentDetector()
            self.ego = (0.0, 0.0, 0.0)
            odom_topic = self.get_parameter('odom_topic').value
            self.create_subscription(Odometry, odom_topic, self._odom, 10)
            self.create_subscription(Image, topic, self._image, 5)
            self.opp_pub = self.create_publisher(MarkerArray, '/camera_opponents', 5)
            self.pose_pub = self.create_publisher(PoseArray, '/camera_opponents_poses', 5)
            self.Marker, self.MarkerArray = Marker, MarkerArray
            self.PoseArray, self.Pose = PoseArray, Pose
            self.ColorRGBA = ColorRGBA
            self.get_logger().info(f'camera_perception subscribed to {topic}')

        def _odom(self, m):
            q = m.pose.pose.orientation
            from transforms3d.euler import quat2euler as q2e
            _, _, yaw = q2e([q.w, q.x, q.y, q.z])
            self.ego = (m.pose.pose.position.x, m.pose.pose.position.y, yaw)

        def _image(self, msg):
            if self.detector is None:
                return
            img = self._decode(msg)
            if img is None:
                return
            opps = []
            for box in self.detector.detect(img):
                xf, yl, rng = box_to_relative(box, self.fx, self.cx, self.car_w)
                wx, wy = relative_to_world(xf, yl, self.ego)
                o = Opponent(wx, wy, rng, self.car_w, math.atan2(yl, xf))
                opps.append(o)
            self._publish(opps)

        @staticmethod
        def _decode(msg):
            try:
                from cv_bridge import CvBridge
                return CvBridge().imgmsg_to_cv2(msg, 'bgr8')
            except Exception:
                if msg.encoding in ('rgb8', 'bgr8'):
                    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
                    return a[:, :, ::-1] if msg.encoding == 'rgb8' else a
                return None

        def _publish(self, opps):
            pa = self.PoseArray()
            pa.header.frame_id = 'map'
            for o in opps:
                p = self.Pose()
                p.position.x, p.position.y = float(o.x), float(o.y)
                p.orientation.w = 1.0
                pa.poses.append(p)
            self.pose_pub.publish(pa)

            arr = self.MarkerArray()
            for k, o in enumerate(opps):
                m = self.Marker()
                m.header.frame_id = 'map'; m.ns = 'camera_opp'; m.id = k
                m.type = self.Marker.CUBE; m.action = self.Marker.ADD
                m.pose.position.x, m.pose.position.y, m.pose.position.z = o.x, o.y, 0.1
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = 0.4; m.scale.z = 0.2
                m.color = self.ColorRGBA(r=0.1, g=0.6, b=0.9, a=0.9)
                arr.markers.append(m)
            self.opp_pub.publish(arr)

    return CameraPerception


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = _make_node()()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
