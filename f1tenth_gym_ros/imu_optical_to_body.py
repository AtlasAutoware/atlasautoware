#!/usr/bin/env python3
"""Re-express an IMU published in a camera *optical* frame in the ROS *body* frame.

Orbbec (and most depth cameras) report accel/gyro in the optical convention:
    x -> right, y -> down, z -> forward
The rest of this stack (velocity_ekf, raceline_mpc) was written against the ROS
body convention (REP 103):
    x -> forward, y -> left, z -> up
so a plain topic remap would feed lateral acceleration as "forward" and roll rate
as "yaw rate". This relay rotates both vectors:
    body.x = optical.z      body.y = -optical.x      body.z = -optical.y
Sanity check with the car level and still: body.z linear_acceleration ~ +9.8.

Params: in_topic (default /camera/gyro_accel/sample), out_topic (/oakd/imu),
        frame_id ('' = keep incoming), flip_x/flip_y/flip_z (for a camera mounted
        upside-down or backwards; applied after the rotation).
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


class ImuOpticalToBody(Node):
    def __init__(self):
        super().__init__('imu_optical_to_body')
        self.declare_parameter('in_topic', '/camera/gyro_accel/sample')
        self.declare_parameter('out_topic', '/oakd/imu')
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('flip_x', False)
        self.declare_parameter('flip_y', False)
        self.declare_parameter('flip_z', False)
        p = lambda n: self.get_parameter(n).value
        self.frame_id = p('frame_id')
        self.sign = (-1.0 if p('flip_x') else 1.0, -1.0 if p('flip_y') else 1.0, -1.0 if p('flip_z') else 1.0)
        # RELIABLE on purpose: velocity_ekf / raceline_mpc subscribe with default (reliable)
        # QoS, which cannot match a best-effort publisher (they would silently get nothing).
        self.pub = self.create_publisher(Imu, p('out_topic'), 50)
        self.create_subscription(Imu, p('in_topic'), self._cb, qos_profile_sensor_data)
        self.get_logger().info(f"imu optical->body: {p('in_topic')} -> {p('out_topic')} (frame {self.frame_id or 'passthrough'})")

    def _rot(self, v):
        sx, sy, sz = self.sign
        return sx * v.z, sy * -v.x, sz * -v.y

    def _cb(self, m):
        out = Imu()
        out.header = m.header
        if self.frame_id:
            out.header.frame_id = self.frame_id
        out.orientation = m.orientation
        out.orientation_covariance = m.orientation_covariance
        a = out.linear_acceleration; a.x, a.y, a.z = self._rot(m.linear_acceleration)
        w = out.angular_velocity; w.x, w.y, w.z = self._rot(m.angular_velocity)
        # covariances: permute diagonal (off-diagonals are zero from these drivers)
        for src, dst in ((m.linear_acceleration_covariance, out.linear_acceleration_covariance),
                         (m.angular_velocity_covariance, out.angular_velocity_covariance)):
            dst[0], dst[4], dst[8] = src[8], src[0], src[4]
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ImuOpticalToBody()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
