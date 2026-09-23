#!/usr/bin/env python3
"""pose_relay: publish the map-frame base pose as Odometry on /pf/pose/odom.

slam_toolbox (and amcl) provide the map->odom transform, not the Odometry message the
racing stack consumes. This looks up map->base_link from TF and republishes it, so the
same /pf/pose/odom topic is populated whether localization comes from SLAM or the
particle filter. Runs with the SLAM launch; not needed when particle_filter is used
(that node publishes /pf/pose/odom itself).
"""
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException


class PoseRelay(Node):
    def __init__(self):
        super().__init__('pose_relay')
        for k, v in (('map_frame', 'map'), ('base_frame', 'base_link'),
                     ('pose_topic', '/pf/pose/odom'), ('rate', 20.0)):
            self.declare_parameter(k, v)
        g = lambda n: self.get_parameter(n).value
        self.mapf, self.base = g('map_frame'), g('base_frame')
        self.buf = Buffer(); TransformListener(self.buf, self)
        self.pub = self.create_publisher(Odometry, g('pose_topic'), 10)
        self.warned = False
        self.create_timer(1.0 / float(g('rate')), self._tick)
        self.get_logger().info(f'pose_relay: {self.mapf}->{self.base} TF -> {g("pose_topic")}')

    def _tick(self):
        try:
            t = self.buf.lookup_transform(self.mapf, self.base, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            if not self.warned:
                self.get_logger().warn('waiting for map->base_link TF (is SLAM/localization up?)')
                self.warned = True
            return
        od = Odometry(); od.header = t.header; od.child_frame_id = self.base
        od.pose.pose.position.x = t.transform.translation.x
        od.pose.pose.position.y = t.transform.translation.y
        od.pose.pose.orientation = t.transform.rotation
        self.pub.publish(od)


def main(args=None):
    rclpy.init(args=args); n = PoseRelay()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    try: n.destroy_node(); rclpy.shutdown()
    except Exception: pass


if __name__ == '__main__':
    main()
