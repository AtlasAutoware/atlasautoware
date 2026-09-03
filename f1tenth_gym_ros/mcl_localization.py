#!/usr/bin/env python3
"""particle_filter: Monte-Carlo localization against a known map.

Publishes the map-frame pose the racing stack expects:
    /pf/pose/odom    nav_msgs/Odometry in `map`           (raceline_mpc odom_topic)
    tf  map -> odom                                        (completes map->odom->base_link)
    /pf/particles    geometry_msgs/PoseArray               (for rviz/debug)

Motion comes from wheel odometry (/vesc/odom, the odom->base_link transform), the scan
from /scan (or /scan_fused). Set the initial pose with the `initial_x/y/theta` params or
by publishing to /initialpose (rviz "2D Pose Estimate"); with no hint it starts global.

Use this when you have a map of the space (from SLAM below, or maps/*.yaml). For mapping
an unknown space, run slam_toolbox instead (launch/slam_online.launch.py).
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose, PoseWithCovarianceStamped, TransformStamped
from tf2_ros import TransformBroadcaster
try:
    from f1tenth_gym_ros.mcl_core import LikelihoodField, ParticleFilter, load_map
except ImportError:
    from mcl_core import LikelihoodField, ParticleFilter, load_map


def yaw_of(q): return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
def quat(yaw): return (0.0, 0.0, math.sin(yaw/2), math.cos(yaw/2))


class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__('particle_filter')
        P = (('map_yaml', ''), ('scan_topic', '/scan'), ('odom_topic', '/vesc/odom'),
             ('pose_topic', '/pf/pose/odom'), ('n_particles', 600), ('beams', 90),
             ('initial_x', 0.0), ('initial_y', 0.0), ('initial_theta', 0.0),
             ('global_init', False), ('publish_rate', 15.0), ('base_frame', 'base_link'),
             ('odom_frame', 'odom'), ('map_frame', 'map'))
        for k, v in P: self.declare_parameter(k, v)
        g = lambda n: self.get_parameter(n).value
        yaml_path = g('map_yaml')
        if not yaml_path:
            self.get_logger().error('particle_filter needs map_yaml:=<maps/xxx.yaml>'); raise SystemExit(1)
        occ, res, origin = load_map(yaml_path)
        self.field = LikelihoodField(occ, res, origin)
        self.pf = ParticleFilter(self.field, int(g('n_particles')), int(g('beams')))
        if g('global_init'):
            self.pf.init_global(); self.get_logger().info('global initialization')
        else:
            self.pf.init_pose(float(g('initial_x')), float(g('initial_y')), float(g('initial_theta')))
        self.base, self.odomf, self.mapf = g('base_frame'), g('odom_frame'), g('map_frame')
        self.last_odom = None; self.have_scan = False
        self.scan = None
        self.create_subscription(Odometry, g('odom_topic'), self._odom, 20)
        self.create_subscription(LaserScan, g('scan_topic'), self._scan, qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self._initpose, 5)
        self.pose_pub = self.create_publisher(Odometry, g('pose_topic'), 10)
        self.parts_pub = self.create_publisher(PoseArray, '/pf/particles', 2)
        self.tfb = TransformBroadcaster(self)
        self.create_timer(1.0 / float(g('publish_rate')), self._tick)
        self.get_logger().info(f"particle_filter up: map {yaml_path} ({occ.shape}), "
                               f"{int(g('n_particles'))} particles -> {g('pose_topic')}")

    def _odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.last_odom = (p.x, p.y, yaw_of(q))

    def _scan(self, m):
        self.scan = m; self.have_scan = True

    def _initpose(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.pf.init_pose(p.x, p.y, yaw_of(q))
        self.get_logger().info(f'reinitialized at ({p.x:.2f},{p.y:.2f})')

    def _tick(self):
        if self.scan is None:                       # scan is required; odom only refines motion
            return
        if self.last_odom is not None:
            self.pf.predict(self.last_odom)         # no odom yet -> zero-motion, still localizes
        m = self.scan
        self.pf.update(m.ranges, m.angle_min, m.angle_increment, m.range_max or 16.0)
        x, y, th, spread = self.pf.estimate()
        now = self.get_clock().now().to_msg()
        od = Odometry(); od.header.stamp = now; od.header.frame_id = self.mapf; od.child_frame_id = self.base
        od.pose.pose.position.x = x; od.pose.pose.position.y = y
        qz = quat(th); od.pose.pose.orientation.z = qz[2]; od.pose.pose.orientation.w = qz[3]
        c = min(0.5, spread * spread + 1e-3)
        od.pose.covariance[0] = od.pose.covariance[7] = c; od.pose.covariance[35] = c
        self.pose_pub.publish(od)
        self._publish_map_to_odom(x, y, th, now)
        self._publish_particles(now)

    def _publish_map_to_odom(self, x, y, th, stamp):
        """map->base = map->odom * odom->base. We have map->base (estimate) and odom->base
        (last wheel odom), so map->odom = map->base * (odom->base)^-1."""
        if self.last_odom is None:
            return
        ox, oy, oth = self.last_odom
        # inverse of odom->base
        c, s = math.cos(-oth), math.sin(-oth)
        ibx = -(c * ox - s * oy); iby = -(s * ox + c * oy)
        # compose map->base (x,y,th) with (ibx,iby,-oth)
        cb, sb = math.cos(th), math.sin(th)
        mx = x + cb * ibx - sb * iby
        my = y + sb * ibx + cb * iby
        mth = th - oth
        t = TransformStamped(); t.header.stamp = stamp; t.header.frame_id = self.mapf; t.child_frame_id = self.odomf
        t.transform.translation.x = mx; t.transform.translation.y = my
        q = quat(mth); t.transform.rotation.z = q[2]; t.transform.rotation.w = q[3]
        self.tfb.sendTransform(t)

    def _publish_particles(self, stamp):
        pa = PoseArray(); pa.header.stamp = stamp; pa.header.frame_id = self.mapf
        P = self.pf.P
        step = max(1, len(P) // 200)
        for x, y, th in P[::step]:
            ps = Pose(); ps.position.x = float(x); ps.position.y = float(y)
            q = quat(float(th)); ps.orientation.z = q[2]; ps.orientation.w = q[3]
            pa.poses.append(ps)
        self.parts_pub.publish(pa)


def main(args=None):
    rclpy.init(args=args); n = ParticleFilterNode()
    try: rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit): pass
    try: n.destroy_node(); rclpy.shutdown()
    except Exception: pass


if __name__ == '__main__':
    main()
