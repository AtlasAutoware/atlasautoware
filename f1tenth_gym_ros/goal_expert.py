#!/usr/bin/env python3
"""goal_expert: follows a path to a goal with pure pursuit and publishes /drive.

The collector plans the route (A* on the map) and describes it in language; this node
only executes it, so it is the goal-conditioned *expert* whose actions the policy
imitates. Same output channel as raceline_mpc and a policy server (/drive), same AEB.

    /goal_expert/path     nav_msgs/Path      route to follow (world frame)
    /goal_expert/status   String json        {done, dist_goal, following}
"""
import json, math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String
try:
    from f1tenth_gym_ros.goal_core import pure_pursuit
except ImportError:
    from goal_core import pure_pursuit


def yaw_of(q): return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class GoalExpert(Node):
    def __init__(self):
        super().__init__('goal_expert')
        P = (('odom_topic', '/pf/pose/odom'), ('scan_topic', '/scan'), ('drive_topic', '/drive'),
             ('lookahead', 0.6), ('wheelbase', 0.33), ('max_steer', 0.4), ('v_max', 1.5),
             ('v_min', 0.4), ('goal_radius', 0.35), ('aeb_dist', 0.30), ('rate', 20.0))
        for k, v in P: self.declare_parameter(k, v)
        g = lambda n: self.get_parameter(n).value
        self.p = {k: g(k) for k, _ in P}
        self.pose = None; self.path = None; self.done = False; self.front = 99.0
        self.create_subscription(Odometry, self.p['odom_topic'], self._odom, 10)
        self.create_subscription(LaserScan, self.p['scan_topic'], self._scan, qos_profile_sensor_data)
        self.create_subscription(Path, '/goal_expert/path', self._path, 5)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.p['drive_topic'], 10)
        self.st_pub = self.create_publisher(String, '/goal_expert/status', 5)
        self.create_timer(1.0 / float(self.p['rate']), self._tick)
        self.get_logger().info('goal_expert ready: send a nav_msgs/Path on /goal_expert/path')

    def _odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.pose = (p.x, p.y, yaw_of(q))

    def _scan(self, m):
        n = len(m.ranges); c = n // 2 if m.angle_min < -1.0 else 0
        w = max(1, int(0.2 / max(m.angle_increment, 1e-6)))
        seg = np.asarray(m.ranges[max(0, c - w):c + w], np.float32)
        seg = seg[np.isfinite(seg) & (seg > 0.05)]
        self.front = float(seg.min()) if seg.size else 99.0

    def _path(self, m):
        self.path = [(ps.pose.position.x, ps.pose.position.y) for ps in m.poses]
        self.done = False
        self.get_logger().info(f'new path: {len(self.path)} pts')

    def _tick(self):
        st = {'done': self.done, 'following': self.path is not None and not self.done}
        cmd = AckermannDriveStamped()
        if self.pose is not None and self.path and not self.done:
            v, steer, done, dg = pure_pursuit(self.pose, self.path, self.p['lookahead'], self.p['wheelbase'],
                                              self.p['max_steer'], self.p['v_max'], self.p['v_min'],
                                              self.p['goal_radius'])
            if self.front < self.p['aeb_dist'] and v > 0:
                v = 0.0                                       # emergency stop, hold steer
            self.done = done
            cmd.drive.speed = v; cmd.drive.steering_angle = steer
            st.update({'dist_goal': round(dg, 2), 'speed': round(v, 2), 'front': round(self.front, 2)})
        self.drive_pub.publish(cmd)
        self.st_pub.publish(String(data=json.dumps(st)))


def main(args=None):
    rclpy.init(args=args); n = GoalExpert()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    try: n.destroy_node(); rclpy.shutdown()
    except Exception: pass


if __name__ == '__main__':
    main()
