"""
Opponent car — a simple, slower raceline follower so the race agent has someone
to attack / defend against.  Pure pursuit on the same raceline, speed-capped.

Run:  python3 f1tenth_gym_ros/opponent_driver.py [--cap 2.5]
"""

import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from transforms3d.euler import quat2euler

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pursuit_agent import (
    find_best_raceline, load_raceline, find_nearest, find_lookahead, pp_steer)


class OpponentDriver(Node):
    def __init__(self, cap):
        super().__init__('opponent_driver')
        self.cap = cap
        self.x = self.y = self.yaw = self.speed = 0.0
        self.nearest = 0
        self._locked = False
        rl = find_best_raceline()
        self.rl_x, self.rl_y, _, _, self.rl_speed = load_raceline(rl)
        self.create_subscription(Odometry, '/opp_racecar/odom', self._odom, 10)
        self.pub = self.create_publisher(AckermannDriveStamped, '/opp_drive', 10)
        self.create_timer(0.02, self._loop)
        self.get_logger().info(f'Opponent driver ready (cap {cap} m/s)')

    def _odom(self, m):
        self.x = m.pose.pose.position.x
        self.y = m.pose.pose.position.y
        self.speed = np.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y)
        q = m.pose.pose.orientation
        _, _, self.yaw = quat2euler([q.w, q.x, q.y, q.z])

    def _loop(self):
        if not self._locked:
            self.nearest = int(np.argmin(np.hypot(self.rl_x - self.x, self.rl_y - self.y)))
            self._locked = True
        self.nearest = find_nearest(self.x, self.y, self.rl_x, self.rl_y, self.nearest)
        L = float(np.clip(0.3 * self.speed + 0.9, 0.9, 2.0))
        tx, ty, _ = find_lookahead(self.x, self.y, self.yaw,
                                   self.rl_x, self.rl_y, L, self.nearest)
        steer = pp_steer(self.x, self.y, self.yaw, tx, ty, L, 0.41)
        spd = min(self.cap, float(self.rl_speed[self.nearest]))
        spd *= (1.0 - 0.3 * abs(steer) / 0.41)
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(max(spd, 0.5))
        self.pub.publish(msg)


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--cap', type=float, default=2.5)
    a, _ = ap.parse_known_args()
    rclpy.init(args=args)
    try:
        rclpy.spin(OpponentDriver(a.cap))
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
