"""
Mapping driver — safe, slow autonomous laps for building a map with SLAM.
========================================================================

On a new track you don't have a map for, you get practice time to make one:
run `slam_toolbox` (see launch/slam_mapping_launch.py) and drive a few clean
laps while it builds the occupancy grid.  This node drives those laps for you,
conservatively, so you don't have to hand-pilot.

It uses the **disparity extender** follow-the-gap algorithm (robust on closed
circuits — far more reliable than naive gap-finding): at every big range step it
"extends" the nearer obstacle by the car's half-width so the car never clips an
edge, then aims at the deepest remaining gap.  Speed is held low and scaled by
forward clearance — the goal is a clean map, not a fast lap.

    python3 f1tenth_gym_ros/mapping_driver.py --speed 1.5

Topics, steering limit and speed are ROS parameters so the same node drives the
sim and the real car.  Hardware (f1tenth_system) uses /scan and /drive too, so
the defaults work as-is; override if your stack remaps them, e.g.
    ros2 run f1tenth_gym_ros mapping_driver --ros-args -p drive_topic:=/nav/drive
"""

import argparse

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class MappingDriver(Node):
    def __init__(self, speed=1.5, car_half=0.20, disparity=0.30, max_range=6.0):
        super().__init__('mapping_driver')
        # ROS params (CLI defaults below are overridden by any -p value).
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('max_steer', 0.41)        # rad, steering limit
        self.declare_parameter('speed', speed)           # m/s, mapping-lap speed
        scan_topic  = self.get_parameter('scan_topic').value
        drive_topic = self.get_parameter('drive_topic').value
        self.max_steer = float(self.get_parameter('max_steer').value)
        self.speed = float(self.get_parameter('speed').value)
        self.car_half = car_half          # m, inflate obstacles by this
        self.disparity = disparity        # m, range step that counts as an edge
        self.max_range = max_range
        self.create_subscription(LaserScan, scan_topic, self._scan, 10)
        self.pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.get_logger().info(
            f'mapping driver @ {self.speed} m/s on {scan_topic} -> {drive_topic} '
            f'— drive clean laps for SLAM')

    def _scan(self, msg):
        r = np.asarray(msg.ranges, np.float32)
        r = np.where(np.isfinite(r), r, self.max_range)
        r = np.clip(r, 0.0, self.max_range)
        n = len(r)
        ang = msg.angle_min + np.arange(n) * msg.angle_increment

        # Consider only the forward 180° (ignore beams pointing backwards).
        fwd = np.abs(ang) < (np.pi / 2)
        proc = r.copy()

        # Disparity extender: at each big step, extend the nearer side by the
        # number of beams that span the car's half-width at that range.
        for i in range(1, n):
            if abs(r[i] - r[i - 1]) > self.disparity:
                near = min(r[i], r[i - 1])
                span = int(np.arctan2(self.car_half, max(near, 0.1)) / msg.angle_increment)
                if r[i] < r[i - 1]:
                    proc[i:min(n, i + span)] = np.minimum(proc[i:min(n, i + span)], near)
                else:
                    proc[max(0, i - span):i] = np.minimum(proc[max(0, i - span):i], near)

        proc_fwd = np.where(fwd, proc, 0.0)
        target = int(np.argmax(proc_fwd))                 # deepest forward gap
        steer = float(np.clip(ang[target], -self.max_steer, self.max_steer))

        front = proc[np.abs(ang) < 0.26]
        clear = float(front.min()) if len(front) else self.max_range
        spd = self.speed * float(np.clip((clear - 0.5) / 2.0, 0.3, 1.0))

        m = AckermannDriveStamped()
        m.drive.steering_angle = steer
        m.drive.speed = max(spd, 0.5)
        self.pub.publish(m)


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=float, default=1.5)
    a, _ = ap.parse_known_args()
    rclpy.init(args=args)
    try:
        rclpy.spin(MappingDriver(speed=a.speed))
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
