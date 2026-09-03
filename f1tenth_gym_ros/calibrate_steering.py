#!/usr/bin/env python3
"""calibrate_steering: measure the steering centre from the car's own sensors.

Drives straight at a low constant speed with zero commanded steering and measures the yaw
rate. If the car is truly centred the yaw rate is zero; anything else is the mechanical
bias, in radians of steering:

    delta_bias = yaw_rate * wheelbase / speed          (bicycle model)

which converts to the servo centre the config should hold:

    offset_true = offset_cfg - gain * delta_bias

It runs several short passes in both directions where possible, discards the acceleration
transient, and reports the median so one bad pass cannot skew it. It prints the number and,
with --apply, writes it to vesc.yaml. This replaces trimming by feel, and unlike a trim it
costs no steering throw.

    ros2 run f1tenth_gym_ros calibrate_steering --ros-args -p speed:=0.6 -p passes:=3

SAFETY: this moves the car. It needs a few metres of clear space, stops on the emergency
brake distance from the lidar, holds a hard time limit per pass, and publishes zero speed
on exit. Hold a key or the gamepad dead-man to override at any time (teleop outranks it).
"""
import json, math, os, statistics, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

VESC = os.path.expanduser('~/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/vesc.yaml')


class Calibrate(Node):
    def __init__(self):
        super().__init__('calibrate_steering')
        P = (('speed', 0.6), ('passes', 3), ('duration', 2.5), ('settle', 0.7),
             ('wheelbase', 0.324), ('aeb_dist', 0.6), ('imu_topic', '/oakd/imu'),
             ('odom_topic', '/odom'), ('scan_topic', '/scan'), ('drive_topic', '/drive'),
             ('vesc_yaml', VESC), ('apply', False))
        for k, v in P: self.declare_parameter(k, v)
        g = lambda n: self.get_parameter(n).value
        self.p = {k: g(k) for k, _ in P}
        self.yaw = None; self.speed = 0.0; self.front = 99.0
        self.create_subscription(Imu, self.p['imu_topic'], self._imu, 50)
        self.create_subscription(Odometry, self.p['odom_topic'], self._odom, 20)
        self.create_subscription(LaserScan, self.p['scan_topic'], self._scan, qos_profile_sensor_data)
        self.pub = self.create_publisher(AckermannDriveStamped, self.p['drive_topic'], 10)

    def _imu(self, m): self.yaw = m.angular_velocity.z
    def _odom(self, m): self.speed = m.twist.twist.linear.x
    def _scan(self, m):
        n = len(m.ranges); c = n // 2 if m.angle_min < -1.0 else 0
        w = max(1, int(0.2 / max(m.angle_increment, 1e-6)))
        seg = np.asarray(m.ranges[max(0, c - w):c + w], np.float32)
        seg = seg[np.isfinite(seg) & (seg > 0.05)]
        self.front = float(seg.min()) if seg.size else 99.0

    def _drive(self, v):
        m = AckermannDriveStamped(); m.drive.speed = float(v); m.drive.steering_angle = 0.0
        self.pub.publish(m)

    def _spin(self, sec):
        t = time.time()
        while time.time() - t < sec: rclpy.spin_once(self, timeout_sec=0.02)

    def one_pass(self, v):
        """Returns (mean yaw rate, mean speed) over the steady window, or None if aborted."""
        ys, vs = [], []
        t0 = time.time()
        while time.time() - t0 < self.p['duration']:
            if self.front < self.p['aeb_dist']:
                self._drive(0.0); self.get_logger().warn(f'obstacle at {self.front:.2f} m — pass aborted')
                self._spin(0.3); return None
            self._drive(v)
            rclpy.spin_once(self, timeout_sec=0.02)
            if time.time() - t0 > self.p['settle'] and self.yaw is not None and self.speed > 0.15:
                ys.append(self.yaw); vs.append(self.speed)
        self._drive(0.0); self._spin(1.2)                      # coast to a stop between passes
        if len(ys) < 20: return None
        return statistics.median(ys), statistics.median(vs)

    def run(self):
        self.get_logger().info('waiting for sensors...')
        t = time.time()
        while (self.yaw is None) and time.time() - t < 10.0: rclpy.spin_once(self, timeout_sec=0.1)
        if self.yaw is None:
            self.get_logger().error(f"no IMU on {self.p['imu_topic']} — is the camera running?"); return 1
        biases = []
        for i in range(int(self.p['passes'])):
            self.get_logger().info(f"pass {i+1}/{self.p['passes']}: driving straight at {self.p['speed']} m/s")
            r = self.one_pass(self.p['speed'])
            if r is None: continue
            yaw, spd = r
            bias = yaw * self.p['wheelbase'] / max(spd, 0.05)
            biases.append(bias)
            self.get_logger().info(f'  yaw {yaw:+.4f} rad/s at {spd:.2f} m/s -> steering bias {bias:+.4f} rad '
                                   f'({math.degrees(bias):+.2f} deg)')
        self._drive(0.0)
        if not biases:
            self.get_logger().error('no usable passes (obstacle, or the car never moved)'); return 1
        bias = statistics.median(biases)
        import re
        v = open(self.p['vesc_yaml']).read()
        gain = float(re.search(r'steering_angle_to_servo_gain:\s*(-?[\d.]+)', v).group(1))
        off = float(re.search(r'steering_angle_to_servo_offset:\s*(-?[\d.]+)', v).group(1))
        new = off - gain * bias
        print(f'\nmedian steering bias {bias:+.4f} rad ({math.degrees(bias):+.2f} deg) over {len(biases)} pass(es)')
        print(f'steering_angle_to_servo_offset: {off:.4f} -> {new:.4f}')
        if self.p['apply']:
            open(self.p['vesc_yaml'], 'w').write(
                re.sub(r'(steering_angle_to_servo_offset:\s*)(-?[\d.]+)', rf'\g<1>{new:.4f}', v))
            print(f'written. rebuild f1tenth_stack and restart remote mode.')
        else:
            print('re-run with -p apply:=true to write it, or set it by hand.')
        return 0


def main(args=None):
    rclpy.init(args=args); n = Calibrate()
    try:
        rc = n.run()
    except KeyboardInterrupt:
        rc = 1
    finally:
        try:
            n._drive(0.0); n._spin(0.2)                        # never leave a speed command latched
        except Exception: pass
    n.destroy_node(); rclpy.shutdown(); sys.exit(rc)


if __name__ == '__main__':
    main()
