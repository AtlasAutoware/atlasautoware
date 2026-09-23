#!/usr/bin/env python3
"""Decide the odometry speed sign without running the motor.

vesc_to_odom turns the VESC eRPM reading into /odom twist.linear.x, and whether it negates
the raw reading changed upstream (commit 1bc8251, on the humble branch). Rather than trust
either version of the driver, this reads what the hardware actually reports while you push
the car forward by hand: a brushless motor back-driven by the wheels still produces a signed
eRPM, so the whole question can be settled with the throttle untouched.

    python3 tools/check_odom_sign.py

Push the car forward about a metre when it says to. It prints the eRPM sign, the resulting
odom sign, and the speed_to_erpm_gain that belongs under vesc_to_odom_node in vesc.yaml.
Publishes nothing.
"""
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from vesc_msgs.msg import VescStateStamped

GAIN = 3575.0            # magnitude from vesc.yaml; only the sign is in question here


class OdomSign(Node):
    def __init__(self):
        super().__init__('check_odom_sign')
        self.erpm, self.vx = [], []
        self.create_subscription(VescStateStamped, '/sensors/core', self._core, 10)
        self.create_subscription(Odometry, '/odom', self._odom, 10)

    def _core(self, m):
        self.erpm.append(float(m.state.speed))

    def _odom(self, m):
        self.vx.append(float(m.twist.twist.linear.x))


def main():
    rclpy.init()
    n = OdomSign()
    print('Push the car FORWARD by hand, about a metre, over the next 10 seconds.')
    print('Do not touch the throttle.')
    t0 = time.time()
    while time.time() - t0 < 10.0:
        rclpy.spin_once(n, timeout_sec=0.1)
    moving = [x for x in n.erpm if abs(x) > 50]
    speeds = [x for x in n.vx if abs(x) > 0.05]
    if not moving:
        print('no motion seen in %d telemetry messages. Is the VESC powered? Push harder.'
              % len(n.erpm))
    else:
        mean_erpm = sum(moving) / len(moving)
        mean_vx = sum(speeds) / len(speeds) if speeds else 0.0
        print('eRPM while pushed forward: mean %+.0f over %d samples' % (mean_erpm, len(moving)))
        print('/odom twist.linear.x:      mean %+.3f m/s over %d samples' % (mean_vx, len(speeds)))
        if mean_vx > 0:
            print('odometry sign is correct with the driver installed now.')
            print('  after the Jazzy rebuild set vesc_to_odom_node speed_to_erpm_gain: %.1f' % -GAIN)
        else:
            print('odometry sign is inverted with the driver installed now.')
            print('  set vesc_to_odom_node speed_to_erpm_gain: %.1f now, and %.1f after the rebuild'
                  % (-GAIN, GAIN))
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
