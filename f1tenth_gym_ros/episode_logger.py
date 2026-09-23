#!/usr/bin/env python3
"""episode_logger: record demonstration episodes for policy training.

Runs alongside remote-pilot mode. Control is over two String topics carrying JSON so
the web page (through web_pilot) or any script can drive it without services:

    /episode/cmd     {"action":"start","instruction":"..."}
                     {"action":"stop","label":"good"|"bad"|"...", "discard":false}
    /episode/status  {"recording":..., "episode_id":..., "frames":..., "ages":{...}}  at 2 Hz

Frames are taken from the camera topic at `fps` (default 10, decimated from 15 or 30).
The action that gets logged is whatever reached the car's Ackermann chain: /teleop
(human) or /drive (policy), whichever is newest, so a policy trained on it imitates
what actually drove the car. Raw /joy axes are kept too. Nothing here commands the car.
"""
import json, math, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, Joy, LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String
try:
    from f1tenth_gym_ros.episode_writer import EpisodeWriter, list_episodes
except ImportError:
    from episode_writer import EpisodeWriter, list_episodes


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class EpisodeLogger(Node):
    def __init__(self):
        super().__init__('episode_logger')
        P = (('root', '~/episodes'), ('fps', 10.0), ('image_topic', '/camera/color/image_raw'),
             ('scan_topic', '/scan'), ('imu_topic', '/oakd/imu'), ('odom_topic', '/vesc/odom'),
             ('teleop_topic', '/teleop'), ('drive_topic', '/drive'), ('joy_topic', '/joy'),
             ('wheelbase', 0.33), ('max_steer', 0.41))
        for k, v in P: self.declare_parameter(k, v)
        p = lambda n: self.get_parameter(n).value
        car = {'wheelbase': float(p('wheelbase')), 'max_steer': float(p('max_steer')),
               'image_topic': p('image_topic'), 'odom_topic': p('odom_topic'), 'imu_topic': p('imu_topic')}
        self.w = EpisodeWriter(p('root'), car)
        self.period = 1.0 / float(p('fps')); self.last_frame = 0.0
        self.create_subscription(Image, p('image_topic'), self._img, qos_profile_sensor_data)
        self.create_subscription(LaserScan, p('scan_topic'), self._scan, qos_profile_sensor_data)
        self.create_subscription(Imu, p('imu_topic'), self._imu, 50)
        self.create_subscription(Odometry, p('odom_topic'), self._odom, 20)
        self.create_subscription(AckermannDriveStamped, p('teleop_topic'), lambda m: self._ack(m, 0), 10)
        self.create_subscription(AckermannDriveStamped, p('drive_topic'), lambda m: self._ack(m, 1), 10)
        self.create_subscription(Joy, p('joy_topic'), self._joy, 10)
        self.create_subscription(String, '/episode/cmd', self._cmd, 10)
        try:
            from vesc_msgs.msg import VescStateStamped
            self.create_subscription(VescStateStamped, '/sensors/core',
                                     lambda m: self.w.update('volts', float(m.state.voltage_input)), 10)
        except Exception:
            pass
        self.status_pub = self.create_publisher(String, '/episode/status', 10)
        self.create_timer(0.5, self._status)
        self.get_logger().info(f"episode logger ready: {self.w.root} at {p('fps')} fps; "
                               f"send {{\"action\":\"start\",\"instruction\":\"...\"}} on /episode/cmd")

    @staticmethod
    def _t(msg):
        s = msg.header.stamp
        return s.sec + s.nanosec * 1e-9 if (s.sec or s.nanosec) else time.time()

    def _img(self, m):
        if not self.w.recording(): return
        now = time.time()
        if now - self.last_frame < self.period * 0.95: return
        self.last_frame = now
        if m.encoding not in ('rgb8', 'bgr8'): return
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        if m.encoding == 'rgb8': a = a[:, :, ::-1]
        self.w.add_frame(np.ascontiguousarray(a), self._t(m))

    def _scan(self, m):
        r = np.asarray(m.ranges, np.float32)
        r[~np.isfinite(r)] = 0.0
        self.w.update('scan', r, self._t(m))

    def _imu(self, m):
        self.w.update('imu', (m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z,
                              m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z), self._t(m))

    def _odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        t = self._t(m)
        self.w.update('pose', (p.x, p.y, _yaw(q)))
        self.w.update('vel', (m.twist.twist.linear.x, m.twist.twist.angular.z))
        self.w.stamp('odom', t)

    def _ack(self, m, src):
        self.w.set_action(m.drive.speed, m.drive.steering_angle, src, self._t(m))

    def _joy(self, m):
        if len(m.axes) > 3:
            self.w.update('joy', (m.axes[1], m.axes[3]), self._t(m))

    def _cmd(self, m):
        try: d = json.loads(m.data)
        except ValueError:
            self.get_logger().warn(f'bad /episode/cmd: {m.data[:80]}'); return
        a = d.get('action')
        if a == 'start':
            ok, msg = self.w.start(d.get('instruction', ''))
        elif a == 'stop':
            ok, msg = self.w.stop(d.get('label', 'unlabelled'), bool(d.get('discard', False)))
        else:
            ok, msg = False, f'unknown action {a!r}'
        (self.get_logger().info if ok else self.get_logger().warn)(f'{a}: {msg}')
        self._status()

    def _status(self):
        st = self.w.status(); st['episodes'] = len(list_episodes(self.w.root)); st['root'] = self.w.root
        self.status_pub.publish(String(data=json.dumps(st)))


def main(args=None):
    rclpy.init(args=args); n = EpisodeLogger()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    if n.w.recording(): n.w.stop('interrupted')
    n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
