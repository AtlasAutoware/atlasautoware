#!/usr/bin/env python3
"""policy_bridge: run the distilled goal-conditioned student on the car (mode 3).

Loads models/student.onnx (from ml/train_student.py on the cluster), builds the same
inputs it was trained on -- front frame 96x128, lidar as a 96x96 bird's-eye image,
state (vx, wz, gx, gy, gz), hashed bag-of-words instruction -- and publishes
AckermannDrive on /drive at `rate` Hz. That is the same channel raceline_mpc uses, so
the mux, the human override, the pilot page's STOP/heartbeat, and the VESC timeout all
apply unchanged. A forward-cone emergency brake from the scan sits in front of the
policy, and the speed is clamped to `max_speed`.

    ros2 run f1tenth_gym_ros policy_bridge --ros-args -p model:=models/student.onnx \
        -p instruction:="turn left, then go straight to the end and stop"

Instruction can also be changed live on /policy/instruction (std_msgs/String).
"""
import hashlib, json, math, os, time
import numpy as np, cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String

FRONT_HW, BEV_HW, MAX_TOK, BEV_EXTENT = (96, 128), (96, 96), 24, 6.0


def text_ids(s, n=4096, max_tok=MAX_TOK):           # keep identical to ml/train_student.py
    toks = ''.join(c if c.isalnum() else ' ' for c in s.lower()).split()[:max_tok]
    ids = [1 + int(hashlib.md5(t.encode()).hexdigest(), 16) % (n - 1) for t in toks]
    return ids + [0] * (max_tok - len(ids))


def bev_image(ranges, angle_min, angle_inc, size=BEV_HW[0], extent=BEV_EXTENT):  # same as episodes_to_lerobot
    img = np.zeros((size, size), np.uint8)
    r = np.asarray(ranges, np.float32); n = len(r)
    ang = angle_min + angle_inc * np.arange(n)
    ok = np.isfinite(r) & (r > 0.05) & (r < extent)
    x, y = r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])
    px = (size / 2 - x / extent * size / 2).astype(int); py = (size / 2 - y / extent * size / 2).astype(int)
    m = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    img[px[m], py[m]] = 255
    img[size // 2 - 1:size // 2 + 2, size // 2 - 1:size // 2 + 2] = 128
    return img


class PolicyBridge(Node):
    def __init__(self):
        super().__init__('policy_bridge')
        P = (('model', 'models/student.onnx'), ('instruction', 'go straight to the end and stop'),
             ('image_topic', '/camera/color/image_raw'), ('scan_topic', '/scan'), ('odom_topic', '/odom'),
             ('imu_topic', '/oakd/imu'), ('drive_topic', '/drive'), ('rate', 10.0), ('max_speed', 1.0),
             ('max_steer', 0.4), ('aeb_dist', 0.35), ('stale', 0.5), ('threads', 2))
        for k, v in P: self.declare_parameter(k, v)
        g = lambda n: self.get_parameter(n).value
        self.p = {k: g(k) for k, _ in P}
        import onnxruntime as ort
        path = os.path.expanduser(self.p['model'])
        if not os.path.isabs(path):
            for base in (os.getcwd(), os.path.expanduser('~/atlas_ws/src/atlasautoware')):
                if os.path.isfile(os.path.join(base, path)): path = os.path.join(base, path); break
        so = ort.SessionOptions(); so.intra_op_num_threads = int(self.p['threads'])
        self.sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
        self.ids = np.asarray([text_ids(self.p['instruction'])], np.int64)
        self.front = None; self.scan = None; self.state = np.zeros(5, np.float32)
        self.t_img = self.t_scan = 0.0; self.front_clear = 99.0; self.n = 0; self.t0 = time.time()
        self.create_subscription(Image, self.p['image_topic'], self._img, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.p['scan_topic'], self._scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.p['odom_topic'], self._odom, 10)
        self.create_subscription(Imu, self.p['imu_topic'], self._imu, 20)
        self.create_subscription(String, '/policy/instruction', self._instr, 5)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.p['drive_topic'], 10)
        self.st_pub = self.create_publisher(String, '/policy/status', 5)
        self.create_timer(1.0 / float(self.p['rate']), self._tick)
        self.get_logger().info(f"policy_bridge: {path} | instruction: {self.p['instruction']!r} | "
                               f"max_speed {self.p['max_speed']} m/s")

    def _img(self, m):
        if m.encoding not in ('rgb8', 'bgr8'): return
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        if m.encoding == 'rgb8': a = a[:, :, ::-1]                   # training frames were BGR (cv2)
        self.front = cv2.resize(a, (FRONT_HW[1], FRONT_HW[0]), interpolation=cv2.INTER_AREA); self.t_img = time.time()

    def _scan(self, m):
        self.scan = (m.ranges, m.angle_min, m.angle_increment); self.t_scan = time.time()
        n = len(m.ranges); c = n // 2 if m.angle_min < -1.0 else 0; w = max(1, int(0.2 / max(m.angle_increment, 1e-6)))
        seg = np.asarray(m.ranges[max(0, c - w):c + w], np.float32); seg = seg[np.isfinite(seg) & (seg > 0.05)]
        self.front_clear = float(seg.min()) if seg.size else 99.0

    def _odom(self, m): self.state[0] = m.twist.twist.linear.x; self.state[1] = m.twist.twist.angular.z
    def _imu(self, m): self.state[2:5] = (m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z)
    def _instr(self, m):
        self.p['instruction'] = m.data; self.ids = np.asarray([text_ids(m.data)], np.int64)
        self.get_logger().info(f'instruction: {m.data!r}')

    def _tick(self):
        now = time.time(); cmd = AckermannDriveStamped(); st = {'instruction': self.p['instruction']}
        fresh = self.front is not None and self.scan is not None and now - self.t_img < self.p['stale'] and now - self.t_scan < self.p['stale']
        if fresh:
            bev = bev_image(*self.scan)
            # channel order matches training: frames were cached from cv2 (BGR), fed unflipped
            feed = {'front': (self.front.transpose(2, 0, 1)[None] / 255.0).astype(np.float32),
                    'bev': (bev[None, None] / 255.0).astype(np.float32),
                    'state': self.state[None].astype(np.float32), 'ids': self.ids}
            steer, speed = [float(v) for v in self.sess.run(['action'], feed)[0][0]]
            steer = max(-self.p['max_steer'], min(self.p['max_steer'], steer))
            speed = max(0.0, min(self.p['max_speed'], speed))
            if self.front_clear < self.p['aeb_dist']: speed = 0.0; st['aeb'] = True
            cmd.drive.speed = speed; cmd.drive.steering_angle = steer
            st.update({'steer': round(steer, 3), 'speed': round(speed, 2), 'front': round(self.front_clear, 2)})
            self.n += 1
        else:
            st['waiting'] = {'image': now - self.t_img > self.p['stale'], 'scan': now - self.t_scan > self.p['stale']}
        self.drive_pub.publish(cmd)
        st['hz'] = round(self.n / max(1e-6, now - self.t0), 1)
        self.st_pub.publish(String(data=json.dumps(st)))


def main(args=None):
    rclpy.init(args=args); n = PolicyBridge()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    try: n.destroy_node(); rclpy.shutdown()
    except Exception: pass


if __name__ == '__main__':
    main()
