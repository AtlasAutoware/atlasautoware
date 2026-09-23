#!/usr/bin/env python3
"""sim_env: a closed-loop 1/10 driving simulator that speaks the car's interface.

It publishes exactly what a policy sees on the real car and consumes exactly what a
policy emits, so the SAME nodes drive it: raceline_mpc, the web pilot, or a policy
server / Qwen-Drive bridge. That makes it the test bench for the driving policy --
run the policy against sim_env, watch it in the browser, and read the score -- and a
data source: drive it and the episode logger records sim episodes in the car's format.

    publishes   /scan (LaserScan)  /odom (Odometry, gt)  /pf/pose/odom (map, gt)
                /camera/color/image_raw (rgb8, synthetic)  /sim/state (String json)
                tf: map->odom (identity), odom->base_link
    consumes    /teleop, /drive (AckermannDriveStamped; teleop wins when fresh, as on the car)

Metrics on /sim/state: collisions, distance travelled, distance to goal, progress %,
and `reached` when within goal_radius -- a closed-loop success signal for policy eval.
"""
import json, math, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
try:
    from f1tenth_gym_ros.sim_core import SimMap, bicycle_step, render_fpv, load_map
except ImportError:
    from sim_core import SimMap, bicycle_step, render_fpv, load_map


def quat(yaw): return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


class SimEnv(Node):
    def __init__(self):
        super().__init__('sim_env')
        P = (('map_yaml', ''), ('start_x', 0.0), ('start_y', 0.0), ('start_theta', 0.0),
             ('goal_x', 0.0), ('goal_y', 0.0), ('goal_radius', 0.5), ('use_goal', False),
             ('wheelbase', 0.33), ('max_steer', 0.41), ('phys_hz', 50.0),
             ('scan_hz', 10.0), ('scan_beams', 540), ('scan_fov', 2 * math.pi),
             ('max_range', 16.0), ('scan_noise', 0.01), ('cmd_timeout', 0.4),
             ('camera', True), ('cam_hz', 15.0), ('cam_w', 640), ('cam_h', 480), ('cam_fx', 460.5))
        for k, v in P: self.declare_parameter(k, v)
        g = lambda n: self.get_parameter(n).value
        occ, res, origin = load_map(g('map_yaml'))
        self.map = SimMap(occ, res, origin)
        self.L = float(g('wheelbase')); self.max_steer = float(g('max_steer'))
        self.state = np.array([float(g('start_x')), float(g('start_y')), float(g('start_theta')), 0.0])
        self.start = self.state[:2].copy()
        self.goal = np.array([float(g('goal_x')), float(g('goal_y'))]); self.use_goal = bool(g('use_goal'))
        self.goal_r = float(g('goal_radius'))
        self.cmd_timeout = float(g('cmd_timeout'))
        self.max_range = float(g('max_range')); self.scan_noise = float(g('scan_noise'))
        nb = int(g('scan_beams')); fov = float(g('scan_fov'))
        self.angles = -fov / 2 + fov * np.arange(nb) / nb
        self.angle_min = float(self.angles[0]); self.angle_inc = float(fov / nb)
        self.teleop = None; self.teleop_t = 0.0; self.drive = None; self.drive_t = 0.0
        self.collisions = 0; self.dist = 0.0; self.min_goal = 1e9; self.reached = False
        self.create_subscription(AckermannDriveStamped, '/teleop', self._teleop, 10)
        self.create_subscription(AckermannDriveStamped, '/drive', self._drive, 10)
        self.create_subscription(String, '/sim/reset', self._reset_cmd, 5)
        self.create_subscription(String, '/sim/goal', self._goal_cmd, 5)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.pf_pub = self.create_publisher(Odometry, '/pf/pose/odom', 10)
        self.state_pub = self.create_publisher(String, '/sim/state', 5)
        self.tfb = TransformBroadcaster(self)
        self.cam_on = bool(g('camera'))
        if self.cam_on:
            self.cam_pub = self.create_publisher(Image, '/camera/color/image_raw', qos_profile_sensor_data)
            self.cam = (int(g('cam_w')), int(g('cam_h')), float(g('cam_fx')))
        self.dt = 1.0 / float(g('phys_hz'))
        self.create_timer(self.dt, self._phys)
        self.create_timer(1.0 / float(g('scan_hz')), self._scan)
        self.create_timer(1.0 / float(g('cam_hz')), self._camera) if self.cam_on else None
        self.create_timer(0.2, self._state)
        self.get_logger().info(f"sim_env: map {g('map_yaml')} ({occ.shape}), start "
                               f"({self.state[0]:.1f},{self.state[1]:.1f}); drive it with /teleop or /drive")

    def _teleop(self, m): self.teleop = m.drive; self.teleop_t = time.time()
    def _drive(self, m): self.drive = m.drive; self.drive_t = time.time()

    def _reset_cmd(self, m):
        try: d = json.loads(m.data)
        except ValueError: d = {}
        self.state = np.array([d.get('x', self.start[0]), d.get('y', self.start[1]), d.get('theta', 0.0), 0.0])
        self.start = self.state[:2].copy()
        self.collisions = 0; self.dist = 0.0; self.min_goal = 1e9; self.reached = False
        self.get_logger().info('sim reset')

    def _goal_cmd(self, m):
        """{"x":..,"y":..,"radius":..} sets the goal for the success metric (or {"clear":true})."""
        try: d = json.loads(m.data)
        except ValueError: return
        if d.get('clear'):
            self.use_goal = False; self.reached = False; return
        self.goal = np.array([float(d['x']), float(d['y'])]); self.goal_r = float(d.get('radius', self.goal_r))
        self.use_goal = True; self.reached = False; self.min_goal = 1e9

    def _cmd(self):
        now = time.time()
        if self.teleop is not None and now - self.teleop_t < self.cmd_timeout:
            return self.teleop.speed, self.teleop.steering_angle          # teleop wins (mux priority)
        if self.drive is not None and now - self.drive_t < self.cmd_timeout:
            return self.drive.speed, self.drive.steering_angle
        return 0.0, 0.0

    def _phys(self):
        speed, steer = self._cmd()
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        nxt = bicycle_step(self.state, speed, steer, self.L, self.dt)
        if self.map.occupied(nxt[0], nxt[1]):                            # crude collision: stop, hold
            if self.state[3] > 0.3: self.collisions += 1
            nxt[0], nxt[1], nxt[3] = self.state[0], self.state[1], 0.0
        self.dist += math.hypot(nxt[0] - self.state[0], nxt[1] - self.state[1])
        self.state = nxt
        if self.use_goal:
            dg = math.hypot(self.state[0] - self.goal[0], self.state[1] - self.goal[1])
            self.min_goal = min(self.min_goal, dg)
            if dg < self.goal_r: self.reached = True
        self._publish_odom()

    def _publish_odom(self):
        now = self.get_clock().now().to_msg()
        x, y, th, v = self.state
        for pub, frame in ((self.odom_pub, 'odom'), (self.pf_pub, 'map')):
            o = Odometry(); o.header.stamp = now; o.header.frame_id = frame; o.child_frame_id = 'base_link'
            o.pose.pose.position.x = float(x); o.pose.pose.position.y = float(y)
            q = quat(th); o.pose.pose.orientation.z = q[2]; o.pose.pose.orientation.w = q[3]
            o.twist.twist.linear.x = float(v)
            pub.publish(o)
        # tf: map->odom identity, odom->base_link = pose
        for parent, child, (px, py, pth) in (('map', 'odom', (0, 0, 0)), ('odom', 'base_link', (x, y, th))):
            t = TransformStamped(); t.header.stamp = now; t.header.frame_id = parent; t.child_frame_id = child
            t.transform.translation.x = float(px); t.transform.translation.y = float(py)
            q = quat(pth); t.transform.rotation.z = q[2]; t.transform.rotation.w = q[3]
            self.tfb.sendTransform(t)

    def _scan(self):
        x, y, th, _ = self.state
        r = self.map.raycast(x, y, th + self.angles, self.max_range)
        if self.scan_noise: r = r + np.random.normal(0, self.scan_noise, len(r)).astype(np.float32)
        m = LaserScan(); m.header.stamp = self.get_clock().now().to_msg(); m.header.frame_id = 'laser'
        m.angle_min = self.angle_min; m.angle_increment = self.angle_inc
        m.angle_max = self.angle_min + self.angle_inc * len(r)
        m.range_min = 0.05; m.range_max = self.max_range
        m.ranges = [float(v) for v in np.clip(r, 0.0, self.max_range)]
        self.scan_pub.publish(m)

    def _camera(self):
        x, y, th, _ = self.state
        w, h, fx = self.cam
        img = render_fpv(self.map, x, y, th, w, h, fx, max_range=min(10.0, self.max_range))
        m = Image(); m.header.stamp = self.get_clock().now().to_msg(); m.header.frame_id = 'camera_link'
        m.height = h; m.width = w; m.encoding = 'rgb8'; m.is_bigendian = 0; m.step = w * 3
        m.data = img.tobytes()
        self.cam_pub.publish(m)

    def _state(self):
        s = {'x': round(float(self.state[0]), 3), 'y': round(float(self.state[1]), 3),
             'theta': round(float(self.state[2]), 3), 'speed': round(float(self.state[3]), 2),
             'collisions': self.collisions, 'distance_m': round(self.dist, 2),
             'sim': True}
        if self.use_goal:
            dg = math.hypot(self.state[0] - self.goal[0], self.state[1] - self.goal[1])
            s.update({'goal': [float(self.goal[0]), float(self.goal[1])], 'dist_to_goal': round(dg, 2),
                      'reached': self.reached, 'min_dist_to_goal': round(self.min_goal, 2)})
        self.state_pub.publish(String(data=json.dumps(s)))


def main(args=None):
    rclpy.init(args=args); n = SimEnv()
    try: rclpy.spin(n)                              # single-threaded: no races; render is vectorized/cheap
    except KeyboardInterrupt: pass
    try: n.destroy_node(); rclpy.shutdown()
    except Exception: pass


if __name__ == '__main__':
    main()
