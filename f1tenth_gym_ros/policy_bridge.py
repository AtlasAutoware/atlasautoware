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

Route-hint models (trained 9/25 on, ONNX config state_mask[2:4] = 1) also need a point
2 m ahead on the planned route in state[2:4]. For those the bridge plans the route itself:
map-frame pose from `pose_topic` (SLAM via pose_relay, or the particle filter), the map from
`map_topic`, the goal from `goal` ("x,y" in map metres), /goal_pose (RViz 2D Goal Pose) or
/policy/goal ("x,y"). A* (route_source.RouteSource, same planner the sim tasks used) runs
in a background thread every `replan` s. No map, goal, fresh pose or route -> zero speed;
inside `goal_tol` of the goal -> zero speed. Older models keep the IMU in those slots
(masked to zero inside the network) and ignore all of this.
"""
import json, math, os, threading, time
import numpy as np, cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String

try:                                                 # installed package
    from f1tenth_gym_ros.policy_io import (text_ids, bev_image, front_image, make_feed,
                                           action_order_of, split_action, front_clear)
    from f1tenth_gym_ros.route_source import RouteSource, model_uses_hint, occ_from_grid, parse_goal
except ImportError:                                  # run from a source checkout
    from policy_io import (text_ids, bev_image, front_image, make_feed,
                           action_order_of, split_action, front_clear)
    from route_source import RouteSource, model_uses_hint, occ_from_grid, parse_goal


class PolicyBridge(Node):
    def __init__(self):
        super().__init__('policy_bridge')
        P = (('model', 'models/student.onnx'), ('instruction', 'go straight to the end and stop'),
             ('image_topic', '/oakd/rgb'), ('scan_topic', '/scan'), ('odom_topic', '/odom'),
             ('imu_topic', '/oakd/imu'), ('drive_topic', '/drive'), ('rate', 10.0), ('max_speed', 1.0),
             ('max_steer', 0.4), ('aeb_dist', 0.35), ('stale', 0.5), ('threads', 4),
             ('pose_topic', '/pf/pose/odom'), ('map_topic', '/map'), ('goal', ''), ('replan', 1.0),
             ('goal_tol', 0.4), ('pose_stale', 0.5))
        for k, v in P: self.declare_parameter(k, v)
        g = lambda n: self.get_parameter(n).value
        self.p = {k: g(k) for k, _ in P}
        import onnxruntime as ort
        path = os.path.expanduser(self.p['model'])
        if not os.path.isabs(path):
            for base in (os.getcwd(), os.path.expanduser('~/atlas_ws/src/atlasautoware')):
                if os.path.isfile(os.path.join(base, path)): path = os.path.join(base, path); break
        # Four threads measured 1.6 ms per frame on the Orin Nano against 2.5 ms on two.
        # The GPU is not worth it here: a TensorRT engine would save under a millisecond
        # of a 100 ms control period, and the Orin's CUDA userspace is not installed.
        so = ort.SessionOptions(); so.intra_op_num_threads = int(self.p['threads'])
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
        # The output order comes from the model (metadata written by ml/train_policy.py; older
        # exports are (speed, steer)). Unpacking it as (steer, speed) was the 9/23 bug.
        self.order = action_order_of(self.sess)
        self.ids = np.asarray([text_ids(self.p['instruction'])], np.int64)
        self.front = None; self.scan = None; self.state = np.zeros(5, np.float32)
        self.t_img = self.t_scan = 0.0; self.front_clear = 99.0; self.n = 0; self.t0 = time.time()
        self.create_subscription(Image, self.p['image_topic'], self._img, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.p['scan_topic'], self._scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.p['odom_topic'], self._odom, 10)
        self.create_subscription(Imu, self.p['imu_topic'], self._imu, 20)
        self.create_subscription(String, '/policy/instruction', self._instr, 5)
        # route hint (state[2:4]) for models trained with it; see the module docstring
        self.use_hint = model_uses_hint(self.sess)
        self.route = RouteSource(goal_tol=float(self.p['goal_tol'])); self.pose = None; self.t_pose = 0.0
        if self.use_hint:
            self.route.set_goal(parse_goal(self.p['goal']))
            self.create_subscription(Odometry, self.p['pose_topic'], self._pose, 20)
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)       # slam_toolbox /map
            self.create_subscription(OccupancyGrid, self.p['map_topic'], self._map, latched)
            self.create_subscription(PoseStamped, '/goal_pose', self._goal_pose, 5)
            self.create_subscription(String, '/policy/goal', self._goal_str, 5)
            threading.Thread(target=self._replan_loop, daemon=True).start()
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.p['drive_topic'], 10)
        self.st_pub = self.create_publisher(String, '/policy/status', 5)
        self.create_timer(1.0 / float(self.p['rate']), self._tick)
        self.get_logger().info(f"policy_bridge: {path} | instruction: {self.p['instruction']!r} | "
                               f"max_speed {self.p['max_speed']} m/s | route hint "
                               f"{'ON, goal ' + str(self.route.goal) if self.use_hint else 'off (model not trained with it)'}")

    def _img(self, m):
        if m.encoding not in ('rgb8', 'bgr8'): return
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        if m.encoding == 'rgb8': a = a[:, :, ::-1]                   # training frames were BGR (cv2)
        self.front = front_image(a); self.t_img = time.time()

    def _scan(self, m):
        self.scan = (m.ranges, m.angle_min, m.angle_increment); self.t_scan = time.time()
        self.front_clear = front_clear(m.ranges, m.angle_min, m.angle_increment)

    def _odom(self, m): self.state[0] = m.twist.twist.linear.x; self.state[1] = m.twist.twist.angular.z
    def _imu(self, m):
        if self.use_hint: self.state[4] = m.angular_velocity.z                  # [2:4] is the route hint
        else: self.state[2:5] = (m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z)

    def _pose(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw); self.t_pose = time.time()

    def _map(self, m):
        i = m.info
        self.route.set_map(occ_from_grid(m.data, i.width, i.height), i.resolution,
                           (i.origin.position.x, i.origin.position.y))

    def _set_goal(self, g, src):
        self.route.set_goal(g); self.get_logger().info(f'goal ({src}): {g}')

    def _goal_pose(self, m): self._set_goal((m.pose.position.x, m.pose.position.y), '/goal_pose')
    def _goal_str(self, m): self._set_goal(parse_goal(m.data), '/policy/goal')

    def _fresh_pose(self):
        return self.pose if time.time() - self.t_pose < self.p['pose_stale'] else None

    def _replan_loop(self):
        # A* can take tens of ms on a big SLAM map; keep it off the control timer
        while rclpy.ok():
            try: self.route.replan(self._fresh_pose(), time.time())
            except Exception as e: self.get_logger().warn(f'replan: {e}', throttle_duration_sec=5.0)
            time.sleep(max(0.1, float(self.p['replan'])))
    def _instr(self, m):
        self.p['instruction'] = m.data; self.ids = np.asarray([text_ids(m.data)], np.int64)
        self.get_logger().info(f'instruction: {m.data!r}')

    def _tick(self):
        now = time.time(); cmd = AckermannDriveStamped(); st = {'instruction': self.p['instruction']}
        fresh = self.front is not None and self.scan is not None and now - self.t_img < self.p['stale'] and now - self.t_scan < self.p['stale']
        hold = False
        if fresh and self.use_hint:
            h, rst = self.route.hint(self._fresh_pose(), now); st.update(rst)
            if h is None: hold = True                                # no route / arrived: don't drive
            else: self.state[2:4] = h; st['hint'] = [round(h[0], 2), round(h[1], 2)]
        if fresh and not hold:
            bev = bev_image(*self.scan)
            # channel order matches training: frames were cached from cv2 (BGR), fed unflipped
            speed, steer = split_action(self.sess.run(['action'], make_feed(self.front, bev, self.state, self.ids))[0][0],
                                        self.order)
            # before the clamp: min()/max() silently turn a NaN into a limit (full lock)
            if not (math.isfinite(speed) and math.isfinite(steer)): speed, steer = 0.0, 0.0; st['nan'] = True
            steer = max(-self.p['max_steer'], min(self.p['max_steer'], steer))
            speed = max(0.0, min(self.p['max_speed'], speed))
            if self.front_clear < self.p['aeb_dist']: speed = 0.0; st['aeb'] = True
            cmd.drive.speed = speed; cmd.drive.steering_angle = steer
            st.update({'steer': round(steer, 3), 'speed': round(speed, 2), 'front': round(self.front_clear, 2)})
            self.n += 1
        elif not fresh:
            st['waiting'] = {'image': now - self.t_img > self.p['stale'], 'scan': now - self.t_scan > self.p['stale']}
        self.drive_pub.publish(cmd)
        st['hz'] = round(self.n / max(1e-6, now - self.t0), 1)
        self.st_pub.publish(String(data=json.dumps(st)))


def main(args=None):
    try:
        from rclpy.executors import ExternalShutdownException
    except ImportError:
        ExternalShutdownException = KeyboardInterrupt
    rclpy.init(args=args); n = PolicyBridge()
    try: rclpy.spin(n)
    except (KeyboardInterrupt, ExternalShutdownException): pass
    finally:
        # leave the actuator an explicit zero instead of the last policy command; drive_node's
        # own timeout is the backstop, this just makes a stop immediate
        try:
            if rclpy.ok(): n.drive_pub.publish(AckermannDriveStamped())
        except Exception: pass
    try: n.destroy_node(); rclpy.shutdown()
    except Exception: pass


if __name__ == '__main__':
    main()
