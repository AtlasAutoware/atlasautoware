from rclpy.qos import qos_profile_sensor_data
"""
Raceline-MPC racing agent — the clean competition car.
=======================================================

The minimal, bulletproof deployment: load the optimized raceline, track it with
the kinematic MPC, brake for anything genuinely in the way, drive.  With
`avoid_opponents` on (the multi-car RoboRacer head-to-head format, up to 4 cars
on track) an avoidance layer shifts the tracked line to pass / cover / give
room and follows a slower car at a gap instead of tripping the AEB.

  best_raceline.csv ──► MPC (track, ± lateral offset) ──► /drive
        localization ──┘        lidar AEB ──┘  ▲
        lidar ──► opponent detector + strategist + follow governor ──┘

Design choices that matter for a race:
  - **MPC with an automatic fallback.**  If osqp is missing or a solve fails,
    the loop transparently falls back to the MAP controller (model- and
    acceleration-based pursuit, Becker et al. ICRA 2023) that tick, so the car
    never stalls on a solver hiccup.  The strategic offset is applied to both.
  - **Friction-limited speeds on demand.**  `reprofile_speeds` replaces the
    raceline CSV's speed column at load with the TUMFTM forward-backward
    profile (lateral budget + friction-ellipse-coupled accel/brake limits).
  - **Avoidance is a reference shaper, not a driver.**  The opponent layer only
    changes the line the controller tracks and caps the speed; AEB, the sensor
    watchdog and the traction governor stay in front of the actuator untouched.
  - **Hardware-portable.**  Every sim-only topic/frame is a ROS parameter; the
    only thing that differs on the real car is `odom_topic` (the pose source from
    your localization — a particle filter publishing map-relative odometry, NOT
    raw drifting VESC odom).
  - **Light control loop.**  Opponent clustering runs once per scan (10 Hz),
    not per control tick (50 Hz); no disk I/O, no heavy perception.

Run:
    # sim
    ros2 run f1tenth_gym_ros raceline_mpc
    # real car (pose from your localization, e.g. the particle filter)
    ros2 run f1tenth_gym_ros raceline_mpc --ros-args -p odom_topic:=/pf/pose/odom
"""

import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
try:                                   # telemetry types — only needed with avoidance on
    from geometry_msgs.msg import PoseArray, Pose
    from std_msgs.msg import String
except ImportError:                    # ROS-free unit tests stub only the core msgs
    PoseArray = Pose = String = None
from transforms3d.euler import quat2euler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pursuit_agent import find_best_raceline, load_raceline, find_nearest
from mpc_controller import KinematicMPC, TractionGovernor, predict_state
from map_controller import MAPController
from velocity_profiler import velocity_profile, segment_lengths
from raceline_refiner import refine_raceline
from drive_safety import forward_clearance, stopping_distance
from opponent_avoidance import AvoidanceLayer


class RacelineMPC(Node):
    def __init__(self):
        super().__init__('raceline_mpc')

        # ── parameters (sim defaults; override on hardware) ────────────────────
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('raceline', '')          # explicit CSV; '' = auto-find
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('max_steer', 0.41)
        self.declare_parameter('steer_offset', 0.0)    # constant steering bias (rad); 0 = none.
                                                       # a nonzero bias steals throw on one side;
                                                       # fix a pull at the servo center instead.
        self.declare_parameter('control_hz', 50.0)
        self.declare_parameter('sensor_timeout', 0.5)
        self.declare_parameter('v_scale', 1.0)          # global speed cap (start low!)
        self.declare_parameter('aeb_dist', 0.45)        # m, hard stop if wall/obstacle closer
        self.declare_parameter('aeb_cone', 0.20)        # rad (~11deg) forward cone
        self.declare_parameter('aeb_decel', 6.0)        # m/s^2 — extends aeb_dist with speed
        self.declare_parameter('min_speed', 0.6)        # m/s creep floor when rolling
        self.declare_parameter('imu_topic', '')         # e.g. /oakd/imu; '' = off (sim)
        self.declare_parameter('max_lat_accel', 6.0)    # m/s^2 traction-governor limit
        self.declare_parameter('actuation_delay', 0.0)  # s sensor->actuator latency
        self.declare_parameter('refine_corridor', 0.0)  # m min-curvature refinement
        self.declare_parameter('reprofile_speeds', False)  # recompute CSV speeds
        self.declare_parameter('profile_a_accel', 4.0)  # m/s^2 engine limit
        self.declare_parameter('profile_a_brake', 8.0)  # m/s^2 braking limit
        self.declare_parameter('profile_v_max', 8.0)    # m/s profile ceiling
        # ── multi-car avoidance (RoboRacer head-to-head, up to 4 cars) ────────
        self.declare_parameter('avoid_opponents', False)
        self.declare_parameter('avoid_max_offset', 0.5)     # m max move off the raceline
        self.declare_parameter('avoid_offset_rate', 0.04)   # m per control tick (2 m/s @50Hz)
        self.declare_parameter('avoid_side_clearance', 0.55)  # m beside an opponent
        self.declare_parameter('avoid_attack_range', 6.0)   # m, engage a car this far ahead
        self.declare_parameter('avoid_defend_range', 5.0)
        self.declare_parameter('avoid_contact_range', 1.0)  # m, "alongside" bubble
        self.declare_parameter('avoid_follow_gap', 1.0)     # m standing gap behind a car
        self.declare_parameter('avoid_time_headway', 0.6)   # s, gap grows with speed
        self.declare_parameter('avoid_follow_gain', 1.5)    # 1/s
        self.declare_parameter('avoid_follow_cone', 0.5)    # rad, travel cone half-angle
        self.declare_parameter('avoid_min_follow_speed', 0.0)
        self.declare_parameter('avoid_allow_boost', False)  # let ATTACK exceed raceline speed
        self.declare_parameter('avoid_max_range', 8.0)      # m, detector range
        self.declare_parameter('avoid_wall_margin', 0.20)   # m kept from measured walls
        self.declare_parameter('avoid_publish_hz', 10.0)    # /opponents + /avoidance_state
        scan_topic  = self.get_parameter('scan_topic').value
        odom_topic  = self.get_parameter('odom_topic').value
        drive_topic = self.get_parameter('drive_topic').value
        self.L          = float(self.get_parameter('wheelbase').value)
        self.max_steer  = float(self.get_parameter('max_steer').value)
        self.steer_offset = float(self.get_parameter('steer_offset').value)
        self.v_scale    = float(self.get_parameter('v_scale').value)
        self.aeb_dist   = float(self.get_parameter('aeb_dist').value)
        self.aeb_cone   = float(self.get_parameter('aeb_cone').value)
        self.aeb_decel  = float(self.get_parameter('aeb_decel').value)
        self.min_speed  = float(self.get_parameter('min_speed').value)
        self.delay      = float(self.get_parameter('actuation_delay').value)
        self._last_cmd  = (0.0, 0.0)                    # published (steer, speed)
        # commands still in flight toward the actuator, oldest first — one per
        # control tick over the delay window.  predict_state must integrate
        # through THESE (the actuator executes the previously issued commands
        # during the latency window, not the newest one).
        hz = float(self.get_parameter('control_hz').value)
        self._delay_ticks = int(round(self.delay * hz))
        self._cmd_buf = [(0.0, 0.0)] * self._delay_ticks

        # ── raceline ───────────────────────────────────────────────────────────
        rl = self.get_parameter('raceline').value or self._find_raceline()
        if not rl or not os.path.exists(rl):
            self.get_logger().error('No raceline CSV found — run the optimizer first.')
            raise FileNotFoundError('no raceline')
        self.rl_x, self.rl_y, self.rl_hdg, self.rl_curv, self.rl_speed = load_raceline(rl)
        self.n = len(self.rl_x)
        corridor = float(self.get_parameter('refine_corridor').value)
        if corridor > 0.0:
            # minimum-curvature refinement (TUMFTM) within +/- corridor of the
            # loaded line — validate wall clearance before enabling on a car
            self.rl_x, self.rl_y, self.rl_hdg, self.rl_curv = refine_raceline(
                self.rl_x, self.rl_y, corridor=corridor)
            self.get_logger().info(
                f'raceline refined (min-curvature, corridor {corridor:.2f} m)')
        if self.get_parameter('reprofile_speeds').value:
            # friction-limited forward-backward profile (TUMFTM) — replaces the
            # CSV speed column with one that provably fits the grip budget
            self.rl_speed = velocity_profile(
                self.rl_curv, segment_lengths(self.rl_x, self.rl_y),
                a_lat_max=float(self.get_parameter('max_lat_accel').value),
                a_accel_max=float(self.get_parameter('profile_a_accel').value),
                a_brake_max=float(self.get_parameter('profile_a_brake').value),
                v_max=float(self.get_parameter('profile_v_max').value))
            self.get_logger().info(
                f'speeds reprofiled (friction-limited): '
                f'{self.rl_speed.min():.1f}-{self.rl_speed.max():.1f} m/s')
        self.v_max = float(self.rl_speed.max())
        self.get_logger().info(
            f'raceline: {self.n} pts, v {self.rl_speed.min():.1f}-{self.v_max:.1f} m/s '
            f'(x{self.v_scale:.2f}) from {os.path.basename(rl)}')
        # left normals for applying the strategic offset to the MAP fallback
        self._nx, self._ny = self._left_normals(self.rl_x, self.rl_y)

        # ── MPC (with pure-pursuit fallback) ───────────────────────────────────
        self.mpc = KinematicMPC(wheelbase=self.L, max_steer=self.max_steer,
                                v_max=self.v_max + 0.5)
        if self.mpc.available:
            self.mpc.set_raceline(self.rl_x, self.rl_y, self.rl_hdg,
                                  self.rl_curv, self.rl_speed)
            self.get_logger().info('controller: MPC (kinematic LTV, osqp)')
        else:
            self.get_logger().warning(
                'osqp not available — using MAP fallback full-time '
                '(pip install osqp==0.6.3 on the car to enable MPC)')

        # ── MAP fallback (Becker et al., ICRA 2023) — replaces pure pursuit ───
        self.map_ctl = MAPController(wheelbase=self.L, max_steer=self.max_steer)
        self.map_ctl.set_raceline(self.rl_x, self.rl_y, self.rl_speed,
                                  curvature=self.rl_curv)
        self.get_logger().info('fallback: MAP (model- and acceleration-based pursuit)')

        # ── traction governor (IMU; inert until imu_topic is set) ─────────────
        self.governor = TractionGovernor(
            max_lat_accel=float(self.get_parameter('max_lat_accel').value))
        self.yaw_rate = 0.0
        imu_topic = self.get_parameter('imu_topic').value
        if imu_topic:
            self.create_subscription(Imu, imu_topic, self._imu_cb, 10)
            self.get_logger().info(f'traction governor on (imu={imu_topic})')

        # ── multi-car avoidance layer ─────────────────────────────────────────
        self.avoid = None
        self._avoid_pub_dt = 0.0
        self._avoid_last_pub = 0.0
        if bool(self.get_parameter('avoid_opponents').value):
            g = lambda k: self.get_parameter(k).value      # noqa: E731
            self.avoid = AvoidanceLayer(
                max_offset=float(g('avoid_max_offset')),
                offset_rate=float(g('avoid_offset_rate')),
                side_clearance=float(g('avoid_side_clearance')),
                attack_range=float(g('avoid_attack_range')),
                defend_range=float(g('avoid_defend_range')),
                contact_range=float(g('avoid_contact_range')),
                follow_gap=float(g('avoid_follow_gap')),
                time_headway=float(g('avoid_time_headway')),
                follow_gain=float(g('avoid_follow_gain')),
                follow_cone=float(g('avoid_follow_cone')),
                min_follow_speed=float(g('avoid_min_follow_speed')),
                allow_boost=bool(g('avoid_allow_boost')),
                max_range=float(g('avoid_max_range')),
                wall_margin=float(g('avoid_wall_margin')))
            pub_hz = float(g('avoid_publish_hz'))
            self._avoid_pub_dt = 1.0 / pub_hz if pub_hz > 0 else float('inf')
            self.opp_pub = self.create_publisher(PoseArray, '/opponents', 5)
            self.state_pub = self.create_publisher(String, '/avoidance_state', 5)
            self.get_logger().info(
                f'multi-car avoidance ON — max offset {g("avoid_max_offset"):.2f} m, '
                f'follow gap {g("avoid_follow_gap"):.1f} m + {g("avoid_time_headway"):.1f} s, '
                f'detector range {g("avoid_max_range"):.0f} m')

        # ── state + ROS wiring ─────────────────────────────────────────────────
        self.x = self.y = self.yaw = self.speed = 0.0
        self.scan = None
        self.scan_seq = 0
        self.have_odom = False
        self.scan_time = self.odom_time = self.imu_time = 0.0
        self.have_imu = False
        self.nearest = 0
        self.lap = 0
        self._prev_near = 0
        self._log = 0
        self._cone_key = None                           # cached AEB cone mask
        self._cone_mask = None
        self.create_subscription(LaserScan, scan_topic, self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.create_timer(1.0 / float(self.get_parameter('control_hz').value), self._loop)
        self.get_logger().info(
            f'raceline_mpc ready — scan={scan_topic} odom={odom_topic} drive={drive_topic}')

    def _find_raceline(self):
        rl = find_best_raceline()                       # sim path / F1_RACELINE / best_*
        if rl and os.path.exists(rl):
            return rl
        # installed package share (ros2 run / launch on the car)
        try:
            from ament_index_python.packages import get_package_share_directory
            share = os.path.join(get_package_share_directory('f1tenth_gym_ros'),
                                 'racelines')
            for cand in ('best_raceline.csv', 'comp_raceline.csv'):
                p = os.path.join(share, cand)
                if os.path.exists(p):
                    return p
        except Exception:
            pass
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local = os.path.join(repo, 'racelines', 'best_raceline.csv')
        return local if os.path.exists(local) else None

    @staticmethod
    def _left_normals(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        tx = np.roll(x, -1) - np.roll(x, 1)
        ty = np.roll(y, -1) - np.roll(y, 1)
        tn = np.hypot(tx, ty) + 1e-9
        return -ty / tn, tx / tn

    # ── callbacks ──────────────────────────────────────────────────────────────
    def _scan_cb(self, m):
        self.scan = m
        self.scan_seq += 1
        self.scan_time = time.monotonic()

    def _odom_cb(self, m):
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        v = m.twist.twist.linear
        quaternion = [q.w, q.x, q.y, q.z]
        # quat2euler treats a zero quaternion as identity; it is missing pose
        # information, not evidence that the car points along the map x axis.
        if (not all(math.isfinite(value) for value in
                    (p.x, p.y, v.x, v.y, *quaternion))
                or math.hypot(*quaternion) < 1e-6):
            self.have_odom = False
            return
        self.x, self.y = p.x, p.y
        self.speed = math.hypot(v.x, v.y)
        _, _, self.yaw = quat2euler(quaternion)
        self.have_odom = True
        self.odom_time = time.monotonic()

    def _imu_cb(self, m):
        yaw_rate = m.angular_velocity.z
        # A NaN poisons the governor's low-pass state even after recovery.
        self.have_imu = math.isfinite(yaw_rate)
        if self.have_imu:
            self.yaw_rate = yaw_rate                  # |.| used; sign-agnostic
            self.imu_time = time.monotonic()

    # ── emergency brake: min range in a narrow forward cone ────────────────────
    def _forward_clear(self):
        s = self.scan
        return forward_clearance(s.ranges, s.angle_min, s.angle_increment,
                                 self.aeb_cone, max(0.03, s.range_min), s.range_max + 1e-3)

    def _stop(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.drive_pub.publish(msg)
        self._last_cmd = (0.0, 0.0)
        self._cmd_buf = [(0.0, 0.0)] * self._delay_ticks
        avoid = getattr(self, 'avoid', None)
        if avoid is not None:
            avoid.applied_offset *= 0.9              # let the line relax while stopped

    def _sensors_ready(self, now, timeout):
        return (math.isfinite(timeout) and timeout > 0.0
                and self.scan is not None and self.have_odom
                and 0.0 <= now - self.scan_time < timeout
                and 0.0 <= now - self.odom_time < timeout
                and all(math.isfinite(v) for v in
                        (self.x, self.y, self.yaw, self.speed)))

    # ── avoidance telemetry (/opponents, /avoidance_state) ─────────────────────
    def _publish_avoidance(self, cmd, now):
        if now - self._avoid_last_pub < self._avoid_pub_dt:
            return
        self._avoid_last_pub = now
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = 'map'
        for o in cmd.opponents:
            p = Pose()
            p.position.x, p.position.y = float(o.x), float(o.y)
            hd = math.atan2(o.vy, o.vx) if math.hypot(o.vx, o.vy) > 0.2 else 0.0
            p.orientation.z, p.orientation.w = math.sin(hd / 2), math.cos(hd / 2)
            pa.poses.append(p)
        self.opp_pub.publish(pa)
        st = String()
        st.data = json.dumps({
            'mode': cmd.mode, 'offset': round(cmd.offset, 3),
            'speed_cap': None if not math.isfinite(cmd.speed_cap) else round(cmd.speed_cap, 2),
            'speed_factor': round(cmd.speed_factor, 3),
            'opp_ahead': None if not math.isfinite(cmd.opp_ahead) else round(cmd.opp_ahead, 2),
            'n_opp': len(cmd.opponents),
            'room_l': round(cmd.room_left, 2), 'room_r': round(cmd.room_right, 2),
            'thought': cmd.thought})
        self.state_pub.publish(st)

    # ── control loop ───────────────────────────────────────────────────────────
    def _loop(self):
        now = time.monotonic()
        stale = float(self.get_parameter('sensor_timeout').value)
        if not self._sensors_ready(now, stale):
            self._stop()
            return
        # Brake before invoking a solver: an obstacle must not wait for an
        # expensive solve, and all emergency paths publish an unbiased stop.
        stop_dist = stopping_distance(self.speed, self.aeb_dist, self.delay, self.aeb_decel)
        clearance = self._forward_clear()
        if clearance is None or not math.isfinite(clearance) or clearance <= stop_dist:
            if self._log % 10 == 0:
                self.get_logger().warning('AEB — obstacle or invalid scan, stopping')
            self._log += 1
            self._stop()
            return
        # delay compensation: solve from where the car will be when the
        # command actually reaches the wheels, not where it was last measured.
        # Integrate through the in-flight command pipeline (oldest first) —
        # holding only the last command over-rotates the prediction whenever
        # the steer is changing and destabilizes the loop at larger delays.
        px, py, pyaw, pv = self.x, self.y, self.yaw, self.speed
        if self.delay > 0.0:
            px, py, pyaw, pv = predict_state(
                px, py, pyaw, pv,
                [c[0] for c in self._cmd_buf] or self._last_cmd[0],
                [c[1] for c in self._cmd_buf] or self._last_cmd[1],
                self.delay, self.L)
        self.nearest = find_nearest(px, py, self.rl_x, self.rl_y, self.nearest)

        # ── multi-car avoidance: opponents -> lateral offset + speed cap ──────
        offset, speed_cap, speed_factor = 0.0, float('inf'), 1.0
        avoid_cmd = None
        avoid = getattr(self, 'avoid', None)
        if avoid is not None:
            s = self.scan
            try:
                avoid_cmd = avoid.update(
                    getattr(self, 'scan_seq', 0), s.ranges, s.angle_min, s.angle_increment,
                    (self.x, self.y, self.yaw), self.speed, self.nearest,
                    self.rl_x, self.rl_y, self.rl_speed, now,
                    steer_cmd=self._last_cmd[0])
                offset = avoid_cmd.offset
                speed_cap = avoid_cmd.speed_cap
                speed_factor = avoid_cmd.speed_factor
                if not (math.isfinite(offset) and math.isfinite(speed_factor)):
                    offset, speed_factor = 0.0, 1.0
            except Exception as e:                  # never let perception kill control
                if self._log % 50 == 0:
                    self.get_logger().warning(f'avoidance layer error: {e} — tracking raceline')
                avoid.reset()
                offset, speed_cap, speed_factor = 0.0, float('inf'), 1.0

        steer = v_cmd = None
        if self.mpc.available:
            out = (self.mpc.solve((px, py, pyaw, pv), self.nearest, offset) if offset
                   else self.mpc.solve((px, py, pyaw, pv), self.nearest))
            if out is not None:
                steer, v_cmd = out
        if steer is None:                               # MPC off or solve failed
            # MAP has no offset input: tracking (raceline + offset) is the same
            # as tracking the raceline from a pose shifted by -offset.
            i = self.nearest
            if offset and hasattr(self, '_nx'):
                px, py = px - offset * self._nx[i], py - offset * self._ny[i]
            steer, v_cmd = self.map_ctl.control(px, py, pyaw, pv, i)

        v_cmd = float(v_cmd) * self.v_scale * speed_factor
        if not math.isfinite(v_cmd) or not math.isfinite(float(steer)):
            self._stop()
            return
        # follow governor: never close on a car ahead faster than the gap allows
        if math.isfinite(speed_cap):
            v_cmd = min(v_cmd, max(0.0, speed_cap))

        # Traction governor — scale down when the IMU says we're past the
        # lateral-grip budget (no-op until an IMU is publishing).
        if self.have_imu and now - self.imu_time < stale:
            v_cmd *= self.governor.update(self.yaw_rate, self.speed)

        # A slow solve can consume the remaining freshness budget. Never
        # publish a new moving command from inputs that expired during it.
        if (not self._sensors_ready(time.monotonic(), stale)
                or not math.isfinite(v_cmd)):
            self._stop()
            return
        # A creep floor must not raise a reduced/zero planner request or speed cap.
        v_cmd = max(0.0, min(v_cmd, self.v_max * self.v_scale))

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(np.clip(steer + self.steer_offset, -self.max_steer, self.max_steer))
        msg.drive.speed = float(v_cmd)
        self.drive_pub.publish(msg)
        self._last_cmd = (msg.drive.steering_angle, msg.drive.speed)
        if self._delay_ticks:                           # advance the pipeline
            self._cmd_buf.append(self._last_cmd)
            self._cmd_buf.pop(0)
        if avoid_cmd is not None:
            self._publish_avoidance(avoid_cmd, now)

        # lap counter (index wraps past start/finish)
        if self._prev_near > self.n - 12 and self.nearest < 12:
            self.lap += 1
            self.get_logger().info(f'lap {self.lap}')
        self._prev_near = self.nearest
        self._log += 1
        if self._log % 50 == 0:
            extra = ''
            if avoid_cmd is not None:
                cap = '' if not math.isfinite(speed_cap) else f' cap={speed_cap:.1f}'
                extra = (f' | {avoid_cmd.mode} off={offset:+.2f}'
                         f' opp={len(avoid_cmd.opponents)}{cap}')
            self.get_logger().info(
                f'wp {self.nearest}/{self.n} v={v_cmd:.1f} steer={math.degrees(steer):.0f}deg{extra}')


def main(args=None):
    rclpy.init(args=args)
    try:
        from rclpy.executors import ExternalShutdownException
    except ImportError:                      # older rclpy
        ExternalShutdownException = KeyboardInterrupt
    try:
        node = RacelineMPC()
        rclpy.spin(node)
    # SIGTERM from the pilot page's STOP / the launch system arrives as
    # ExternalShutdownException; it used to escape as a traceback on every stop
    except (FileNotFoundError, KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
