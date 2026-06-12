"""
Race agent — opponent-aware racing with a live, visible thought process.
========================================================================

Follows the optimized raceline (reusing the pure-pursuit primitives from
`pursuit_agent`), but overlays the `race_brain` perception + strategy:

  - every cycle it separates **racers from walls** in the lidar scan,
  - the strategist picks CRUISE / ATTACK / DEFEND / EVADE and a target line,
  - pure pursuit follows the raceline shifted by that strategic offset,
  - and the whole decision is published as RViz markers (opponent boxes, the
    planned line, and a floating text readout of the current mode + reasoning),
    so you can watch it think live.

Run (race day, raceline already on disk):
    python3 f1tenth_gym_ros/race_agent.py
"""

import json
import math
import threading
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseArray
from std_msgs.msg import ColorRGBA
from transforms3d.euler import quat2euler

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pursuit_agent import (
    find_best_raceline, load_raceline, find_nearest, find_lookahead, pp_steer)
from race_brain import OpponentDetector, RaceStrategist, Opponent, fuse_opponents
from mpc_controller import KinematicMPC


MODE_COLOR = {                       # r, g, b
    'CRUISE': (0.2, 0.8, 0.2),
    'ATTACK': (0.95, 0.2, 0.1),
    'DEFEND': (0.1, 0.4, 0.95),
    'EVADE':  (0.95, 0.75, 0.0),
}


class RaceAgent(Node):
    MAX_OFFSET = 0.6                     # max lateral move off the raceline (m)

    def __init__(self):
        super().__init__('race_agent')
        self.x = self.y = self.yaw = self.speed = 0.0
        self.scan = None
        self.nearest = 0
        self.applied_offset = 0.0
        self.prev_steer = 0.0
        self._log = 0
        self.lap_count = 0
        self._prev_near = 0
        # Live-state file in the mounted repo so the dashboard (on the host) can
        # read telemetry without a ROS bridge.
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._runtime_dir = os.path.join(repo, 'runtime')
        os.makedirs(self._runtime_dir, exist_ok=True)
        # Telemetry path — overridable to a fast local path (the Windows-Docker
        # bind mount has high write latency that can stall a synchronous writer).
        self._state_path = os.environ.get(
            'RACE_STATE_PATH', os.path.join(self._runtime_dir, 'race_state.json'))

        rl = find_best_raceline()
        if not rl:
            self.get_logger().error('No raceline found — run the optimizer first.')
            raise FileNotFoundError('no raceline')
        self.rl_x, self.rl_y, self.rl_hdg, self.rl_curv, self.rl_speed = load_raceline(rl)
        self.n = len(self.rl_x)
        # Mean waypoint spacing — lets all lookahead windows be expressed in METERS
        # and converted to point counts here, so the controller behaves identically
        # whether the raceline has 300 or 1300 points (i.e. on any track size /
        # optimizer resolution).  Without this, fixed point-count windows mean
        # wildly different physical distances as the resolution changes.
        dxy = np.hypot(np.diff(self.rl_x, append=self.rl_x[0]),
                       np.diff(self.rl_y, append=self.rl_y[0]))
        self.pt_spacing = float(np.mean(dxy))
        self.get_logger().info(
            f'Loaded raceline ({self.n} pts, ~{self.pt_spacing:.2f} m/pt) from {rl}')

        # overtaking zones = the slowest-exit longest straights (reuse curvature)
        self.overtake_idxs = self._overtake_zones()

        # pursuit params (overridable by the auto-tuner via runtime/tune_config.json)
        cfg = self._load_tune()
        self.max_steer = cfg.get('max_steer', 0.41)
        self.min_L     = cfg.get('min_L', 0.9)
        self.max_L     = cfg.get('max_L', 2.2)
        self.kL        = cfg.get('kL', 0.32)
        self.steer_smooth = cfg.get('steer_smooth', 0.70)   # higher = crisper
        self.anticip_k = cfg.get('anticip_k', 0.9)          # lookahead/brake window
        self.curv_gain = cfg.get('curv_gain', 1.2)          # corner lookahead shrink
        self.k_e = cfg.get('k_e', 1.4)                      # Stanley cross-track gain
        self.k_soft = cfg.get('k_soft', 2.0)               # Stanley low-speed softening
        self.L_wb = 0.33                                    # wheelbase (front-axle proj.)
        self.v_scale = cfg.get('v_scale', 1.0)              # overall speed scaling (tuner)
        # signed path curvature (for the steady-state feed-forward steer term)
        self._signed_curv = self._compute_signed_curv()
        self.v_max = float(self.rl_speed.max())

        # ── lateral controller selection: 'stanley' (default) or 'mpc' ──────────
        # MPC is opt-in so it can never destabilize the proven geometric path: if
        # selected but the solver (osqp) is missing or a solve fails at runtime,
        # the loop transparently falls back to the Stanley command computed below.
        self.declare_parameter('controller', 'stanley')
        self.controller = self.get_parameter('controller').value
        self.mpc = None
        if self.controller == 'mpc':
            self.mpc = KinematicMPC(wheelbase=self.L_wb, max_steer=self.max_steer,
                                    v_max=self.v_max + 0.5)
            if self.mpc.available:
                self.mpc.set_raceline(self.rl_x, self.rl_y, self.rl_hdg,
                                      self.rl_curv, self.rl_speed)
                self.get_logger().info('controller: MPC (kinematic LTV, osqp)')
            else:
                self.get_logger().warning(
                    'controller=mpc but osqp not available — using Stanley '
                    '(pip install osqp on the car to enable MPC)')
                self.mpc = None

        # ── Safety & smoothing layer — all thresholds are ROS2 parameters ──────
        self.declare_parameter('ttc_threshold', 0.35)    # s,  AEB trip
        self.declare_parameter('max_xt_error', 99.0)      # m,  cross-track deadman (disabled)
                                                          # (racing line uses track
                                                          # width; only trip when
                                                          # genuinely wall-bound)
        self.declare_parameter('watchdog_timeout', 0.1)  # s,  sensor staleness (100 ms)
        self.declare_parameter('xte_consec_limit', 3)    # cycles over xt limit
        self.declare_parameter('aeb_cone', 0.35)         # rad (~±20°), travel-path beams
                                                          # narrow: ignore side walls
                                                          # passed in normal cornering
        self.declare_parameter('kp_base', 2.5)           # steering gain numerator
        self.declare_parameter('kp_min', 0.45)           # steering gain floor
        self.declare_parameter('a_max', 4.0)             # m/s^2 accel rate limit
        self.declare_parameter('a_brake', 8.0)           # m/s^2 decel rate limit
        self.declare_parameter('loop_dt', 0.02)          # s,  control period
        self._SAFE_PARAMS = ('ttc_threshold', 'max_xt_error', 'watchdog_timeout',
                             'xte_consec_limit', 'aeb_cone', 'kp_base', 'kp_min',
                             'a_max', 'a_brake', 'loop_dt')
        for k in self._SAFE_PARAMS:
            setattr(self, k, self.get_parameter(k).value)
        self.add_on_set_parameters_callback(self._on_params)

        # Thread-safe fail-safe state.
        self._lock = threading.Lock()
        self.scan_t = None              # arrival times for the watchdog
        self.odom_t = None
        self.v_cmd = 0.0                # rate-limited speed command (filter state)
        self.xte_count = 0              # consecutive cycles over the XT boundary
        self._emergency = False
        self._safe_log_t = {}           # per-reason last-log time (throttle)
        self._t_start = self.get_clock().now().nanoseconds * 1e-9
        self.aeb_min_speed = 0.3        # below this, allow slow cornering without AEB lock-out
        self.xte_min_speed = 1.0        # deadman = *high-speed* traction loss only
        # Redundant opponent brake (camera+lidar fused): a confirmed car this
        # close ahead, in the travel cone, forces a back-off even if the lidar
        # AEB hasn't tripped — so a car in the single-plane lidar's blind spot
        # that the camera sees still stops us.
        self.opp_brake_dist = 1.2       # m, opponent-ahead distance that backs us off
        self.opp_brake_cone = 0.5       # rad, forward cone (centred on commanded steer)

        # Telemetry writer runs off the control thread (mounted-volume I/O is slow).
        self._latest_state = None
        self._writer_stop = False
        threading.Thread(target=self._state_writer, daemon=True).start()

        # Hot-reload tuning config every 2 seconds (the practice tuner writes it).
        self._tune_path = os.path.join(self._runtime_dir, 'tune_config.json')
        self._tune_mtime = 0.0
        self._last_reload = 0.0
        # Per-lap telemetry for the practice tuner (reset on each lap crossing).
        self._lap_xte_max = 0.0
        self._lap_xte_sum = 0.0
        self._lap_ticks = 0
        self._lap_start_wall = time.time()
        self._lap_history_path = os.path.join(self._runtime_dir, 'lap_history.jsonl')

        self.det = OpponentDetector()
        self.strat = RaceStrategist()
        self.cam_opps = []          # latest camera opponents (world frame)
        self.cam_t = -1.0           # time of last camera message
        self.CAM_FOV = 1.0          # camera half-FOV (rad, ~57°)

        # Sensor intake runs in its own (reentrant) group so it never gets
        # starved by the heavy control timer — keeps the watchdog measuring true
        # sensor freshness, not executor scheduling lag.
        sg = ReentrantCallbackGroup()
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10, callback_group=sg)
        self.create_subscription(Odometry, '/ego_racecar/odom', self._odom_cb, 10,
                                 callback_group=sg)
        self.create_subscription(PoseArray, '/camera_opponents_poses', self._cam_cb, 5,
                                 callback_group=sg)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.viz_pub = self.create_publisher(MarkerArray, '/race_thinking', 10)
        self.create_timer(0.02, self._loop)
        self.get_logger().info('Race agent ready — watch /race_thinking in RViz')

    def _load_tune(self):
        try:
            repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            p = os.path.join(repo, 'runtime', 'tune_config.json')
            if os.path.exists(p):
                with open(p) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _hot_reload_tune(self):
        """Re-read tune_config.json if it changed (called from _loop)."""
        now = time.time()
        if now - self._last_reload < 2.0:
            return
        self._last_reload = now
        try:
            mt = os.path.getmtime(self._tune_path)
            if mt <= self._tune_mtime:
                return
            self._tune_mtime = mt
            cfg = self._load_tune()
            changed = []
            for k in ('k_e', 'k_soft', 'steer_smooth', 'anticip_k', 'curv_gain',
                       'kL', 'max_L', 'min_L', 'v_scale', 'max_steer'):
                if k in cfg and abs(cfg[k] - getattr(self, k, cfg[k])) > 1e-6:
                    setattr(self, k, cfg[k])
                    changed.append(f'{k}={cfg[k]}')
            if changed:
                self.get_logger().info(f'[TUNE] hot-reloaded: {", ".join(changed)}')
        except Exception:
            pass

    def _on_params(self, params):
        """Live-adjust safety/tuning thresholds via `ros2 param set`."""
        for p in params:
            if p.name in self._SAFE_PARAMS:
                setattr(self, p.name, p.value)
        return SetParametersResult(successful=True)

    # ── callbacks ────────────────────────────────────────────────────────────
    def _odom_cb(self, m):
        self.x = m.pose.pose.position.x
        self.y = m.pose.pose.position.y
        self.speed = np.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y)
        q = m.pose.pose.orientation
        _, _, self.yaw = quat2euler([q.w, q.x, q.y, q.z])
        with self._lock:
            self.odom_t = self.get_clock().now().nanoseconds * 1e-9

    def _scan_cb(self, m):
        self.scan = m
        with self._lock:
            self.scan_t = self.get_clock().now().nanoseconds * 1e-9

    def _cam_cb(self, m):
        self.cam_t = self.get_clock().now().nanoseconds * 1e-9
        self.cam_opps = [Opponent(p.position.x, p.position.y,
                                  np.hypot(p.position.x - self.x, p.position.y - self.y),
                                  0.3, 0.0, source='camera') for p in m.poses]

    # ── helpers ──────────────────────────────────────────────────────────────
    def _overtake_zones(self):
        """Indices at the end of the longest straights (overtake-under-braking)."""
        n = self.n
        spacing = np.hypot(self.rl_x[1]-self.rl_x[0], self.rl_y[1]-self.rl_y[0])
        fast = self.rl_speed > 0.9 * self.rl_speed.max()
        zones, i = [], 0
        runs = []
        cur = None
        for k in range(n):
            if fast[k] and cur is None:
                cur = k
            elif not fast[k] and cur is not None:
                runs.append((cur, k - 1)); cur = None
        if cur is not None:
            runs.append((cur, n - 1))
        runs.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
        for a, b in runs[:2]:
            zones.append(b)                          # braking point at straight end
        return zones

    def _compute_signed_curv(self):
        """Signed path curvature (+ = turning left) for the feed-forward term."""
        n = self.n
        sc = np.zeros(n)
        for i in range(n):
            dh = math.atan2(math.sin(self.rl_hdg[(i+1) % n] - self.rl_hdg[(i-1) % n]),
                            math.cos(self.rl_hdg[(i+1) % n] - self.rl_hdg[(i-1) % n]))
            sc[i] = math.copysign(float(self.rl_curv[i]), dh)
        return sc

    def _room_lr(self):
        """Free distance to walls on the left / right from the lidar (m)."""
        s = self.scan
        r = np.asarray(s.ranges, np.float32)
        r = np.where(np.isfinite(r) & (r > 0.05), r, 30.0)
        ang = s.angle_min + np.arange(len(r)) * s.angle_increment
        left = r[(ang > 1.3) & (ang < 1.84)]         # ~ +90°
        right = r[(ang < -1.3) & (ang > -1.84)]
        room_l = float(left.min()) if len(left) else 1.0
        room_r = float(right.min()) if len(right) else 1.0
        return max(room_l - 0.2, 0.0), max(room_r - 0.2, 0.0)

    # ── safety & smoothing helpers ─────────────────────────────────────────────
    def _throttled_log(self, key, msg, error=False, period=0.5):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._safe_log_t.get(key, -1e9) >= period:
            self._safe_log_t[key] = now
            (self.get_logger().error if error else self.get_logger().warning)(msg)

    def _aeb_min_ttc(self, steer_cmd=0.0):
        """Minimum Time-To-Collision over the beams along the car's travel arc (s).

        The cone is centred on the *commanded steering angle*, not straight ahead:
        on a corner the car is turning into the open track, so the relevant
        obstacle is what lies along that arc — not the outside wall that sits dead
        ahead but which the car is steering away from.  A fixed straight-ahead cone
        trips on every tight corner (the wall is always 'in front'); centring it on
        the steer makes AEB fire only on a genuine impending collision, which is
        what lets the car carry speed through hairpins of any radius.
        """
        s = self.scan
        r = np.asarray(s.ranges, np.float32)
        ang = s.angle_min + np.arange(len(r)) * s.angle_increment
        # Closing speed along beam i = v*cos(theta_i); only beams we're driving
        # *toward* (closing > 0) can collide — the rest are ignored.
        closing = self.speed * np.cos(ang)
        mask = (np.isfinite(r) & (r > 0.03) & (closing > 1e-6)
                & (np.abs(ang - steer_cmd) < self.aeb_cone))
        if not mask.any():
            return float('inf')
        return float(np.min(r[mask] / np.maximum(closing[mask], 1e-6)))

    def _nearest_opp_ahead(self, opps, steer_cmd=0.0):
        """Distance (m) to the closest opponent within the forward travel cone.

        `opps` is already the lidar tracker fused with camera detections, so this
        is the camera acting as a *backup obstacle sensor*: it fires for a car the
        camera confirms even when that car sits in a lidar blind spot (below or
        above the single horizontal scan plane).  The cone is centred on the
        commanded steer (like the lidar AEB) so it watches the arc the car is
        actually taking, not a fixed straight-ahead wedge.
        """
        best = float('inf')
        for o in opps:
            dx, dy = o.x - self.x, o.y - self.y
            bearing = math.atan2(dy, dx) - self.yaw
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            if abs(bearing - steer_cmd) < self.opp_brake_cone:
                best = min(best, math.hypot(dx, dy))
        return best

    def _safety_check(self, now, xte, steer_cmd=0.0):
        """Returns (level, reason).  level: '' none | 'stop' hard-stop | 'limp'.

        'stop'  (collision / sensor loss): cut throttle AND centre the wheels.
        'limp'  (drifted off the line): slow hard but KEEP steering toward the
                line, so the car recovers instead of freezing — centring the
                wheels while off-line would strand the car (it can't steer back).
        """
        # Watchdog (after a short startup grace while sensors warm up) -> STOP.
        if now - self._t_start > 1.0:
            with self._lock:
                st, ot = self.scan_t, self.odom_t
            if st is not None and (now - st) > self.watchdog_timeout:
                return 'stop', f'WATCHDOG /scan stale {1000*(now-st):.0f}ms'
            if ot is not None and (now - ot) > self.watchdog_timeout:
                return 'stop', f'WATCHDOG localization stale {1000*(now-ot):.0f}ms'
        # AEB — only meaningful once actually rolling -> STOP.
        if self.speed > self.aeb_min_speed:
            ttc = self._aeb_min_ttc(steer_cmd)
            if ttc < self.ttc_threshold:
                return 'stop', f'AEB Triggered! TTC: {ttc:.2f}s'
        # Cross-track deadman -> LIMP.  5-second startup grace lets the car acquire
        # the line from its spawn offset (0.5-1.5 m off) without tripping.
        if now - self._t_start < 5.0:
            self.xte_count = 0
            return '', ''
        if self.speed > self.xte_min_speed and xte > self.max_xt_error:
            self.xte_count += 1
        else:
            self.xte_count = 0
        if self.xte_count > self.xte_consec_limit:
            return 'limp', f'XT-Error {xte:.2f}m for {self.xte_count} cycles — recovering'
        return '', ''

    def _pts(self, meters):
        """Convert a lookahead distance (m) to a raceline point count (>=2)."""
        return int(np.clip(round(meters / self.pt_spacing), 2, self.n // 2))

    def _kp_gain(self, v):
        """Velocity-scaled steering gain: full bite slow, damped fast (<=1, no amp)."""
        return float(np.clip(self.kp_base / max(v, 0.5), self.kp_min, 1.0))

    def _ramp(self, v_target, dt):
        """Accel/decel rate-limited speed command (low-pass on torque)."""
        dv = float(np.clip(v_target - self.v_cmd, -self.a_brake * dt, self.a_max * dt))
        self.v_cmd += dv
        return self.v_cmd

    # ── main loop ────────────────────────────────────────────────────────────
    def _loop(self):
        if self.scan is None:
            return
        s = self.scan
        t = self.get_clock().now().nanoseconds * 1e-9

        # Perception: lidar detection, fused with camera when the camera is live.
        lidar_opps = self.det.detect(s.ranges, s.angle_min, s.angle_increment,
                                     (self.x, self.y, self.yaw), t)
        cam_live = (t - self.cam_t) < 0.5                  # recent camera frames?
        if cam_live:
            opps = fuse_opponents(lidar_opps, self.cam_opps,
                                  (self.x, self.y, self.yaw), fov=self.CAM_FOV)
        else:
            opps = lidar_opps                               # lidar-only (e.g. sim)
        self.nearest = find_nearest(self.x, self.y, self.rl_x, self.rl_y, self.nearest)
        room_l, room_r = self._room_lr()
        d = self.strat.decide(self.nearest, self.speed, self.rl_x, self.rl_y,
                              self.rl_speed, room_l, room_r, opps, self.overtake_idxs)

        # Smoothly move to the strategic offset, hard-clamped to stay on track
        # (lidar room over-estimates where the inside opens up at corners).
        target_off = float(np.clip(d.offset, -self.MAX_OFFSET, self.MAX_OFFSET))
        self.applied_offset += np.clip(target_off - self.applied_offset, -0.05, 0.05)
        self.applied_offset = float(np.clip(self.applied_offset,
                                            -self.MAX_OFFSET, self.MAX_OFFSET))

        # ── Stanley lateral control on the (offset) raceline ───────────────────
        # δ = heading_error − atan2(k_e·e, v + k_soft), evaluated at the FRONT axle.
        # The explicit heading-alignment term rotates the car onto the path through
        # an apex instead of letting the heading lag (which made pure pursuit run
        # wide on the tight hairpins).
        fx = self.x + self.L_wb * math.cos(self.yaw)
        fy = self.y + self.L_wb * math.sin(self.yaw)
        fi = find_nearest(fx, fy, self.rl_x, self.rl_y, self.nearest)
        nx, ny = self._normal(fi)
        # signed cross-track error from the *intended* line (raceline + offset);
        # + = front axle is left of the path.
        e = (fx - self.rl_x[fi]) * nx + (fy - self.rl_y[fi]) * ny - self.applied_offset
        heading_err = math.atan2(math.sin(self.rl_hdg[fi] - self.yaw),
                                 math.cos(self.rl_hdg[fi] - self.yaw))
        # feed-forward: steady-state steer for the path's curvature (bicycle model)
        ff = math.atan(self.L_wb * float(self._signed_curv[fi]))
        steer = ff + heading_err - math.atan2(self.k_e * e, self.speed + self.k_soft)
        steer = self.steer_smooth * steer + (1.0 - self.steer_smooth) * self.prev_steer
        self.prev_steer = steer

        # Anticipatory preview distance in METERS (speed * preview-time), clamped
        # to a sane window, then converted to a point count for the current
        # raceline resolution.  Expressing it in metres makes the corner-anticipation
        # behave the same on any track size or waypoint density.
        prev_m = float(np.clip(self.speed * self.anticip_k, 3.5, 11.0))
        nahead = self._pts(prev_m)
        look = nahead
        curv_ahead = max(abs(float(self.rl_curv[(self.nearest + k) % self.n]))
                         for k in range(look + 1))
        li = (fi + self._pts(2.4)) % self.n          # viz/steer target ~2.4 m ahead
        lnx, lny = self._normal(li)
        tx = self.rl_x[li] + self.applied_offset * lnx
        ty = self.rl_y[li] + self.applied_offset * lny

        # Anticipatory braking: slow for the slowest raceline point within reach.
        base = float(min(self.rl_speed[(self.nearest + k) % self.n]
                         for k in range(nahead + 1)))
        spd = base * d.speed_factor
        # Clamp steer to max_steer before the speed-penalty formula so that a large
        # Stanley correction at a tight apex can't drive spd negative (which traps
        # the car in a slow loop and prevents AEB recovery).
        steer_clamped = float(np.clip(abs(steer), 0.0, self.max_steer))
        spd *= max(0.0, 1.0 - 0.30 * steer_clamped / self.max_steer)

        # ── MPC override (opt-in) ──────────────────────────────────────────────
        # Plans steer + speed over a horizon to track the (offset) raceline; on a
        # solver miss this is skipped and the Stanley command above stands.  The
        # strategist's speed_factor still scales the MPC speed so opponent logic
        # (EVADE / brake-to-pass) keeps authority over pace.
        if self.mpc is not None:
            mpc_out = self.mpc.solve((self.x, self.y, self.yaw, self.speed),
                                     self.nearest, offset=self.applied_offset)
            if mpc_out is not None:
                steer, v_mpc = mpc_out
                steer = float(np.clip(steer, -self.max_steer, self.max_steer))
                self.prev_steer = steer
                spd = float(v_mpc) * d.speed_factor

        # ══ SAFETY & SMOOTHING LAYER — preempts pursuit on any fail-safe ════════
        # Cross-track error = deviation from the *intended* line (raceline + the
        # commanded strategic offset), so legitimate overtaking offsets don't trip
        # the deadman; only genuine understeer / traction loss does.
        cnx, cny = self._normal(self.nearest)
        lateral = (self.x - self.rl_x[self.nearest]) * cnx + \
                  (self.y - self.rl_y[self.nearest]) * cny
        xte = abs(lateral - self.applied_offset)
        # Pass the commanded steer so AEB watches the car's actual arc, not a
        # straight cone that would trip on every corner's outer wall.
        level, reason = self._safety_check(t, xte, steer)

        # Redundant opponent brake — the camera as a backup obstacle sensor.  If
        # a fused (camera+lidar) car is close ahead in the travel cone and the
        # line-following safety hasn't already tripped, drop to limp: slow hard
        # but keep steering, so the strategist can still pick an evade line.
        if level == '':
            opp_ahead = self._nearest_opp_ahead(opps, steer)
            if opp_ahead < self.opp_brake_dist:
                level = 'limp'
                reason = (f'opponent {opp_ahead:.2f}m ahead (camera+lidar) '
                          f'— backing off')

        if level == 'stop':
            # Collision / sensor loss: instant hard cut, centre the wheels.
            with self._lock:
                self._emergency = True
            self.v_cmd = 0.0                       # bypass ramp: instant, hard cut
            self.prev_steer = 0.0
            out_steer, out_speed = 0.0, 0.0        # center wheels, full stop
            self._throttled_log('safe', f'[CRITICAL SAFE] {reason} — hard stop',
                                error=True)
        elif level == 'limp':
            # Off the line: keep the (full-authority) Stanley steer pointing back
            # at the line and crawl, so the car actively recovers instead of
            # freezing.  Full steering bite (no speed-gain damping) to turn back.
            with self._lock:
                self._emergency = True
            out_steer = float(np.clip(steer, -self.max_steer, self.max_steer))
            out_speed = self._ramp(self.xte_min_speed, self.loop_dt)
            self._throttled_log('safe', f'[SAFE] {reason}', error=False)
        else:
            with self._lock:
                self._emergency = False
            # Stanley already accounts for speed in its atan2(k_e*e, v+k_soft)
            # term — no extra speed-dependent gain scaling, which double-damps
            # at speed and prevents the car from steering into corners.
            out_steer = float(np.clip(steer, -self.max_steer, self.max_steer))
            # (5) longitudinal accel/brake rate limiting — smooth torque delivery
            out_speed = self._ramp(max(spd, 0.6), self.loop_dt)

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(out_steer)
        msg.drive.speed = float(out_speed)
        self.drive_pub.publish(msg)

        # Lap counter: index wraps past the start/finish.  Zone scales with the
        # raceline density (a fixed ±12 spans 24% of a sparse 100-pt line but
        # can be jumped in one 50 Hz tick on a dense one).
        zone = max(12, self.n // 25)
        if self._prev_near > self.n - zone and self.nearest < zone:
            self.lap_count += 1
        self._prev_near = self.nearest

        # Throttle the non-critical work (viz / telemetry) so the control loop
        # stays light and never stalls under it (which would trip the watchdog).
        self._log += 1
        if self._log % 3 == 0:
            self._publish_viz(d, opps, (tx, ty))
        if self._log % 5 == 0:
            self._write_state(d, opps, spd)
        if self._log % 25 == 0:
            self.get_logger().info(f'[{d.mode}] {d.thought} | opp={len(opps)} '
                                   f'off={self.applied_offset:+.2f} v={spd:.1f}')

    def _write_state(self, d, opps, spd):
        state = {
            'running': True,
            'ts': time.time(),
            'mode': d.mode,
            'thought': d.thought,
            'speed': round(float(self.v_cmd), 2),
            'raceline_speed': round(float(self.rl_speed[self.nearest]), 2),
            'offset': round(float(self.applied_offset), 2),
            'emergency': bool(self._emergency),
            'opp_count': len(opps),
            'lap': self.lap_count,
            'nearest': int(self.nearest),
            'ego': [round(float(self.x), 2), round(float(self.y), 2),
                    round(float(self.yaw), 3)],
            'opponents': [{'x': round(float(o.x), 2), 'y': round(float(o.y), 2),
                           'source': getattr(o, 'source', 'lidar')} for o in opps],
        }
        # Hand off to the writer thread — NEVER do disk I/O on the control loop
        # (the repo is a Windows-mounted volume; a slow write would stall the
        # executor, starve /scan, and trip the watchdog).
        self._latest_state = state

    def _state_writer(self):
        """Daemon: flush the latest telemetry to disk, off the control thread."""
        while not self._writer_stop:
            time.sleep(0.1)
            st = self._latest_state
            if st is None:
                continue
            try:
                tmp = self._state_path + '.tmp'
                with open(tmp, 'w') as f:
                    json.dump(st, f)
                os.replace(tmp, self._state_path)
            except Exception:
                pass

    def _normal(self, i):
        tx = self.rl_x[(i+1) % self.n] - self.rl_x[(i-1) % self.n]
        ty = self.rl_y[(i+1) % self.n] - self.rl_y[(i-1) % self.n]
        tn = np.hypot(tx, ty) + 1e-9
        return -ty / tn, tx / tn

    # ── visualization ────────────────────────────────────────────────────────
    def _publish_viz(self, d, opps, target):
        arr = MarkerArray()
        col = MODE_COLOR.get(d.mode, (1, 1, 1))

        # detected opponents as boxes
        for k, o in enumerate(opps):
            m = Marker()
            m.header.frame_id = 'map'; m.ns = 'opponents'; m.id = k
            m.type = Marker.CUBE; m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y = float(o.x), float(o.y)
            m.pose.position.z = 0.1; m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.45; m.scale.z = 0.2
            src = getattr(o, 'source', 'lidar')      # lidar=red, camera=blue, fused=green
            sc = {'lidar': (0.9, 0.1, 0.1), 'camera': (0.1, 0.6, 0.9),
                  'fused': (0.1, 0.9, 0.3)}.get(src, (0.9, 0.1, 0.1))
            m.color = ColorRGBA(r=sc[0], g=sc[1], b=sc[2], a=0.9)
            arr.markers.append(m)

        # planned line: ego -> target
        line = Marker()
        line.header.frame_id = 'map'; line.ns = 'plan'; line.id = 0
        line.type = Marker.LINE_STRIP; line.action = Marker.ADD
        line.scale.x = 0.08
        line.color = ColorRGBA(r=col[0], g=col[1], b=col[2], a=0.9)
        line.points = [Point(x=float(self.x), y=float(self.y), z=0.15),
                       Point(x=float(target[0]), y=float(target[1]), z=0.15)]
        arr.markers.append(line)

        # floating thought text above the car
        txt = Marker()
        txt.header.frame_id = 'map'; txt.ns = 'thought'; txt.id = 0
        txt.type = Marker.TEXT_VIEW_FACING; txt.action = Marker.ADD
        txt.pose.position.x = float(self.x); txt.pose.position.y = float(self.y)
        txt.pose.position.z = 1.0; txt.pose.orientation.w = 1.0
        txt.scale.z = 0.6
        txt.color = ColorRGBA(r=col[0], g=col[1], b=col[2], a=1.0)
        txt.text = d.thought
        arr.markers.append(txt)
        self.viz_pub.publish(arr)


def main(args=None):
    # Avoid stop-the-world GC pauses stalling the real-time control loop (the
    # node's objects are acyclic / refcount-freed); a gen-2 sweep can exceed the
    # 100 ms watchdog. Freeze long-lived objects and disable cyclic GC.
    import gc
    gc.freeze()
    gc.disable()
    rclpy.init(args=args)
    node = None
    try:
        node = RaceAgent()
        # Multi-threaded so sensor callbacks run parallel to the control loop.
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
    except (FileNotFoundError, KeyboardInterrupt):
        pass
    finally:
        if node is not None:
            node._writer_stop = True
        rclpy.shutdown()


if __name__ == '__main__':
    main()
