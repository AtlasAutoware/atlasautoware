"""
F1TENTH Championship Racing Agent — Progressive Learning Edition
----------------------------------------------------------------
Designed for one training day on circuit before race day.

Learning phases (auto-progressing):

  LAP 1  — SURVIVAL
    Conservative gap-follow. Priority: complete the lap without crashing.
    Records driven path. Speed capped at 2.5 m/s.

  LAP 2  — CENTERLINE
    First raceline fitted from lap 1 path. Pure Pursuit activated.
    Speed capped at 60% of optimal. Refines path with actual pursuit data.

  LAP 3  — SPEED RAMP
    Raceline re-fitted from laps 1+2. Speed cap raised to 80%.
    Starts probing corners harder.

  LAP 4  — OPTIMIZATION
    Raceline re-fitted from all prior laps (weighted toward better laps).
    Speed cap raised to 95%. Near race pace.

  LAP 5+  — RACE PACE
    Final raceline locked in at 100% optimized speed.
    Opponent avoidance active. This is the race configuration saved to disk.

On race day (no training):
    Loads best saved raceline → straight to race pace.
    If no raceline → falls back to gap-follow until one can be fitted.

Key features:
  - Per-lap raceline improvement with weighted path merging
  - Automatic lap timing for performance tracking
  - Best-lap raceline always preserved on disk
  - TrackProfile auto-tunes all params per track
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

import numpy as np
import csv, os, glob, time
from scipy.interpolate import splprep, splev
from transforms3d.euler import quat2euler


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

RACELINE_DIR      = '/sim_ws/src/f1tenth_gym_ros/racelines/'
NUM_RACELINE_PTS  = 200
MAX_SPEED         = 7.0
MIN_SPEED         = 1.5
MAX_DECEL         = 5.0
A_LAT_MAX         = 6.0
MIN_WPT_DIST      = 0.15    # m between recorded path points
LAP_CLOSE_DIST    = 1.2     # m — radius to detect lap completion
LAP_DEPART_DIST   = 2.5     # m — must travel this far before a lap can complete
GAP_BUBBLE        = 0.35
OPPONENT_ANGLE    = 0.6

# Lidar safety net (race day).  The optimized raceline is the source of truth;
# lidar only intervenes for obstacles the raceline cannot know about (opponents).
EMERGENCY_DIST    = 0.45    # m — obstacle this close ahead → brake (opponent)
EMERGENCY_CONE    = 0.15    # rad — narrow forward cone (~9°), avoids wall pickup
STEER_SMOOTH      = 0.55    # steering low-pass: new = a*cmd + (1-a)*prev

# Speed caps per lap phase
LAP_SPEED_CAPS = {
    1: 0.0,    # gap follow only, no pursuit
    2: 0.60,
    3: 0.80,
    4: 0.95,
    5: 1.00,   # race pace
}

# How many path samples to keep from each prior lap (weighted blend)
# More recent laps weighted higher
LAP_PATH_WEIGHTS = {
    1: 0.6,
    2: 0.8,
    3: 0.9,
    4: 1.0,
}

EXPLORE_SPEED     = 2.5     # m/s during gap-follow laps


# ══════════════════════════════════════════════════════════════════════════════
# TRACK PROFILE AUTO-TUNER
# ══════════════════════════════════════════════════════════════════════════════

class TrackProfile:
    def __init__(self, curvatures, speeds):
        self.max_curv  = float(np.percentile(curvatures, 95))
        self.mean_curv = float(np.mean(curvatures))

        if self.max_curv > 0.8:
            self.type           = 'TECHNICAL'
            self.speed_scale    = 0.82
            self.min_lookahead  = 0.6
            self.max_lookahead  = 1.8
            self.lookahead_k    = 0.25
            self.max_steer      = 0.42
            self.blocked_thresh = 0.8
        elif self.max_curv > 0.4:
            self.type           = 'MIXED'
            self.speed_scale    = 0.88
            self.min_lookahead  = 0.8
            self.max_lookahead  = 2.2
            self.lookahead_k    = 0.30
            self.max_steer      = 0.40
            self.blocked_thresh = 0.8
        else:
            self.type           = 'FAST'
            self.speed_scale    = 0.94
            self.min_lookahead  = 0.9
            self.max_lookahead  = 2.2
            self.lookahead_k    = 0.32
            self.max_steer      = 0.41
            self.blocked_thresh = 0.8

    def summary(self):
        return (f'{self.type} | max_curv={self.max_curv:.3f} | '
                f'speed_scale={self.speed_scale:.2f} | '
                f'lookahead=[{self.min_lookahead},{self.max_lookahead}]')


# ══════════════════════════════════════════════════════════════════════════════
# RACELINE I/O
# ══════════════════════════════════════════════════════════════════════════════

def find_best_raceline():
    """Find raceline marked as 'best' first, then newest."""
    env = os.environ.get('F1_RACELINE')
    if env and os.path.exists(env):
        return env
    # Prefer a file named 'best_*.csv'
    bests = glob.glob(os.path.join(RACELINE_DIR, 'best_*.csv'))
    if bests:
        return max(bests, key=os.path.getmtime)
    csvs = glob.glob(os.path.join(RACELINE_DIR, '*.csv'))
    if csvs:
        return max(csvs, key=os.path.getmtime)
    return None


def load_raceline(path):
    xs, ys, hdgs, curvs, spds = [], [], [], [], []
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            xs.append(float(row['x']));   ys.append(float(row['y']))
            hdgs.append(float(row['heading']))
            curvs.append(float(row['curvature']))
            spds.append(float(row['speed']))
    return (np.array(xs), np.array(ys), np.array(hdgs),
            np.array(curvs), np.array(spds))


def save_raceline(path, x, y, hdg, curv, spd):
    os.makedirs(RACELINE_DIR, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y', 'heading', 'curvature', 'speed'])
        for i in range(len(x)):
            w.writerow([round(x[i],4), round(y[i],4), round(hdg[i],4),
                        round(curv[i],6), round(spd[i],3)])


# ══════════════════════════════════════════════════════════════════════════════
# RACELINE FITTER
# ══════════════════════════════════════════════════════════════════════════════

def fit_raceline(path_x, path_y, n_pts=NUM_RACELINE_PTS):
    px, py = np.array(path_x), np.array(path_y)
    smooth = max(len(px) * 2.0, 10.0)
    try:
        tck, _ = splprep([px, py], s=smooth, per=True, k=3)
    except Exception:
        tck, _ = splprep([px, py], s=smooth * 4, per=True, k=3)

    u = np.linspace(0, 1, n_pts)
    xf, yf = splev(u, tck)
    xf, yf = np.array(xf), np.array(yf)
    n = len(xf)

    hdg  = np.zeros(n)
    curv = np.zeros(n)
    for i in range(n):
        p, nx = (i-1)%n, (i+1)%n
        hdg[i] = np.arctan2(yf[nx]-yf[p], xf[nx]-xf[p])
        ax,ay = xf[p],yf[p]; bx,by = xf[i],yf[i]; cx,cy = xf[nx],yf[nx]
        area2 = abs((bx-ax)*(cy-ay)-(cx-ax)*(by-ay))
        d1=np.hypot(bx-ax,by-ay); d2=np.hypot(cx-bx,cy-by); d3=np.hypot(cx-ax,cy-ay)
        denom = d1*d2*d3
        curv[i] = area2/denom if denom > 1e-9 else 0.0

    spd = np.where(curv>1e-4, np.sqrt(A_LAT_MAX/(curv+1e-9)), MAX_SPEED)
    spd = np.clip(spd, MIN_SPEED, MAX_SPEED)
    for i in range(1, n):
        d = np.hypot(xf[i]-xf[i-1], yf[i]-yf[i-1])
        spd[i] = min(spd[i], np.sqrt(spd[i-1]**2 + 2*MAX_DECEL*d))
    for i in range(n-2, -1, -1):
        d = np.hypot(xf[i+1]-xf[i], yf[i+1]-yf[i])
        spd[i] = min(spd[i], np.sqrt(spd[i+1]**2 + 2*MAX_DECEL*d))

    return xf, yf, hdg, curv, spd


def merge_paths(laps_data, weights):
    """
    Merge multiple laps of path data with weights.
    laps_data: list of (path_x, path_y) tuples
    weights:   list of floats, same length
    Returns merged (path_x, path_y)
    """
    all_x, all_y = [], []
    for (px, py), w in zip(laps_data, weights):
        # Subsample each lap proportionally to its weight
        n = len(px)
        keep = max(10, int(n * w))
        idx = np.round(np.linspace(0, n-1, keep)).astype(int)
        all_x.extend([px[i] for i in idx])
        all_y.extend([py[i] for i in idx])
    return all_x, all_y


# ══════════════════════════════════════════════════════════════════════════════
# PURE PURSUIT
# ══════════════════════════════════════════════════════════════════════════════

def find_nearest(x, y, rl_x, rl_y, prev, search=60):
    n = len(rl_x)
    bd, bi = float('inf'), prev
    for off in range(-5, search):
        idx = (prev+off)%n
        d = np.hypot(rl_x[idx]-x, rl_y[idx]-y)
        if d < bd: bd, bi = d, idx
    return bi

def find_lookahead(x, y, yaw, rl_x, rl_y, L, nearest):
    n = len(rl_x)
    for off in range(1, n):
        idx = (nearest+off)%n
        if np.hypot(rl_x[idx]-x, rl_y[idx]-y) >= L:
            return rl_x[idx], rl_y[idx], idx
    idx = (nearest+10)%n
    return rl_x[idx], rl_y[idx], idx

def pp_steer(x, y, yaw, tx, ty, L, max_steer):
    dx, dy = tx-x, ty-y
    lx =  dx*np.cos(yaw)+dy*np.sin(yaw)
    ly = -dx*np.sin(yaw)+dy*np.cos(yaw)
    steer = np.arctan2(2.0*0.33*np.sin(np.arctan2(ly,lx)), L)
    return np.clip(steer, -max_steer, max_steer)


# ══════════════════════════════════════════════════════════════════════════════
# LIDAR
# ══════════════════════════════════════════════════════════════════════════════

def best_gap(ranges, angle_min, angle_inc):
    r = np.array(ranges, dtype=np.float32)
    r = np.where(np.isfinite(r)&(r>0.05), r, 10.0)
    angles = angle_min + np.arange(len(r))*angle_inc
    mi = np.argmin(r)
    bub = max(1, int(np.degrees(np.arctan2(GAP_BUBBLE, max(r[mi],0.1)))))
    rs = r.copy()
    rs[max(0,mi-bub):min(len(r),mi+bub+1)] = 0.0
    thresh = max(0.6, rs.max()*0.4)
    ig = rs > thresh
    if not ig.any(): return angles[np.argmax(rs)], r
    bs,be,bl,cs,cl = 0,0,0,0,0
    for i,g in enumerate(ig):
        if g:
            if cl==0: cs=i
            cl+=1
            if cl>bl: bl,bs,be=cl,cs,i
        else: cl=0
    return angles[(bs+be)//2], r

def forward_clearance(ranges, angle_min, angle_inc):
    """Min distance in a narrow forward cone — what an obstacle ahead looks like."""
    r = np.array(ranges, dtype=np.float32)
    r = np.where(np.isfinite(r) & (r > 0.05), r, 30.0)
    angles = angle_min + np.arange(len(r)) * angle_inc
    cone = np.abs(angles) < EMERGENCY_CONE
    return float(r[cone].min()) if cone.any() else 30.0


# ══════════════════════════════════════════════════════════════════════════════
# LAP DETECTOR + TIMER
# ══════════════════════════════════════════════════════════════════════════════

class LapTracker:
    def __init__(self):
        self.start_x = self.start_y = None
        self.lap_count  = 0
        self.lap_times  = []
        self.best_lap   = float('inf')
        self.lap_start_t = None
        self.departed   = False

    def update(self, x, y):
        """Returns True if a new lap just completed."""
        now = time.time()
        if self.start_x is None:
            self.start_x, self.start_y = x, y
            self.lap_start_t = now
            return False

        dist = np.hypot(x-self.start_x, y-self.start_y)

        if not self.departed:
            if dist > LAP_DEPART_DIST:
                self.departed = True
            return False

        if dist < LAP_CLOSE_DIST:
            lap_t = now - self.lap_start_t
            self.lap_count  += 1
            self.lap_times.append(lap_t)
            if lap_t < self.best_lap:
                self.best_lap = lap_t
            self.departed    = False
            self.lap_start_t = now
            return True
        return False

    def current_lap_time(self):
        if self.lap_start_t is None: return 0.0
        return time.time() - self.lap_start_t

    def summary(self):
        if not self.lap_times: return 'No laps completed'
        times_str = ' | '.join([f'L{i+1}:{t:.1f}s' for i,t in enumerate(self.lap_times)])
        return f'{times_str} | Best: {self.best_lap:.1f}s'


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AGENT
# ══════════════════════════════════════════════════════════════════════════════

class RacingAgent(Node):

    def __init__(self):
        super().__init__('racing_agent')

        # State
        self.x = self.y = self.yaw = self.speed = 0.0
        self.latest_scan = None
        self._log_count  = 0
        self.prev_steer  = 0.0      # for steering low-pass
        self._idx_locked = False    # one-time global nearest-point lock

        # Per-lap path storage: list of (path_x, path_y) per lap
        self.all_laps_paths: list = []
        self.current_path_x: list = []
        self.current_path_y: list = []

        # Raceline
        self.rl_x = self.rl_y = None
        self.rl_speed = self.rl_curv = None
        self.nearest_idx = 0
        self.profile: TrackProfile = None
        self.raceline_speed_cap = 0.0   # applied on top of profile scale

        # Controller params (defaults, overridden by profile)
        self.max_steer      = 0.40
        self.blocked_thresh = 0.8
        self.min_lookahead  = 0.8
        self.max_lookahead  = 2.2
        self.lookahead_k    = 0.30

        # Lap tracking
        self.lap_tracker = LapTracker()

        # Training day mode flag
        self.training_day = not bool(find_best_raceline())
        if self.training_day:
            self.get_logger().info(
                '=== TRAINING DAY MODE ==='
                ' No saved raceline found. Will learn from scratch.'
            )
        else:
            self.get_logger().info(
                '=== RACE DAY MODE ==='
                ' Saved raceline found. Loading at full speed.'
            )

        # Load or start fresh
        existing = find_best_raceline()
        if existing and not self.training_day:
            self._activate_raceline(existing, cap=1.0)
        # If training day, lap 1 is always gap-follow regardless

        # ROS — topics are parameters so the same agent runs in sim and on the
        # real car.  Sim publishes /ego_racecar/odom; f1tenth_system uses /odom.
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        scan_topic  = self.get_parameter('scan_topic').value
        odom_topic  = self.get_parameter('odom_topic').value
        drive_topic = self.get_parameter('drive_topic').value
        self.create_subscription(LaserScan, scan_topic,  self.scan_cb, 10)
        self.create_subscription(Odometry,  odom_topic,  self.odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.create_timer(0.025, self.control_loop)
        self.get_logger().info(
            f'topics: scan={scan_topic} odom={odom_topic} drive={drive_topic}')

        self.get_logger().info(
            f'Agent ready | training_day={self.training_day} | '
            f'lap_tracker initialized'
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _activate_raceline(self, path_or_arrays, cap=1.0):
        """Load raceline from file path or (x,y,hdg,curv,spd) tuple and activate."""
        if isinstance(path_or_arrays, str):
            arrays = load_raceline(path_or_arrays)
        else:
            arrays = path_or_arrays

        self.rl_x, self.rl_y, self.rl_hdg, self.rl_curv, self.rl_speed = arrays
        self.profile = TrackProfile(self.rl_curv, self.rl_speed)
        self.raceline_speed_cap = cap

        # Apply both profile scale and lap cap
        base_speed = self.rl_speed * self.profile.speed_scale * cap
        self.rl_speed_active = np.clip(base_speed, MIN_SPEED, MAX_SPEED)

        # Push profile params to controller
        self.max_steer      = self.profile.max_steer
        self.blocked_thresh = self.profile.blocked_thresh
        self.min_lookahead  = self.profile.min_lookahead
        self.max_lookahead  = self.profile.max_lookahead
        self.lookahead_k    = self.profile.lookahead_k
        self.nearest_idx    = 0

        self.get_logger().info(
            f'Raceline activated | {self.profile.summary()} | '
            f'cap={cap:.0%} | '
            f'speed {self.rl_speed_active.min():.1f}–{self.rl_speed_active.max():.1f} m/s'
        )

    def _record_pos(self, x, y):
        if not self.current_path_x:
            self.current_path_x.append(x); self.current_path_y.append(y); return
        if np.hypot(x-self.current_path_x[-1], y-self.current_path_y[-1]) >= MIN_WPT_DIST:
            self.current_path_x.append(x); self.current_path_y.append(y)

    def _on_lap_complete(self, lap_num, lap_time):
        """Called when a lap finishes. Decides what to do next."""
        self.get_logger().info(
            f'=== LAP {lap_num} COMPLETE | {lap_time:.2f}s | '
            f'{self.lap_tracker.summary()} ==='
        )

        # Save this lap's path
        self.all_laps_paths.append(
            (self.current_path_x.copy(), self.current_path_y.copy())
        )
        self.current_path_x.clear()
        self.current_path_y.clear()

        if not self.training_day:
            return  # race day: just keep going at full speed

        # Training day: re-fit raceline and raise speed cap
        next_cap = LAP_SPEED_CAPS.get(lap_num + 1, 1.0)

        if lap_num >= 1 and len(self.all_laps_paths) >= 1:
            self.get_logger().info(
                f'Fitting raceline from {lap_num} lap(s)... '
                f'Next speed cap: {next_cap:.0%}'
            )
            try:
                # Merge paths from all laps with recency weighting
                laps = self.all_laps_paths
                weights = [LAP_PATH_WEIGHTS.get(i+1, 1.0) for i in range(len(laps))]
                merged_x, merged_y = merge_paths(laps, weights)

                arrays = fit_raceline(merged_x, merged_y)
                xf, yf, hdg, curv, spd = arrays

                # Save with lap number tag
                out = os.path.join(RACELINE_DIR, f'lap{lap_num}_raceline.csv')
                save_raceline(out, xf, yf, hdg, curv, spd)

                # If this is the fastest lap so far, save as best
                if self.lap_tracker.best_lap == lap_time or lap_num >= 4:
                    best_out = os.path.join(RACELINE_DIR, 'best_raceline.csv')
                    save_raceline(best_out, xf, yf, hdg, curv, spd)
                    self.get_logger().info(f'Best raceline updated → {best_out}')

                self._activate_raceline((xf, yf, hdg, curv, spd), cap=next_cap)
                self.get_logger().info(
                    f'Speed unlocked to {next_cap:.0%} for lap {lap_num+1}'
                )
            except Exception as e:
                self.get_logger().error(f'Raceline fit failed: {e}')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.speed = np.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        q = msg.pose.pose.orientation
        _, _, self.yaw = quat2euler([q.w, q.x, q.y, q.z])
        self._record_pos(self.x, self.y)

        if self.lap_tracker.update(self.x, self.y):
            lc = self.lap_tracker.lap_count
            lt = self.lap_tracker.lap_times[-1]
            self._on_lap_complete(lc, lt)

    def scan_cb(self, msg):
        self.latest_scan = msg

    # ── Control Loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        if self.latest_scan is None:
            return

        scan = self.latest_scan
        lc   = self.lap_tracker.lap_count       # laps completed so far
        drive_msg = AckermannDriveStamped()

        fwd_min = forward_clearance(
            scan.ranges, scan.angle_min, scan.angle_increment
        )

        # The raceline is authoritative once we have one (race day, or after the
        # first exploration lap on training day).  Lidar is a safety overlay only.
        use_pursuit = (self.rl_x is not None) and (lc >= 1 or not self.training_day)

        if use_pursuit:
            # ── Pure Pursuit on the optimized raceline ───────────────────────
            if not self._idx_locked:
                # Global lock: car may start anywhere on the loop.
                d = np.hypot(self.rl_x - self.x, self.rl_y - self.y)
                self.nearest_idx = int(np.argmin(d))
                self._idx_locked = True
            self.nearest_idx = find_nearest(
                self.x, self.y, self.rl_x, self.rl_y, self.nearest_idx
            )
            L = np.clip(
                self.lookahead_k*self.speed + self.min_lookahead,
                self.min_lookahead, self.max_lookahead
            )
            # Shorten lookahead for the sharpest curvature COMING UP, not just at
            # the current point — so the car turns in early on a decreasing-radius
            # hairpin instead of running wide and clipping the outer wall.
            n_rl0 = len(self.rl_curv)
            look_n = int(np.clip(self.speed * 0.9, 4, 9))
            curv_ahead = max(abs(float(self.rl_curv[(self.nearest_idx + k) % n_rl0]))
                             for k in range(look_n + 1))
            L = min(L, max(0.7, 1.2/(curv_ahead + 0.5)))
            tx, ty, _ = find_lookahead(
                self.x, self.y, self.yaw,
                self.rl_x, self.rl_y, L, self.nearest_idx
            )
            pp = pp_steer(self.x, self.y, self.yaw, tx, ty, L, self.max_steer)

            # Anticipatory speed: brake for the slowest raceline point within
            # braking range ahead, not just the current one (whose index lags).
            n_rl = len(self.rl_speed_active)
            n_ahead = int(np.clip(self.speed * 0.9, 4, 9))
            base_spd = float(min(
                self.rl_speed_active[(self.nearest_idx + k) % n_rl]
                for k in range(n_ahead + 1)
            ))

            # Steering is ALWAYS pure pursuit — it is the only thing that knows
            # the line.  Handing steering to a gap-follower on a clear track just
            # fights the raceline and throws the car into orbits.  Lidar only
            # ever brakes (for an opponent ahead it cannot otherwise see).
            steer = pp
            tspd  = base_spd * (1.0 - 0.30*(abs(pp)/self.max_steer))

            # Emergency brake ONLY for something genuinely in the path and close
            # (an opponent).  Keep enough speed to still steer out — braking to a
            # crawl pins the car against corner walls and it can never recover.
            if fwd_min < EMERGENCY_DIST:
                tspd = min(tspd, 2.0)
                mode_str = f'BRAKE d={fwd_min:.2f}m wp={self.nearest_idx}'
            else:
                mode_str = (f'PURSUIT cap={self.raceline_speed_cap:.0%} '
                            f'wp={self.nearest_idx} L={L:.2f}m')

            # Low-pass the steering to damp high-speed oscillation.
            steer = STEER_SMOOTH*steer + (1.0-STEER_SMOOTH)*self.prev_steer

        else:
            # ── Gap follow (training lap 1 / no raceline yet) ────────────────
            # Steer TOWARD the gap (positive lidar angle = left = positive steer).
            gap_angle, _ = best_gap(scan.ranges, scan.angle_min,
                                    scan.angle_increment)
            steer = np.clip(gap_angle*0.9, -self.max_steer, self.max_steer)
            tspd  = EXPLORE_SPEED * np.clip((fwd_min-0.5)/3.0, 0.25, 1.0)
            mode_str = (f'GAP_FOLLOW lap={lc} '
                        f'pts={len(self.current_path_x)} '
                        f't={self.lap_tracker.current_lap_time():.0f}s')

        steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        self.prev_steer = steer
        drive_msg.drive.steering_angle = steer
        drive_msg.drive.speed          = float(max(tspd, 0.5))
        self.drive_pub.publish(drive_msg)

        self._log_count += 1
        if self._log_count % 20 == 0:
            self.get_logger().info(
                f'[{mode_str}] spd={tspd:.2f} '
                f'steer={np.degrees(steer):.1f}° '
                f'pos=({self.x:.1f},{self.y:.1f})'
            )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    try:
        node = RacingAgent()
        rclpy.spin(node)
    except (FileNotFoundError, KeyboardInterrupt):
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
