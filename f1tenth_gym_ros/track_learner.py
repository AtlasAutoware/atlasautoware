"""
Track learner — live "watch the optimal raceline build itself" node.
=====================================================================

This is the heart of the practice-session model: while you drive a new track
(hands-free via `mapping_driver`, or hand-teleop), `slam_toolbox` fuses the
VESC odometry + lidar into a live occupancy map.  This node watches that map
(`/map`) and, every few seconds, re-runs the minimum-curvature optimizer on the
*current* state of the map — so the racing line slowly sharpens lap after lap as
the walls fill in and loop-closure tightens the geometry.

  drive laps ──► slam_toolbox ──► /map ──► (this node) ──► learned_raceline.csv
                 (VESC + lidar)            re-optimize live   + overlay PNG

Pipeline per optimize pass (runs off the ROS thread so it never stalls intake):
  1. snapshot the latest /map OccupancyGrid,
  2. write it as a ROS map_server PGM + YAML (the exact format the optimizer
     loads — reusing all of the verified offline pipeline, nothing forked),
  3. seed the corridor flood-fill at the car's current map-frame pose (from TF),
  4. optimize -> write racelines/learned_raceline.csv (+ _overlay.png),
  5. log coverage, length, est lap time, feasibility — so you watch it converge.

Early in a session the loop isn't closed yet; the optimizer can't trace a
corridor, so the pass is skipped and retried — the line simply appears once the
track first closes, then refines.  On shutdown the best line is promoted to
best_raceline.csv so the race agent picks it up.

    ros2 run f1tenth_gym_ros track_learner
    ros2 run f1tenth_gym_ros track_learner --ros-args -p interval:=4.0
"""

import os
import shutil
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import tf2_ros
from transforms3d.euler import quat2euler

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raceline_optimizer as ro


class TrackLearner(Node):
    def __init__(self):
        super().__init__('track_learner')

        # ── parameters (all overridable via --ros-args -p) ─────────────────────
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'ego_racecar/base_link')
        self.declare_parameter('interval', 5.0)        # s between optimize passes
        self.declare_parameter('min_growth', 0.02)     # re-opt only if known-area
                                                       # grew >=2% since last pass
        self.declare_parameter('margin', 0.30)         # wall clearance (m)
        self.declare_parameter('apex_bias', 1.0)       # late-apex emphasis
        self.declare_parameter('a_lat', 6.5)           # cornering grip (m/s^2)
        self.declare_parameter('v_max', 7.0)           # top speed (m/s)
        # Output dirs default to the source repo (works when run from source or
        # symlink-installed). If your build *copies* the package, set these to
        # the repo so the line lands where the follower looks for it.
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.declare_parameter('runtime_dir', os.path.join(repo, 'runtime'))
        self.declare_parameter('racelines_dir', os.path.join(repo, 'racelines'))
        self.map_topic  = self.get_parameter('map_topic').value
        self.map_frame  = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.interval   = float(self.get_parameter('interval').value)
        self.min_growth = float(self.get_parameter('min_growth').value)

        self._runtime = self.get_parameter('runtime_dir').value
        self._racelines = self.get_parameter('racelines_dir').value
        os.makedirs(self._runtime, exist_ok=True)
        os.makedirs(self._racelines, exist_ok=True)
        self._map_pgm  = os.path.join(self._runtime, 'live_map.pgm')
        self._map_yaml = os.path.join(self._runtime, 'live_map.yaml')
        self._out_csv  = os.path.join(self._racelines, 'learned_raceline.csv')
        self._best_csv = os.path.join(self._racelines, 'best_raceline.csv')

        # ── shared state (map snapshot is swapped under the lock) ───────────────
        self._lock = threading.Lock()
        self._latest = None            # (grid uint8 0..100/-1, info) snapshot
        self._last_known = 0           # known-cell count at last optimize
        self._passes = 0
        self._best_lap = float('inf')

        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)
        self.create_subscription(OccupancyGrid, self.map_topic, self._map_cb, 1)

        self._stop = False
        threading.Thread(target=self._optimize_loop, daemon=True).start()
        self.get_logger().info(
            f'track_learner up — watching {self.map_topic}; re-optimizing every '
            f'{self.interval:.0f}s. Drive clean laps; the line sharpens as you go.')

    # ── map intake ─────────────────────────────────────────────────────────────
    def _map_cb(self, msg: OccupancyGrid):
        info = msg.info
        if info.width == 0 or info.height == 0:
            return
        try:
            grid = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        except ValueError:
            return                                   # data/size mismatch — skip frame
        with self._lock:
            self._latest = (grid, info)

    # ── seed pose: where the car is, in the map frame (for the flood-fill) ──────
    def _seed_pose(self):
        """Car pose in the map frame from TF; (None) if TF not ready yet."""
        try:
            tf = self.tf_buf.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            _, _, yaw = quat2euler([q.w, q.x, q.y, q.z])
            return float(t.x), float(t.y), float(yaw)
        except Exception:
            return None

    # ── write the live OccupancyGrid as a map_server PGM + YAML ─────────────────
    def _write_map(self, grid, info):
        """OccupancyGrid -> grayscale PGM (+ YAML) in the convention GridMap reads.

        map_server: occupied->black(0), free->white(254), unknown->mid(150 so it
        is neither free nor occ under the optimizer's thresholds, which keeps the
        corridor flood-fill bounded to *explored* free space).  OccupancyGrid row
        0 is the origin (bottom) row; image row 0 is the top, so flip vertically.
        """
        from PIL import Image
        img = np.full(grid.shape, 150, np.uint8)        # unknown
        img[(grid >= 0) & (grid <= 25)] = 254           # free
        img[grid >= 65] = 0                             # occupied (wall)
        img = np.flipud(img)                            # bottom-origin -> top-row
        Image.fromarray(img, mode='L').save(self._map_pgm)

        ox = float(info.origin.position.x)
        oy = float(info.origin.position.y)
        with open(self._map_yaml, 'w') as f:
            f.write(f'image: {os.path.basename(self._map_pgm)}\n')
            f.write(f'resolution: {float(info.resolution):.6f}\n')
            f.write(f'origin: [{ox:.6f}, {oy:.6f}, 0.0]\n')
            f.write('negate: 0\n')
            f.write('occupied_thresh: 0.65\n')
            f.write('free_thresh: 0.25\n')

    # ── background optimize loop (never on the ROS callback thread) ─────────────
    def _optimize_loop(self):
        while not self._stop:
            time.sleep(self.interval)
            with self._lock:
                snap = self._latest
            if snap is None:
                continue
            grid, info = snap
            known = int(np.count_nonzero(grid >= 0))
            total = grid.size
            if known == 0:
                continue
            # Skip if the map hasn't meaningfully grown since the last pass.
            grew = (known - self._last_known) / float(total)
            if self._passes > 0 and grew < self.min_growth:
                continue

            seed = self._seed_pose()
            sx, sy, syaw = seed if seed else (0.0, 0.0, 0.0)
            try:
                self._write_map(grid, info)
                vp = ro.VehicleParams()
                vp.a_lat_max = float(self.get_parameter('a_lat').value)
                vp.v_max     = float(self.get_parameter('v_max').value)
                vp.safety_marg = float(self.get_parameter('margin').value)
                race, hdg, curv, spd = ro.optimize(
                    self._map_yaml, self._out_csv,
                    seed_world=(sx, sy), seed_heading=syaw,
                    vp=vp, overlay=True, verbose=False,
                    apex_bias=float(self.get_parameter('apex_bias').value))
            except Exception as e:
                self.get_logger().warn(
                    f'map not lap-complete yet (pass skipped: {type(e).__name__}: '
                    f'{e}) — keep driving, the line appears once the loop closes')
                continue

            self._last_known = known
            self._passes += 1
            n = len(race)
            ds = np.hypot(*(np.roll(race, -1, axis=0) - race).T)
            lap_t = float(np.sum(ds / np.maximum(spd, 1e-3)))
            n_bad = int((curv > vp.kappa_budget).sum())
            cover = 100.0 * known / float(total)
            feas = 'feasible' if n_bad == 0 else f'{n_bad} tight pt(s)'
            self.get_logger().info(
                f'[pass {self._passes}] coverage {cover:.0f}% | line {n} pts, '
                f'{ds.sum():.0f} m | est lap {lap_t:.1f}s | {feas} | '
                f'-> {os.path.basename(self._out_csv)}')

            # Promote the best (shortest est-lap) feasible line to best_raceline.
            if n_bad == 0 and lap_t < self._best_lap:
                self._best_lap = lap_t
                shutil.copy(self._out_csv, self._best_csv)
                self.get_logger().info(
                    f'  new best line (est {lap_t:.1f}s) -> '
                    f'{os.path.basename(self._best_csv)}')

    def destroy_node(self):
        self._stop = True
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrackLearner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
