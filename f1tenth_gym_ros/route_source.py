"""route_source: the car-side source of the policy's route hint (state[2:4]).

In the sim the hint comes from the task's planned path. On the car the same thing is
rebuilt from what the car has: its map-frame pose (SLAM via pose_relay, or the particle
filter, both on /pf/pose/odom), the occupancy map (/map), and a goal point. The route is
planned with the same A* planner (goal_core.Planner) the sim tasks were planned with, and
replanned about once a second so the hint follows the map as SLAM fills it in.

ROS-free on purpose so it can be unit-tested and run inside the simulator
(ml/car_route_check.py) exactly as the bridge runs it.
"""
import json, math
import numpy as np

try:
    from f1tenth_gym_ros.goal_core import Planner
    from f1tenth_gym_ros.policy_io import route_hint
except ImportError:
    from goal_core import Planner
    from policy_io import route_hint


def model_uses_hint(session):
    """True when the ONNX model was trained with the route hint unmasked (state_mask[2:4]).
    Models trained before 9/25 zero those slots, so they keep working unchanged."""
    try:
        meta = session.get_modelmeta().custom_metadata_map or {}
        mask = json.loads(meta.get('config', '{}')).get('state_mask', '0,0,0,0,0')
        m = [float(x) for x in str(mask).split(',')]
        return len(m) >= 4 and (m[2] != 0 or m[3] != 0)
    except Exception:
        return False


def occ_from_grid(data, width, height, occupied=50, unknown_blocked=True):
    """nav_msgs/OccupancyGrid data -> bool[H, W] in the image convention Planner uses.
    OccupancyGrid row 0 is at origin.y (bottom); map images and Planner put row 0 at the
    top, so the rows are flipped. Unknown (-1) counts as blocked by default: the route
    should not be planned through space the car has never seen."""
    g = np.asarray(data, np.int16).reshape(int(height), int(width))
    occ = g > occupied
    if unknown_blocked:
        occ |= g < 0
    return np.flipud(occ)


def parse_goal(s):
    """'x,y' in map metres -> (x, y); '' or junk -> None."""
    try:
        x, y = (float(v) for v in str(s).replace(' ', '').split(',')[:2])
        return (x, y) if math.isfinite(x) and math.isfinite(y) else None
    except (ValueError, TypeError):
        return None


class RouteSource:
    """Holds the map, goal and current route. replan() is the slow part (A*), meant for a
    background thread or a 1 Hz timer; hint() is cheap and runs every control tick."""

    def __init__(self, goal_tol=0.4, snap_m=0.6, keep_s=3.0, inflate_m=0.28, cell_m=0.10):
        self.goal_tol, self.snap_m, self.keep_s = goal_tol, snap_m, keep_s
        self.inflate_m, self.cell_m = inflate_m, cell_m
        self._map = None; self._map_dirty = False; self.planner = None
        self.goal = None; self.path = None; self.t_path = -1e9; self.plan_ms = 0.0; self.fail = ''

    def set_map(self, occ, resolution, origin):
        self._map = (occ, float(resolution), (float(origin[0]), float(origin[1]))); self._map_dirty = True

    def set_goal(self, goal):
        self.goal = None if goal is None else (float(goal[0]), float(goal[1]))
        self.path = None; self.t_path = -1e9

    def _snap(self, p):
        """Nearest planner-free point within snap_m (the car's own cell can sit inside the
        inflated wall margin, which A* would refuse as a start)."""
        pl = self.planner
        if pl.free(*p): return p
        r0, c0 = pl.w2c(*p); R = int(math.ceil(self.snap_m / pl.cell))
        best = None
        for r in range(max(0, r0 - R), min(pl.Hc, r0 + R + 1)):
            for c in range(max(0, c0 - R), min(pl.Wc, c0 + R + 1)):
                if pl.grid[r, c]: continue
                x, y = pl.c2w(r, c); d = math.hypot(x - p[0], y - p[1])
                if d <= self.snap_m and (best is None or d < best[0]): best = (d, (x, y))
        return None if best is None else best[1]

    def replan(self, pose, now):
        """Plan pose -> goal on the latest map. Returns True if a route is available."""
        import time as _t
        t0 = _t.perf_counter()
        if self._map_dirty and self._map is not None:
            occ, res, origin = self._map
            self.planner = Planner(occ, res, origin, inflate_m=self.inflate_m, cell_m=self.cell_m)
            self._map_dirty = False
        goal = self.goal                     # the bridge may set a new goal while A* runs
        if self.planner is None or goal is None or pose is None:
            return self.path is not None
        s = self._snap((pose[0], pose[1])); g = self._snap(goal)
        path = self.planner.plan(s, g) if (s is not None and g is not None) else None
        self.plan_ms = (_t.perf_counter() - t0) * 1000.0
        if goal != self.goal:
            return False                     # stale: planned for the previous goal
        if path is not None and len(path) >= 2:
            # the route has to end at the goal the user asked for, not the snapped cell
            self.path = [tuple(p) for p in path] + ([self.goal] if g != self.goal else [])
            self.t_path = now; self.fail = ''
            return True
        self.fail = 'start blocked' if s is None else 'goal blocked' if g is None else 'no route'
        return self.path is not None and now - self.t_path < self.keep_s

    def hint(self, pose, now):
        """-> (hint or None, status dict). None means: do not drive."""
        st = {'goal': self.goal}
        if self._map is None: st['route'] = 'no map'; return None, st
        if self.goal is None: st['route'] = 'no goal'; return None, st
        if pose is None: st['route'] = 'no pose'; return None, st
        d = math.hypot(self.goal[0] - pose[0], self.goal[1] - pose[1]); st['to_goal'] = round(d, 2)
        if d < self.goal_tol: st['route'] = 'arrived'; return None, st
        if self.path is None or now - self.t_path > self.keep_s:
            st['route'] = 'no route' + (f' ({self.fail})' if self.fail else ''); return None, st
        st['route'] = 'ok'; st['plan_ms'] = round(self.plan_ms, 1)
        return route_hint(pose, self.path), st
