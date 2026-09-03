#!/usr/bin/env python3
"""Goal-reaching expert core (ROS-free): A* on the occupancy map + pure pursuit, plus a
route-to-language labeller. This is the sim's *goal-conditioned* expert: given any
(start, goal) it produces a drivable path, the controls to follow it, and an
instruction string describing the route, so the sim can generate diverse
language-labelled demonstrations that are not just laps.
"""
import heapq, math
import numpy as np
from scipy import ndimage


class Planner:
    def __init__(self, occ, resolution, origin, inflate_m=0.28, cell_m=0.10):
        """occ: bool[H,W] map grid. Plans on a coarser inflated grid (cell_m) for speed."""
        self.res = float(resolution); self.ox, self.oy = float(origin[0]), float(origin[1])
        self.H, self.W = occ.shape
        # inflate obstacles by the car's half-width + margin, then downsample
        r = max(1, int(round(inflate_m / self.res)))
        inflated = ndimage.binary_dilation(occ, iterations=r)
        f = max(1, int(round(cell_m / self.res)))
        self.f = f; self.cell = self.res * f
        Hc, Wc = self.H // f, self.W // f
        self.grid = inflated[:Hc * f, :Wc * f].reshape(Hc, f, Wc, f).any(axis=(1, 3))
        self.Hc, self.Wc = self.grid.shape

    def w2c(self, x, y):
        col = int((x - self.ox) / self.cell); row = int((self.H - (y - self.oy) / self.res) / self.f)
        return row, col

    def c2w(self, row, col):
        x = self.ox + (col + 0.5) * self.cell
        y = self.oy + (self.H - (row + 0.5) * self.f) * self.res
        return x, y

    def free(self, x, y):
        r, c = self.w2c(x, y)
        return 0 <= r < self.Hc and 0 <= c < self.Wc and not self.grid[r, c]

    def sample_free(self, rng, n=1):
        rows, cols = np.where(~self.grid)
        idx = rng.integers(0, len(rows), n)
        return [self.c2w(rows[i], cols[i]) for i in idx]

    def sample_free_near(self, rng, center, r_min, r_max):
        """A free point whose straight-line distance from `center` is in [r_min, r_max]."""
        cr, cc = self.w2c(*center); R = int(r_max / self.cell) + 1
        r0, r1 = max(0, cr - R), min(self.Hc, cr + R + 1); c0, c1 = max(0, cc - R), min(self.Wc, cc + R + 1)
        sub = ~self.grid[r0:r1, c0:c1]
        rows, cols = np.where(sub)
        if len(rows) == 0: return None
        for _ in range(50):
            i = rng.integers(0, len(rows)); x, y = self.c2w(rows[i] + r0, cols[i] + c0)
            d = math.hypot(x - center[0], y - center[1])
            if r_min <= d <= r_max: return (x, y)
        return None

    def plan(self, start, goal):
        """A* (8-connected). Returns list of (x,y) world points or None."""
        s = self.w2c(*start); g = self.w2c(*goal)
        if not (self.free(*start) and self.free(*goal)):
            return None
        H, W = self.Hc, self.Wc
        def h(a): return math.hypot(a[0] - g[0], a[1] - g[1])
        openq = [(h(s), 0.0, s)]; came = {s: None}; gcost = {s: 0.0}
        nbrs = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        while openq:
            _, gc, cur = heapq.heappop(openq)
            if cur == g:
                path = []
                while cur is not None: path.append(self.c2w(*cur)); cur = came[cur]
                return self._smooth(path[::-1])
            if gc > gcost.get(cur, 1e18): continue
            for dr, dc, w in nbrs:
                nr, nc = cur[0] + dr, cur[1] + dc
                if not (0 <= nr < H and 0 <= nc < W) or self.grid[nr, nc]: continue
                if w > 1 and (self.grid[cur[0] + dr, cur[1]] or self.grid[cur[0], cur[1] + dc]): continue
                ng = gc + w
                if ng < gcost.get((nr, nc), 1e18):
                    gcost[(nr, nc)] = ng; came[(nr, nc)] = cur
                    heapq.heappush(openq, (ng + h((nr, nc)), ng, (nr, nc)))
        return None

    def _smooth(self, path, passes=3):
        p = np.asarray(path, float)
        if len(p) < 3: return [tuple(q) for q in p]
        for _ in range(passes):
            q = p.copy()
            q[1:-1] = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
            ok = np.array([self.free(x, y) for x, y in q])
            p = np.where(ok[:, None], q, p)
        return [tuple(q) for q in p]


def pure_pursuit(pose, path, lookahead=0.6, wheelbase=0.33, max_steer=0.4, v_max=1.5,
                 v_min=0.4, goal_radius=0.35):
    """pose (x,y,theta). Returns (speed, steer, done, dist_to_goal)."""
    x, y, th = pose
    P = np.asarray(path)
    d = np.hypot(P[:, 0] - x, P[:, 1] - y)
    i = int(np.argmin(d))
    dist_goal = float(np.hypot(P[-1, 0] - x, P[-1, 1] - y))
    if dist_goal < goal_radius:
        return 0.0, 0.0, True, dist_goal
    # first point at least `lookahead` ahead of the nearest
    j = i
    while j < len(P) - 1 and np.hypot(P[j, 0] - x, P[j, 1] - y) < lookahead: j += 1
    tx, ty = P[j]
    dx, dy = tx - x, ty - y
    lx = math.cos(-th) * dx - math.sin(-th) * dy      # target in car frame
    ly = math.sin(-th) * dx + math.cos(-th) * dy
    ld = max(0.2, math.hypot(lx, ly))
    curv = 2.0 * ly / (ld * ld)
    steer = max(-max_steer, min(max_steer, math.atan(wheelbase * curv)))
    v = v_max * max(0.3, 1.0 - abs(steer) / max_steer * 0.7)   # slow in turns
    v = min(v, max(v_min, 1.2 * dist_goal))                    # slow into the goal
    if lx < 0: v = v_min                                        # target behind: creep + turn
    return float(v), float(steer), False, dist_goal


def describe_route(path, turn_thresh=math.radians(35)):
    """Turn a path into a short natural-language instruction from its heading changes."""
    P = np.asarray(path)
    if len(P) < 4: return 'drive forward and stop'
    seg = np.diff(P, axis=0); head = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    L = np.hypot(seg[:, 0], seg[:, 1]).cumsum(); total = float(L[-1])
    steps, acc, last = [], 0.0, head[0]
    for k in range(1, len(head)):
        acc = head[k] - last
        if abs(acc) > turn_thresh:
            steps.append(('left' if acc > 0 else 'right', float(L[k] / total))); last = head[k]
    parts = []
    if not steps:
        parts.append('go straight')
    else:
        pos = 0.0
        for side, frac in steps:
            gap = frac - pos
            lead = 'turn ' if not parts else ('then turn ' if gap < 0.35 else 'go straight, then turn ')
            parts.append(f'{lead}{side}'); pos = frac
        if 1.0 - pos > 0.3: parts.append('then go straight')
    tail = ' to the end and stop' if total > 4.0 else ' and stop'
    s = ', '.join(parts) + tail
    return s[0].upper() + s[1:]
