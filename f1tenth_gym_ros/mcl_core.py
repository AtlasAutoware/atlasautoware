#!/usr/bin/env python3
"""Monte-Carlo localization core (ROS-free, numpy only).

A particle filter that localizes a 2-D lidar against a known occupancy map and returns
the map-frame pose. The sensor model is a likelihood field: the map's obstacles are
distance-transformed once, so scoring a beam endpoint is one array lookup instead of a
ray cast. That is what makes a few hundred particles run at scan rate on the Orin without
range_libc or CUDA.

Frames follow the F1TENTH convention: the filter estimates map -> base_link, and the ROS
wrapper publishes map -> odom (given odom -> base_link from wheel odometry) plus an
Odometry on /pf/pose/odom in the map frame, which is what raceline_mpc consumes.

Everything here is testable without ROS; mcl_localization.py is the thin node around it.
"""
import numpy as np


class LikelihoodField:
    """Occupancy grid + distance-to-nearest-obstacle, in metres."""

    def __init__(self, occ, resolution, origin, z_hit=0.9, z_rand=0.1, sigma_hit=0.12,
                 max_dist=2.0):
        # occ: bool array [H, W], True = occupied. origin: (x, y) of pixel (row 0, col 0)
        # lower-left corner in map frame (ROS map yaml convention), theta assumed 0.
        from scipy import ndimage
        self.res = float(resolution)
        self.ox, self.oy = float(origin[0]), float(origin[1])
        self.H, self.W = occ.shape
        free = ~occ
        # distance (px) from each cell to the nearest occupied cell, capped
        d = ndimage.distance_transform_edt(free) * self.res
        self.dist = np.minimum(d, max_dist).astype(np.float32)
        self.occ = occ
        self.z_hit, self.z_rand, self.sigma = z_hit, z_rand, sigma_hit
        self.max_range_default = max_dist
        # The beam log-likelihood is a function of the cell distance alone, and the
        # distances are fixed once the map is loaded — so bake it into the map. score()
        # then never evaluates exp or log, which was two thirds of its cost.
        self._ll = np.log(z_hit * np.exp(-(self.dist * self.dist) /
                                         (2.0 * sigma_hit * sigma_hit)) + z_rand
                          ).astype(np.float32).ravel()
        self._ll_oob = np.float32(np.log(z_hit * np.exp(-(max_dist * max_dist) /
                                                        (2.0 * sigma_hit * sigma_hit)) + z_rand))
        self._inv_res = np.float32(1.0 / self.res)
        self._ox32, self._oy32 = np.float32(self.ox), np.float32(self.oy)

    def world_to_px(self, x, y):
        # ROS maps: image row 0 is the TOP, which is the MAX y. origin is the lower-left.
        col = (x - self.ox) / self.res
        row = self.H - (y - self.oy) / self.res
        return col, row

    def sample_free(self, n, rng):
        """n random (x, y, theta) on free cells — for global init."""
        ys, xs = np.where(~self.occ)
        idx = rng.integers(0, len(xs), n)
        col = xs[idx] + rng.random(n)
        row = ys[idx] + rng.random(n)
        x = self.ox + col * self.res
        y = self.oy + (self.H - row) * self.res
        th = rng.uniform(-np.pi, np.pi, n)
        return np.stack([x, y, th], 1)

    def score(self, particles, ex, ey):
        """Log-likelihood of each particle given beam endpoints (ex, ey) in the SENSOR
        frame (metres, x forward, y left). particles: [N, 3] (x, y, theta)."""
        p = np.asarray(particles, np.float32)
        ex = np.asarray(ex, np.float32); ey = np.asarray(ey, np.float32)
        c, s = np.cos(p[:, 2]), np.sin(p[:, 2])
        # world endpoints for every particle: [N, B]. float32 throughout — the map cells
        # are 5 cm, so single precision is three orders of magnitude finer than the grid.
        wx = p[:, 0:1] + c[:, None] * ex - s[:, None] * ey
        wy = p[:, 1:2] + s[:, None] * ex + c[:, None] * ey
        col = ((wx - self._ox32) * self._inv_res).astype(np.int32)
        row = (np.float32(self.H) - (wy - self._oy32) * self._inv_res).astype(np.int32)
        inb = col >= 0; inb &= col < self.W; inb &= row >= 0; inb &= row < self.H
        # clamp and gather unconditionally, then patch the strays: one flat take beats
        # a boolean fancy-index on both sides of an assignment
        np.clip(col, 0, self.W - 1, out=col); np.clip(row, 0, self.H - 1, out=row)
        flat = row; flat *= self.W; flat += col
        ll = self._ll[flat]
        return np.where(inb, ll, self._ll_oob).sum(1, dtype=np.float32)


class ParticleFilter:
    def __init__(self, field, n_particles=600, beams=90, motion_noise=(0.08, 0.08, 0.05),
                 seed=0):
        self.f = field
        self.n = int(n_particles)
        self.beams = int(beams)
        self.a1, self.a2, self.a3 = motion_noise      # trans-from-trans, rot-from-rot, trans-from-rot
        self.rng = np.random.default_rng(seed)
        self.P = None                                  # [N, 3]
        self.w = None
        self.last_odom = None
        self._beams_key = None                         # subsample layout of the last scan

    def init_global(self):
        self.P = self.f.sample_free(self.n, self.rng)
        self.w = np.full(self.n, 1.0 / self.n)
        self.last_odom = None

    def init_pose(self, x, y, theta, spread=(0.3, 0.3, 0.2)):
        self.P = np.column_stack([
            self.rng.normal(x, spread[0], self.n),
            self.rng.normal(y, spread[1], self.n),
            self.rng.normal(theta, spread[2], self.n)])
        self.w = np.full(self.n, 1.0 / self.n)
        self.last_odom = None

    def predict(self, odom_xytheta):
        """Odometry motion model: move particles by the delta since the last odom, with noise."""
        if self.last_odom is None:
            self.last_odom = np.asarray(odom_xytheta, float); return
        o0 = self.last_odom; o1 = np.asarray(odom_xytheta, float)
        dx, dy = o1[0] - o0[0], o1[1] - o0[1]
        trans = np.hypot(dx, dy)
        rot1 = np.arctan2(dy, dx) - o0[2] if trans > 1e-3 else 0.0
        dth = np.arctan2(np.sin(o1[2] - o0[2]), np.cos(o1[2] - o0[2]))
        rot2 = dth - rot1
        self.last_odom = o1
        N = self.n
        rot1n = rot1 - self.rng.normal(0, self.a2 * abs(rot1) + self.a3 * trans, N)
        transn = trans - self.rng.normal(0, self.a1 * trans + self.a3 * (abs(rot1) + abs(rot2)), N)
        rot2n = rot2 - self.rng.normal(0, self.a2 * abs(rot2) + self.a3 * trans, N)
        th = self.P[:, 2]
        self.P[:, 0] += transn * np.cos(th + rot1n)
        self.P[:, 1] += transn * np.sin(th + rot1n)
        self.P[:, 2] = self._wrap(th + rot1n + rot2n)

    def update(self, ranges, angle_min, angle_inc, max_range):
        """Reweight by the scan (subsampled to self.beams), then resample if degenerate."""
        r = np.asarray(ranges, np.float32)
        n = len(r)
        key = (n, angle_min, angle_inc)
        if key != self._beams_key:                 # the lidar layout is fixed; the bearings
            step = max(1, n // self.beams)         # and their sin/cos are not per-scan work
            idx = np.arange(0, n, step)
            ang = (angle_min + angle_inc * idx).astype(np.float32)
            self._beams_key = key
            self._beam_idx = idx
            self._beam_cos = np.cos(ang); self._beam_sin = np.sin(ang)
        r = r[self._beam_idx]
        # NaN fails `> 0.05`, so isfinite is redundant
        good = (r > 0.05) & (r < max_range)
        if good.sum() < 5:
            return
        r = r[good]
        ex = r * self._beam_cos[good]; ey = r * self._beam_sin[good]
        ll = self.f.score(self.P, ex, ey)
        ll -= ll.max()
        w = np.exp(ll) * self.w
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            return
        self.w = w / s
        neff = 1.0 / np.sum(self.w ** 2)
        if neff < self.n / 2.0:
            self._resample()

    def _resample(self):
        # low-variance resampling + a little roughening to fight particle depletion
        pos = (self.rng.random() + np.arange(self.n)) / self.n
        cdf = np.cumsum(self.w)
        idx = np.searchsorted(cdf, pos)
        idx = np.clip(idx, 0, self.n - 1)
        self.P = self.P[idx].copy()
        self.P[:, :2] += self.rng.normal(0, 0.02, (self.n, 2))
        self.P[:, 2] += self.rng.normal(0, 0.01, self.n)
        self.w = np.full(self.n, 1.0 / self.n)

    def estimate(self):
        """Weighted mean pose; theta via circular mean."""
        x = np.sum(self.w * self.P[:, 0]); y = np.sum(self.w * self.P[:, 1])
        th = np.arctan2(np.sum(self.w * np.sin(self.P[:, 2])),
                        np.sum(self.w * np.cos(self.P[:, 2])))
        # spread (position std) as a rough confidence
        var = np.sum(self.w * ((self.P[:, 0] - x) ** 2 + (self.P[:, 1] - y) ** 2))
        return float(x), float(y), float(th), float(np.sqrt(var))

    @staticmethod
    def _wrap(a):
        return np.arctan2(np.sin(a), np.cos(a))


def load_map(yaml_path):
    """Read a ROS map yaml + image -> (occ bool[H,W], resolution, origin)."""
    import os, yaml
    from PIL import Image
    meta = yaml.safe_load(open(yaml_path))
    img_path = meta['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(yaml_path), img_path)
    g = np.asarray(Image.open(img_path).convert('L'))
    res = float(meta['resolution']); origin = meta['origin']
    negate = int(meta.get('negate', 0))
    occ_th = float(meta.get('occupied_thresh', 0.65))
    p = (255 - g) / 255.0 if not negate else g / 255.0     # occupancy probability
    occ = p > occ_th
    return occ, res, origin
