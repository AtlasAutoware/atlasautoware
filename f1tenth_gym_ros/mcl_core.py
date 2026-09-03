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
        # precompute the per-distance likelihood is not possible (continuous), but the
        # gaussian is cheap; keep a constant background for z_rand
        self._norm = 1.0

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
        N = len(particles); B = len(ex)
        c, s = np.cos(particles[:, 2]), np.sin(particles[:, 2])
        # world endpoints for every particle: [N, B]
        wx = particles[:, 0:1] + np.outer(c, ex) - np.outer(s, ey)
        wy = particles[:, 1:2] + np.outer(s, ex) + np.outer(c, ey)
        col = ((wx - self.ox) / self.res).astype(np.int32)
        row = (self.H - (wy - self.oy) / self.res).astype(np.int32)
        inb = (col >= 0) & (col < self.W) & (row >= 0) & (row < self.H)
        d = np.full((N, B), self.max_range_default, np.float32)
        d[inb] = self.dist[row[inb], col[inb]]
        p = self.z_hit * np.exp(-(d * d) / (2 * self.sigma * self.sigma)) + self.z_rand
        return np.log(p).sum(1)


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
        step = max(1, n // self.beams)
        idx = np.arange(0, n, step)
        r = r[idx]; ang = angle_min + angle_inc * idx
        good = np.isfinite(r) & (r > 0.05) & (r < max_range)
        r, ang = r[good], ang[good]
        if len(r) < 5:
            return
        ex = r * np.cos(ang); ey = r * np.sin(ang)
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
