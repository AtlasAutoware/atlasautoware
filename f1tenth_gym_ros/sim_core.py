#!/usr/bin/env python3
"""Headless F1TENTH-style simulator core (ROS-free, numpy).

A kinematic bicycle on an occupancy map, a ray-cast 2-D lidar, and a crude forward
camera render, so the episode logger records sim drives in the SAME format as the car
(image, lidar, odom, action, instruction) for bulk data collection without hardware.

The camera view is a raycast wall render (distance-shaded), not photorealistic. It is
enough to exercise the whole pipeline and to train/evaluate a lidar-first or
depth-first policy; a vision policy trained on it will face a sim-to-real gap, which is
the honest limitation of any quick simulator.
"""
import math
import numpy as np


class SimMap:
    def __init__(self, occ, resolution, origin):
        self.occ = occ.astype(bool)
        self.res = float(resolution)
        self.ox, self.oy = float(origin[0]), float(origin[1])
        self.H, self.W = occ.shape

    def occupied(self, x, y):
        col = int((x - self.ox) / self.res)
        row = int(self.H - (y - self.oy) / self.res)
        if row < 0 or row >= self.H or col < 0 or col >= self.W:
            return True
        return bool(self.occ[row, col])

    def raycast(self, x, y, angles, max_range=16.0, step=None):
        """Distance to first obstacle along each absolute-world angle. Vectorized over beams."""
        step = step or self.res
        n = len(angles)
        ca, sa = np.cos(angles), np.sin(angles)
        out = np.full(n, max_range, np.float32)
        hit = np.zeros(n, bool)
        r = np.full(n, step)
        steps = int(max_range / step)
        for _ in range(steps):
            px = x + r * ca; py = y + r * sa
            col = ((px - self.ox) / self.res).astype(np.int32)
            row = (self.H - (py - self.oy) / self.res).astype(np.int32)
            oob = (col < 0) | (col >= self.W) | (row < 0) | (row >= self.H)
            occ = np.zeros(n, bool)
            inb = ~oob & ~hit
            occ[inb] = self.occ[row[inb], col[inb]]
            newhit = (occ | oob) & ~hit
            out[newhit] = r[newhit]; hit |= newhit
            if hit.all(): break
            r = np.where(hit, r, r + step)
        return out


def bicycle_step(state, speed_cmd, steer, wheelbase, dt, accel=6.0):
    """state (x,y,theta,v) -> next, first-order speed tracking + kinematic bicycle."""
    x, y, th, v = state
    v = v + np.clip(speed_cmd - v, -accel * dt, accel * dt)
    x += v * math.cos(th) * dt
    y += v * math.sin(th) * dt
    th += v / wheelbase * math.tan(steer) * dt
    th = math.atan2(math.sin(th), math.cos(th))
    return np.array([x, y, th, v])


def render_fpv(simmap, x, y, theta, w=640, h=480, fx=460.5, cx=None, max_range=10.0):
    """Raycast wall render: distance-shaded verticals, sky/floor bands. Returns HxWx3 rgb8."""
    cx = cx if cx is not None else w / 2.0
    cols = np.arange(w)
    bearings = np.arctan2(cols - cx, fx)                     # per-column ray angle
    d = simmap.raycast(x, y, theta + bearings, max_range=max_range, step=simmap.res * 2)
    d = np.maximum(d, 0.05)
    img = np.empty((h, w, 3), np.uint8)
    img[:h // 2] = (60, 60, 70)                              # ceiling/sky
    img[h // 2:] = (35, 35, 38)                              # floor
    wall_h = np.clip((1.2 / d) * h, 4, h).astype(int)        # nearer = taller
    shade = np.clip(230 - d / max_range * 200, 25, 230).astype(np.uint8)
    for i in range(w):
        top = (h - wall_h[i]) // 2; bot = top + wall_h[i]
        c = shade[i]
        img[top:bot, i] = (c, c, int(c * 0.85))
    return img


def load_map(yaml_path):
    import os, yaml
    from PIL import Image
    meta = yaml.safe_load(open(yaml_path))
    ip = meta['image']
    if not os.path.isabs(ip):
        ip = os.path.join(os.path.dirname(yaml_path), ip)
    g = np.asarray(Image.open(ip).convert('L'))
    res = float(meta['resolution']); origin = meta['origin']
    negate = int(meta.get('negate', 0)); occ_th = float(meta.get('occupied_thresh', 0.65))
    p = (255 - g) / 255.0 if not negate else g / 255.0
    return (p > occ_th), res, origin
