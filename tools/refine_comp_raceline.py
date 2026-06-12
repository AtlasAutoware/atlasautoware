#!/usr/bin/env python3
"""
Refine, re-profile, validate, and install the competition raceline.
===================================================================

The shipped racelines/comp_raceline.csv was fit from driven laps and never
refined.  Checking it against the occupancy grid (maps/comp_track.yaml/.png)
shows it is not merely unrefined but UNSAFE: its minimum distance to the
nearest occupied cell is 0.000 m — at five corner regions (around indices
93-96, 116-124, 157-160, 187-191, 271-274/285-288 of the original 300-point
line) it cuts up to 0.52 m INSIDE mapped walls.  A wall-blind minimum-
curvature refinement (raceline_refiner with a scalar corridor, as in
tools/benchmark_refiner.py) makes this worse: it cuts the same apexes deeper
(5th-percentile clearance drops from 0.149 m to 0.001 m at corridor 0.35 m).

This tool therefore runs a WALL-AWARE pipeline, end to end:

  1. Build distance fields from the occupancy grid
     (scipy.ndimage.distance_transform_edt, scaled by map resolution):
     the unsigned distance-to-nearest-occupied-cell field used for all
     clearance statistics, plus a signed variant used to escape from
     inside walls.
  2. SAFETY REPAIR — waypoints (resampled to 0.40 m spacing) whose
     adjacent-segment clearance is below FLOOR + SLACK are moved along
     their normals, by the minimum distance needed, toward the in-corridor
     clearance maximum.  Only the 5 wall-cutting corner regions move; this
     step is exempt from the refinement corridor because staying inside a
     mapped wall is strictly less safe than any move out of it.
  3. MIN-CURVATURE REFINEMENT — raceline_refiner.refine_raceline with a
     SPATIALLY-VARYING corridor (the refiner accepts per-point budgets):
     budget_i = clip(min(clearance_i - FLOOR, CAP - spent_i), 0, CAP), so
     no point may approach a wall closer than FLOOR and the total
     displacement from the repaired baseline never exceeds the corridor
     CAP.  (Clearance is 1-Lipschitz, so clearance_new >= FLOOR is
     guaranteed at the waypoints; segment-level clearance is enforced via
     the adjacent-segment minimum and re-checked densely afterwards.)
  4. RE-PROFILE — speeds from the same profiler and the SAME friction /
     accel parameters the shipped line was profiled with.  Provenance:
     regenerating the shipped speed column with
     raceline_optimizer.velocity_profile + VehicleParams defaults
     (a_lat 6.5, a_accel 5.0, a_brake 7.0 m/s^2, v 1.5..7.0 m/s)
     reproduces it to < 1e-3 m/s, so those are the original limits and
     they are reused here unchanged (no friction limit is raised).
  5. VALIDATION GATES (per corridor CAP, largest first 0.35 .. 0.10):
       - clearance: min dense clearance >= max(min(old_min, 0.45), FLOOR)
         (old_min is 0.000, so FLOOR = 0.20 m is the binding bound);
       - closed-loop MPC lap (tests/closed_loop.run_lap, KinematicMPC,
         wheelbase 0.33, horizon 12, dt 0.08, v_max = max(speed)+0.5)
         must complete and be FASTER than the old line, xte_max < 0.6;
       - MAP-controller lap must satisfy the test-suite bounds
         (xte_mean < 0.15, xte_max < 0.6);
       - delay-compensated MPC lap at 0.10 s actuator delay must satisfy
         the test-suite bounds (xte_mean < 0.2, xte_max < 0.6);
       - the MAP delay-compensation property of
         tests/test_racing_tech.py::test_delay_compensation_recovers_tracking
         (comp xte_max < 0.75 * plain xte_max, comp no slower) must hold.
     The first CAP that passes everything is installed as
     racelines/comp_raceline.csv; if none passes, the original file is
     restored byte-for-byte and the tool exits nonzero.

The original line is preserved as racelines/comp_raceline_unrefined.csv
(created on first run).  The final line is emitted at 0.75 m spacing,
comparable to the original 0.80 m, with heading/curvature recomputed by
raceline_refiner.heading_curvature (the CSV stencil).

    python3 tools/refine_comp_raceline.py
"""

import csv
import math
import os
import shutil
import sys

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.join(REPO, 'tests'))

from closed_loop import load_raceline, run_lap                      # noqa: E402
from raceline_refiner import (refine_raceline, heading_curvature,   # noqa: E402
                              segment_lengths, left_normals)
from raceline_optimizer import (VehicleParams,                      # noqa: E402
                                velocity_profile as optimizer_velocity_profile)
from mpc_controller import KinematicMPC, predict_state              # noqa: E402
from map_controller import MAPController, build_lat_accel_lut       # noqa: E402

MAP_YAML = os.path.join(REPO, 'maps', 'comp_track.yaml')
RACELINE = os.path.join(REPO, 'racelines', 'comp_raceline.csv')
UNREFINED = os.path.join(REPO, 'racelines', 'comp_raceline_unrefined.csv')

CAPS = (0.35, 0.30, 0.25, 0.20, 0.15, 0.10)  # corridor caps, largest first
FLOOR = 0.20           # m, required minimum wall clearance of the new line
SLACK = 0.20           # m, extra clearance the repair step aims for, so the
                       #    refiner has room to smooth the rerouted sections
WORK_SPACING = 0.40    # m, waypoint spacing during repair + refinement
OUT_SPACING = 0.75     # m, waypoint spacing of the emitted CSV (orig: 0.80)
OUTER_PASSES = 5       # budget-recompute -> refine passes
REFINE_ITERS = 3       # linearize/solve iterations inside each refine call
WHEELBASE = 0.33


# ─────────────────────────────────────────────────────────────────────────────
# Occupancy grid -> distance fields
# ─────────────────────────────────────────────────────────────────────────────

class TrackMap:
    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        img_path = meta['image']
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(yaml_path), img_path)
        self.res = float(meta['resolution'])
        self.origin = meta['origin']
        img = np.array(Image.open(img_path).convert('L'))
        self.H, self.W = img.shape
        p = img / 255.0 if int(meta.get('negate', 0)) else (255 - img) / 255.0
        occ = p > float(meta.get('occupied_thresh', 0.65))
        # distance to the nearest occupied cell, meters (the task's gate field)
        self.edt = ndimage.distance_transform_edt(~occ) * self.res
        # signed variant: negative inside occupied cells (depth of penetration)
        self.sd = self.edt - ndimage.distance_transform_edt(occ) * self.res

    def sample(self, field, x, y):
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        col = (x - self.origin[0]) / self.res
        row = (self.H - 1) - (y - self.origin[1]) / self.res
        return ndimage.map_coordinates(field, [row, col], order=1)

    def clearance(self, x, y):                       # unsigned, at points
        return self.sample(self.edt, x, y)

    def dense_clearance(self, x, y, step=0.02):
        """Unsigned clearance sampled every `step` m along the closed polyline."""
        return self.sample(self.edt, *dense_points(x, y, step))


def dense_points(x, y, step=0.02):
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    xs, ys = [], []
    for i in range(len(x)):
        seg = np.hypot(xn[i] - x[i], yn[i] - y[i])
        k = max(int(np.ceil(seg / step)), 1)
        t = np.arange(k) / k
        xs.append(x[i] + t * (xn[i] - x[i]))
        ys.append(y[i] + t * (yn[i] - y[i]))
    return np.concatenate(xs), np.concatenate(ys)


def resample_arc(x, y, spacing):
    """Uniform arc-length resampling of the closed polyline (linear interp)."""
    ds = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    m = int(round(s[-1] / spacing))
    su = np.arange(m) * s[-1] / m
    return (np.interp(su, s, np.append(x, x[0])),
            np.interp(su, s, np.append(y, y[0])))


# ─────────────────────────────────────────────────────────────────────────────
# Wall-aware repair + spatially-varying-corridor refinement
# ─────────────────────────────────────────────────────────────────────────────

def seg_min_clearance(tm, x, y, step=0.05):
    """Per waypoint: min signed clearance over its two adjacent segments."""
    n = len(x)
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    segmin = np.zeros(n)
    for i in range(n):
        seg = np.hypot(xn[i] - x[i], yn[i] - y[i])
        k = max(int(np.ceil(seg / step)), 1)
        t = np.arange(k + 1) / k
        segmin[i] = tm.sample(tm.sd, x[i] + t * (xn[i] - x[i]),
                              y[i] + t * (yn[i] - y[i])).min()
    return np.minimum(segmin, np.roll(segmin, 1))


def best_along_normal(tm, x, y, reach=1.5, step=0.01):
    """Per waypoint: (max clearance, offset) within the free corridor segment
    nearest to the line along its left normal (never jumps across a wall)."""
    nx, ny = left_normals(x, y)
    offs = np.arange(-reach, reach + 1e-9, step)
    C = np.stack([tm.sample(tm.sd, x + o * nx, y + o * ny) for o in offs])
    i0 = int(np.argmin(np.abs(offs)))
    best = np.full(len(x), -np.inf)
    boff = np.zeros(len(x))
    for j in range(len(x)):
        c = C[:, j]
        free = np.where(c > 0)[0]
        if len(free) == 0:
            continue
        k = i0 if c[i0] > 0 else free[np.argmin(np.abs(free - i0))]
        a = b = k
        while a > 0 and c[a - 1] > 0:
            a -= 1
        while b < len(offs) - 1 and c[b + 1] > 0:
            b += 1
        m = a + int(np.argmax(c[a:b + 1]))
        best[j] = c[m]
        boff[j] = offs[m]
    return best, boff, nx, ny


def repair(tm, x, y, target, rounds=30):
    """Move waypoints whose adjacent-segment clearance is below `target`
    along their normals toward open space, by the minimum needed."""
    x, y = x.copy(), y.copy()
    for _ in range(rounds):
        per_pt = seg_min_clearance(tm, x, y)
        ach, aoff, nx, ny = best_along_normal(tm, x, y)
        tgt = np.minimum(target, ach - 0.02)
        bad = np.where(per_pt < tgt - 1e-3)[0]
        if len(bad) == 0:
            break
        for j in bad:
            deficit = tgt[j] - per_pt[j]
            sgn = 1.0 if aoff[j] >= 0 else -1.0
            step = min(0.6 * deficit + 0.01, abs(aoff[j]))
            x[j] += sgn * step * nx[j]
            y[j] += sgn * step * ny[j]
    return x, y


def build_candidate(tm, rx0, ry0, cap):
    """Repair -> per-point-corridor min-curvature refinement -> downsample."""
    x, y = resample_arc(rx0, ry0, WORK_SPACING)
    x, y = repair(tm, x, y, FLOOR + SLACK)
    bx, by = x.copy(), y.copy()              # safe baseline: cap measured here
    for _ in range(OUTER_PASSES):
        per_pt = seg_min_clearance(tm, x, y)
        spent = np.hypot(x - bx, y - by)
        budget = np.clip(np.minimum(per_pt - FLOOR, cap - spent), 0.0, cap)
        x, y, _, _ = refine_raceline(x, y, corridor=budget,
                                     iterations=REFINE_ITERS)
    x, y = repair(tm, x, y, FLOOR + 0.01, rounds=12)   # re-lift residual dips
    x, y = resample_arc(x, y, OUT_SPACING)
    x, y = repair(tm, x, y, FLOOR + 0.01, rounds=12)   # fix downsample chords
    hdg, curv = heading_curvature(x, y)
    return x, y, hdg, curv


def profile_speeds(x, y, curv):
    """Same profiler + parameters the shipped line was profiled with."""
    return optimizer_velocity_profile(np.column_stack([x, y]), curv,
                                      VehicleParams())


# ─────────────────────────────────────────────────────────────────────────────
# Closed-loop validation harnesses (mirror the test suite)
# ─────────────────────────────────────────────────────────────────────────────

def mpc_lap(rx, ry, rh, rc, rv):
    mpc = KinematicMPC(wheelbase=WHEELBASE, horizon=12, dt=0.08,
                       v_max=float(rv.max()) + 0.5)
    if not mpc.available:
        print('FAIL: osqp not available')
        sys.exit(1)
    mpc.set_raceline(rx, ry, rh, rc, rv)

    def control(px, py, yaw, v, j):
        out = mpc.solve((px, py, yaw, v), j)
        return out if out is not None else (0.0, max(v, 1.0))
    return run_lap(control, rx, ry, rh)


def delay_comp_mpc_lap(rx, ry, rh, rc, rv, delay=0.10, ctrl_dt=0.02):
    """tests/test_mpc.py::test_delay_compensated_lap_tracks_tightly flow."""
    n = len(rx)
    mpc = KinematicMPC(wheelbase=WHEELBASE, horizon=12, dt=0.08,
                       v_max=float(rv.max()) + 0.5)
    mpc.set_raceline(rx, ry, rh, rc, rv)
    buf = [(0.0, 2.0)] * int(round(delay / ctrl_dt))
    near = [0]

    def nearest(px, py, prev):
        best, bi = 1e18, prev
        for o in range(-5, 80):
            j = (prev + o) % n
            d = (rx[j] - px) ** 2 + (ry[j] - py) ** 2
            if d < best:
                best, bi = d, j
        return bi

    def control(px, py, yaw, v, j):
        px, py, yaw, v = predict_state(px, py, yaw, v,
                                       [c[0] for c in buf],
                                       [c[1] for c in buf], delay, WHEELBASE)
        near[0] = nearest(px, py, near[0])
        out = mpc.solve((px, py, yaw, v), near[0])
        steer, v_t = out if out is not None else (0.0, max(v, 1.0))
        buf.append((steer, v_t))
        buf.pop(0)
        return steer, v_t
    return run_lap(control, rx, ry, rh, wheelbase=WHEELBASE, dt=ctrl_dt,
                   actuator_delay=delay)


def map_lap(lut, rx, ry, rh, rv):
    ctl = MAPController(lut=lut)
    ctl.set_raceline(rx, ry, rv)
    return run_lap(ctl.control, rx, ry, rh)


def map_delay_pair(lut, rx, ry, rh, rv, delay=0.10):
    """test_racing_tech.py::test_delay_compensation_recovers_tracking flow."""
    ctl = MAPController(lut=lut)
    ctl.set_raceline(rx, ry, rv)
    plain = run_lap(ctl.control, rx, ry, rh, actuator_delay=delay)
    last = {'steer': 0.0, 'v': 2.0}

    def compensated(px, py, yaw, v, j):
        px, py, yaw, v = predict_state(px, py, yaw, v, last['steer'],
                                       last['v'], delay, WHEELBASE)
        j = int(np.argmin((rx - px) ** 2 + (ry - py) ** 2))
        steer, v_t = ctl.control(px, py, yaw, v, j)
        last['steer'], last['v'] = float(steer), float(v_t)
        return steer, v_t
    comp = run_lap(compensated, rx, ry, rh, actuator_delay=delay)
    return plain, comp


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def est_lap_time(x, y, v):
    ds = segment_lengths(x, y)
    v_seg = np.maximum(0.5 * (v + np.roll(v, -1)), 1e-6)
    return float(np.sum(ds / v_seg))


def max_dist_to_polyline(px, py, qx, qy):
    """Max over points p of the distance to the closed polyline q."""
    ax, ay = qx, qy
    bx, by = np.roll(qx, -1), np.roll(qy, -1)
    ex, ey = bx - ax, by - ay
    ee = ex * ex + ey * ey + 1e-12
    worst = 0.0
    for j in range(len(px)):
        t = np.clip(((px[j] - ax) * ex + (py[j] - ay) * ey) / ee, 0.0, 1.0)
        d2 = (ax + t * ex - px[j]) ** 2 + (ay + t * ey - py[j]) ** 2
        worst = max(worst, math.sqrt(float(d2.min())))
    return worst


def save_raceline(path, x, y, hdg, curv, spd):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y', 'heading', 'curvature', 'speed'])
        for i in range(len(x)):
            w.writerow([round(x[i], 4), round(y[i], 4), round(hdg[i], 4),
                        round(curv[i], 6), round(spd[i], 3)])


def round_like_csv(x, y, hdg, curv, spd):
    """Validate exactly what will be written to disk."""
    return (np.round(x, 4), np.round(y, 4), np.round(hdg, 4),
            np.round(curv, 6), np.round(spd, 3))


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(UNREFINED):
        shutil.copyfile(RACELINE, UNREFINED)
        print(f'[backup] {os.path.basename(RACELINE)} -> '
              f'{os.path.basename(UNREFINED)}')
    tm = TrackMap(MAP_YAML)
    lut = build_lat_accel_lut()

    ox, oy, oh, oc, ov = load_raceline(UNREFINED)
    old_dc = tm.dense_clearance(ox, oy)
    old_ds = segment_lengths(ox, oy)
    old_pen = -float(tm.sample(tm.sd, *dense_points(ox, oy)).min())
    old_mpc = mpc_lap(ox, oy, oh, oc, ov)
    gate = max(min(float(old_dc.min()), 0.45), FLOOR)

    print(f'[old] n={len(ox)} length {old_ds.sum():.2f} m | '
          f'|k| max {np.abs(oc).max():.3f} mean {np.abs(oc).mean():.4f}')
    print(f'[old] clearance min {old_dc.min():.3f} m  '
          f'p5 {np.percentile(old_dc, 5):.3f} m  '
          f'(max wall penetration {old_pen:.3f} m)')
    print(f'[old] est lap {est_lap_time(ox, oy, ov):.2f} s | closed-loop MPC '
          f'{old_mpc["lap_time"]:.2f} s xte {old_mpc["xte_mean"]:.3f}/'
          f'{old_mpc["xte_max"]:.3f}')
    print(f'[gate] required min clearance: max(min(old_min, 0.45), FLOOR) '
          f'= {gate:.2f} m; new lap must beat {old_mpc["lap_time"]:.2f} s')

    for cap in CAPS:
        print(f'\n=== corridor cap {cap:.2f} m '
              f'(floor {FLOOR:.2f} m, slack {SLACK:.2f} m) ===')
        x, y, hdg, curv = build_candidate(tm, ox, oy, cap)
        spd = profile_speeds(x, y, curv)
        x, y, hdg, curv, spd = round_like_csv(x, y, hdg, curv, spd)

        dc = tm.dense_clearance(x, y)
        c_min, c_p5 = float(dc.min()), float(np.percentile(dc, 5))
        print(f'[clearance] min {c_min:.3f} m  p5 {c_p5:.3f} m '
              f'(gate >= {gate:.2f})')
        if c_min < gate - 1e-3:
            print('[gate] FAIL clearance — stepping corridor down')
            continue

        new_mpc = mpc_lap(x, y, hdg, curv, spd)
        ok_fast = (new_mpc['completed']
                   and new_mpc['lap_time'] < old_mpc['lap_time']
                   and new_mpc['xte_max'] < 0.6)
        print(f'[mpc] lap {new_mpc["lap_time"]:.2f} s xte '
              f'{new_mpc["xte_mean"]:.3f}/{new_mpc["xte_max"]:.3f} '
              f'completed {new_mpc["completed"]} -> '
              f'{"OK" if ok_fast else "FAIL"}')
        if not ok_fast:
            continue

        rmap = map_lap(lut, x, y, hdg, spd)
        ok_map = (rmap['completed'] and rmap['xte_mean'] < 0.15
                  and rmap['xte_max'] < 0.6)
        print(f'[map] lap {rmap["lap_time"]:.2f} s xte '
              f'{rmap["xte_mean"]:.3f}/{rmap["xte_max"]:.3f} -> '
              f'{"OK" if ok_map else "FAIL"}')
        if not ok_map:
            continue

        rdc = delay_comp_mpc_lap(x, y, hdg, curv, spd)
        ok_dc = (rdc['completed'] and rdc['xte_mean'] < 0.2
                 and rdc['xte_max'] < 0.6)
        print(f'[mpc+delay 0.10s comp] lap {rdc["lap_time"]:.2f} s xte '
              f'{rdc["xte_mean"]:.3f}/{rdc["xte_max"]:.3f} -> '
              f'{"OK" if ok_dc else "FAIL"}')
        if not ok_dc:
            continue

        plain, comp = map_delay_pair(lut, x, y, hdg, spd)
        ok_pair = (plain['completed'] and comp['completed']
                   and comp['xte_max'] < plain['xte_max'] * 0.75
                   and comp['lap_time'] <= plain['lap_time'] + 0.1)
        print(f'[map delay pair] plain xte_max {plain["xte_max"]:.3f} '
              f'comp {comp["xte_max"]:.3f} '
              f'(ratio {comp["xte_max"] / plain["xte_max"]:.2f} < 0.75) '
              f'laps {plain["lap_time"]:.2f}/{comp["lap_time"]:.2f} -> '
              f'{"OK" if ok_pair else "FAIL"}')
        if not ok_pair:
            continue

        # all gates green — install
        save_raceline(RACELINE, x, y, curv=curv, hdg=hdg, spd=spd)
        ds = segment_lengths(x, y)
        print(f'\n[install] wrote {RACELINE} ({len(x)} pts)')
        print(f'[report] corridor cap {cap:.2f} m, clearance floor '
              f'{FLOOR:.2f} m')
        print(f'[report] clearance  min {old_dc.min():.3f} -> {c_min:.3f} m | '
              f'p5 {np.percentile(old_dc, 5):.3f} -> {c_p5:.3f} m')
        print(f'[report] length {old_ds.sum():.2f} -> {ds.sum():.2f} m | '
              f'|k| max {np.abs(oc).max():.3f} -> {np.abs(curv).max():.3f}, '
              f'mean {np.abs(oc).mean():.4f} -> {np.abs(curv).mean():.4f}')
        print(f'[report] max lateral displacement vs old line '
              f'{max_dist_to_polyline(x, y, ox, oy):.3f} m '
              f'(safety repair exceeds the corridor only where the old line '
              f'is inside a wall)')
        print(f'[report] est lap {est_lap_time(ox, oy, ov):.2f} -> '
              f'{est_lap_time(x, y, spd):.2f} s | closed-loop MPC '
              f'{old_mpc["lap_time"]:.2f} -> {new_mpc["lap_time"]:.2f} s '
              f'({old_mpc["lap_time"] - new_mpc["lap_time"]:+.2f} s)')
        print(f'[report] speeds {spd.min():.2f}-{spd.max():.2f} m/s '
              f'(profile: a_lat 6.5, a_accel 5.0, a_brake 7.0, '
              f'v 1.5-7.0 — unchanged from the original)')
        return 0

    shutil.copyfile(UNREFINED, RACELINE)
    print('\n[gate] NO corridor passed every gate — original raceline '
          'restored byte-for-byte')
    return 1


if __name__ == '__main__':
    sys.exit(main())
