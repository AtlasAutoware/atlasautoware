#!/usr/bin/env python3
"""
Benchmark: speed-axis interpolation in the MAP controller's steer LUT.
======================================================================

The original steer_from_lat_accel picked the nearest LUT speed column
(argmin |lut_speed - v|), so steering jumped (up to ~0.12 rad) every time
the speed crossed a column midpoint.  The fix linearly interpolates the
inverted steer across the two bracketing speed columns.

This script compares nearest-column (baseline, reproduced here as a
subclass) against the interpolated implementation on the shared
closed-loop harness (tests/closed_loop.py, kinematic bicycle @ 50 Hz)
over the competition raceline, at nominal speed and with the speed
profile scaled x1.15 / x1.30 — reporting lap time, cross-track error and
steering-rate smoothness (std / max of d(steer)/dt).

    python3 tools/benchmark_map.py
"""

import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.join(REPO, 'tests'))

from map_controller import MAPController, build_lat_accel_lut  # noqa: E402
from closed_loop import load_raceline, run_lap                 # noqa: E402

SPEED_FACTORS = [1.0, 1.15, 1.30]
DT = 0.02                                       # harness step (50 Hz)


class NearestColumnMAP(MAPController):
    """Baseline: the ORIGINAL nearest-speed-column steer lookup."""

    def steer_from_lat_accel(self, a_lat, v):
        a = abs(float(a_lat))
        if v < 1.0:
            steer = math.atan(self.L_wb * a / max(v, 0.5) ** 2)
            return math.copysign(min(steer, self.max_steer), a_lat)
        j = int(np.argmin(np.abs(self.lut_speed - v)))
        col = self.lut_alat[:, j]
        idx = np.flatnonzero(np.isfinite(col))
        if len(idx) < 2:
            steer = math.atan(self.L_wb * a / v ** 2)
            return math.copysign(min(steer, self.max_steer), a_lat)
        rising = np.maximum.accumulate(col[idx])
        last = int(np.argmax(rising)) + 1
        xs, ys = col[idx[:last]], self.lut_steer[idx[:last]]
        steer = ys[-1] if a >= xs[-1] else float(np.interp(a, xs, ys))
        return math.copysign(min(steer, self.max_steer), a_lat)


def run_case(cls, lut, rx, ry, rh, rc, rv, factor):
    ctl = cls(lut=lut)
    ctl.set_raceline(rx, ry, rv * factor, curvature=rc)
    steers = []

    def control(px, py, yaw, v, j):
        steer, v_t = ctl.control(px, py, yaw, v, j)
        steers.append(float(steer))
        return steer, v_t

    res = run_lap(control, rx, ry, rh)
    rate = np.diff(np.array(steers)) / DT
    rate = rate[100:]                           # post-settle, like xte_mean
    res['steer_rate_std'] = float(rate.std())
    res['steer_rate_max'] = float(np.abs(rate).max())
    return res


def main():
    rx, ry, rh, rc, rv = load_raceline(
        os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    lut = build_lat_accel_lut()

    print(f"{'v_factor':>8} {'variant':>8} {'completed':>9} {'lap_time':>9} "
          f"{'xte_mean':>9} {'xte_max':>8} {'sr_std':>8} {'sr_max':>8}")
    print('-' * 74)
    for f in SPEED_FACTORS:
        for name, cls in (('nearest', NearestColumnMAP),
                          ('interp', MAPController)):
            r = run_case(cls, lut, rx, ry, rh, rc, rv, f)
            print(f"{f:>8.2f} {name:>8} {str(r['completed']):>9} "
                  f"{r['lap_time']:>9.2f} {r['xte_mean']:>9.4f} "
                  f"{r['xte_max']:>8.4f} {r['steer_rate_std']:>8.4f} "
                  f"{r['steer_rate_max']:>8.4f}")
        print('-' * 74)


if __name__ == '__main__':
    main()
