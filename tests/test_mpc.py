"""
Closed-loop validation of the kinematic MPC — no ROS, no gym, just the model.
=============================================================================

Loads the real optimized raceline, then simulates the *true* kinematic bicycle
forward at a fine timestep while the MPC controls it at the control rate.  This
is a faithful closed-loop test of the controller (plant = the same model the MPC
plans with, plus an initial pose error to recover from): if the MPC tracks here,
the maths and signs are right.  It asserts the car completes a lap with bounded
cross-track error.

    python3 tests/test_mpc.py            # (inside the container: osqp + scipy)
"""

import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
from mpc_controller import KinematicMPC


def load_raceline(path):
    x, y, h, c, v = [], [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            x.append(float(r['x'])); y.append(float(r['y']))
            h.append(float(r['heading'])); c.append(float(r['curvature']))
            v.append(float(r['speed']))
    return (np.array(x), np.array(y), np.array(h), np.array(c), np.array(v))


def nearest_idx(px, py, rx, ry, prev, n, win=80):
    best, bi = 1e18, prev
    for o in range(-5, win):
        j = (prev + o) % n
        d = (rx[j] - px) ** 2 + (ry[j] - py) ** 2
        if d < best:
            best, bi = d, j
    return bi


def cross_track(px, py, rx, ry, j, n):
    tx = rx[(j + 1) % n] - rx[(j - 1) % n]
    ty = ry[(j + 1) % n] - ry[(j - 1) % n]
    tn = math.hypot(tx, ty) + 1e-9
    nx, ny = -ty / tn, tx / tn
    return abs((px - rx[j]) * nx + (py - ry[j]) * ny)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rl_path = os.path.join(repo, 'racelines', 'comp_raceline.csv')
    if not os.path.exists(rl_path):
        rl_path = os.path.join(repo, 'racelines', 'best_raceline.csv')
    rx, ry, rh, rc, rv = load_raceline(rl_path)
    n = len(rx)
    print(f'raceline: {n} pts, {os.path.basename(rl_path)}, '
          f'v {rv.min():.1f}-{rv.max():.1f} m/s')

    L = 0.33
    mpc = KinematicMPC(wheelbase=L, horizon=12, dt=0.08,
                       v_max=float(rv.max()) + 0.5)
    if not mpc.available:
        print('FAIL: osqp not available — cannot run MPC')
        sys.exit(1)
    mpc.set_raceline(rx, ry, rh, rc, rv)

    # true plant: start ON the line but with a deliberate pose error to recover.
    px, py = float(rx[0]) + 0.4, float(ry[0]) - 0.3   # ~0.5 m off
    yaw = float(rh[0]) + 0.25                          # ~14 deg heading error
    v = 2.0
    dt_ctrl, dt_sim = 0.08, 0.01
    sub = int(round(dt_ctrl / dt_sim))

    j = nearest_idx(px, py, rx, ry, 0, n)
    cum = 0                                  # forward index progress (lap = n)
    prev_j = j
    xte_hist, v_hist, solve_ms, fails = [], [], [], 0
    steps = 0
    max_steps = 4000                         # plenty for one lap

    while cum < n + 5 and steps < max_steps:
        steps += 1
        j = nearest_idx(px, py, rx, ry, prev_j, n)
        d = (j - prev_j)
        if d < -n / 2:
            d += n
        if 0 < d < n / 2:
            cum += d
        prev_j = j

        _t0 = time.perf_counter()
        out = mpc.solve((px, py, yaw, v), j)
        solve_ms.append((time.perf_counter() - _t0) * 1e3)
        if out is None:
            fails += 1
            steer, v_tgt = 0.0, max(v, 1.0)
        else:
            steer, v_tgt = out
        # speed command -> bounded accel toward target (matches the agent ramp)
        a = float(np.clip((v_tgt - v) / dt_ctrl, -8.0, 4.0))

        for _ in range(sub):                 # integrate the true plant
            px += v * math.cos(yaw) * dt_sim
            py += v * math.sin(yaw) * dt_sim
            yaw += v * math.tan(steer) / L * dt_sim
            v += a * dt_sim
            v = max(0.0, v)
        xte_hist.append(cross_track(px, py, rx, ry, j, n))
        v_hist.append(v)

    laps = cum / float(n)
    xte = np.array(xte_hist)
    imax = int(np.argmax(xte))
    settle = 60                              # steps allowed for initial recovery
    steady = xte[settle:]
    print(f'steps {steps} | progress {laps:.2f} lap | solver fails {fails}')
    print(f'cross-track err: mean {xte.mean():.3f} m, max {xte.max():.3f} m '
          f'@step {imax} (v={v_hist[imax]:.1f})')
    print(f'  transient (<{settle}) max {xte[:settle].max():.3f} m | '
          f'steady max {steady.max():.3f} m, mean {steady.mean():.3f} m')
    print(f'speed: mean {np.mean(v_hist):.2f}, max {np.max(v_hist):.2f} m/s')
    sm = np.array(solve_ms)
    print(f'solve time: mean {sm.mean():.1f} ms, p95 {np.percentile(sm,95):.1f} ms, '
          f'max {sm.max():.1f} ms  (control budget 20 ms @50Hz)')

    ok = True
    if laps < 1.0:
        print('FAIL: did not complete a lap'); ok = False
    # judge tracking on the steady-state (after the deliberate initial error)
    if steady.max() > 0.45:
        print(f'FAIL: steady cross-track too large ({steady.max():.2f} m)'); ok = False
    if steady.mean() > 0.18:
        print(f'FAIL: steady tracking loose ({steady.mean():.2f} m)'); ok = False
    if fails > steps * 0.05:
        print(f'FAIL: too many solver failures ({fails})'); ok = False

    print('RESULT:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
