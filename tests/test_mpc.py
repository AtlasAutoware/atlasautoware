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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpc_controller import KinematicMPC, predict_state
from closed_loop import run_lap


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


# ─────────────────────────────────────────────────────────────────────────────
# Regression tests (pytest): MPC under actuator delay, with the ROS node's
# delay-compensation flow (predict through the in-flight command pipeline,
# recompute the nearest index from the predicted pose, then solve).
# ─────────────────────────────────────────────────────────────────────────────

def _comp_raceline():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rl_path = os.path.join(repo, 'racelines', 'comp_raceline.csv')
    if not os.path.exists(rl_path):
        rl_path = os.path.join(repo, 'racelines', 'best_raceline.csv')
    return load_raceline(rl_path)


def test_predict_state_history_matches_scalar_when_constant():
    # a pipeline of identical commands must be bit-identical to the scalar form
    s = predict_state(1.0, 2.0, 0.3, 3.0, 0.12, 4.0, 0.1, 0.33)
    h = predict_state(1.0, 2.0, 0.3, 3.0, [0.12] * 5, [4.0] * 5, 0.1, 0.33)
    assert s == h


def test_predict_state_history_applies_oldest_first():
    # [a then b] over the window == integrating a for delay/2 then b for delay/2
    a, b = (0.2, 3.0), (-0.1, 5.0)
    two = predict_state(0.0, 0.0, 0.0, 4.0, [a[0], b[0]], [a[1], b[1]],
                        0.10, 0.33)
    mid = predict_state(0.0, 0.0, 0.0, 4.0, a[0], a[1], 0.05, 0.33)
    chained = predict_state(*mid, b[0], b[1], 0.05, 0.33)
    assert all(abs(p - q) < 1e-12 for p, q in zip(two, chained))
    # and it must NOT equal holding only the newest command (the old bug)
    newest_only = predict_state(0.0, 0.0, 0.0, 4.0, b[0], b[1], 0.10, 0.33)
    assert abs(two[2] - newest_only[2]) > 1e-3


def test_delay_compensated_lap_tracks_tightly():
    """Closed loop with 0.10 s actuator delay + the node's compensation flow.

    Mirrors raceline_mpc._loop exactly: predict_state through the in-flight
    command buffer (oldest first), recompute nearest from the predicted pose,
    solve from there.  Must complete the lap with near-zero-delay tracking.
    """
    rx, ry, rh, rc, rv = _comp_raceline()
    n = len(rx)
    L, delay, ctrl_dt = 0.33, 0.10, 0.02
    mpc = KinematicMPC(wheelbase=L, horizon=12, dt=0.08,
                       v_max=float(rv.max()) + 0.5)
    assert mpc.available, 'osqp not available'
    mpc.set_raceline(rx, ry, rh, rc, rv)

    ticks = int(round(delay / ctrl_dt))
    buf = [(0.0, 2.0)] * ticks        # in flight; matches run_lap's prefill
    near = [0]

    def control(px, py, yaw, v, j):
        px, py, yaw, v = predict_state(
            px, py, yaw, v, [c[0] for c in buf], [c[1] for c in buf],
            delay, L)
        near[0] = nearest_idx(px, py, rx, ry, near[0], n)
        out = mpc.solve((px, py, yaw, v), near[0])
        steer, v_t = out if out is not None else (0.0, max(v, 1.0))
        buf.append((steer, v_t))
        buf.pop(0)
        return steer, v_t

    res = run_lap(control, rx, ry, rh, wheelbase=L, dt=ctrl_dt,
                  actuator_delay=delay)
    assert res['completed'], 'did not complete a lap under 0.10 s delay'
    assert res['xte_mean'] < 0.2, f"loose tracking ({res['xte_mean']:.3f} m)"
    assert res['xte_max'] < 0.6, f"ran wide ({res['xte_max']:.3f} m)"


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
