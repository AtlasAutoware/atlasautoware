"""
Actuator-delay sweep: naive MPC vs delay-compensated MPC (ROS-free).
====================================================================

Runs tests/closed_loop.run_lap on the competition raceline across a sweep of
actuator delays and compares:

  (a) naive       — mpc.solve((px,py,yaw,v), j) straight from the measured
                    state, with the harness's own nearest index j (no
                    compensation); this is what the baseline numbers measure.
  (b) compensated — exactly the ROS node's flow (raceline_mpc._loop):
                    predict_state(measured state, in-flight command pipeline
                    [steer, v_target] oldest first, delay) -> recompute the
                    nearest raceline index from the *predicted* pose ->
                    mpc.solve(predicted, nearest).  The compensator is given
                    the true delay (same value passed to run_lap), i.e.
                    perfect delay knowledge.

Historical note: the node originally fed predict_state only the *last*
commanded (steer, v_target), held over the whole delay window.  That is the
wrong command for a pipeline latency (the newest command hasn't reached the
actuator yet) and measurably destabilizes the loop for delays >= 0.08 s —
xte_mean 0.51 / lap cut short at 0.12 s.  Reproduce by replacing the buffer
below with [st['buf'][-1]].

Prints a table; with --json [PATH] also emits the results as JSON (to PATH, or
to stdout when PATH is omitted).

    python3 tools/benchmark_delay.py
    python3 tools/benchmark_delay.py --json results.json
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.join(REPO, 'tests'))

from closed_loop import load_raceline, run_lap          # noqa: E402
from mpc_controller import KinematicMPC, predict_state  # noqa: E402

DELAYS = (0.0, 0.04, 0.08, 0.10, 0.12, 0.14)
WHEELBASE = 0.33


def find_nearest(x, y, rl_x, rl_y, prev, search=60):
    """Verbatim copy of pursuit_agent.find_nearest (that module needs rclpy)."""
    n = len(rl_x)
    bd, bi = float('inf'), prev
    for off in range(-5, search):
        idx = (prev + off) % n
        d = np.hypot(rl_x[idx] - x, rl_y[idx] - y)
        if d < bd:
            bd, bi = d, idx
    return bi


def make_mpc(rx, ry, rh, rc, rv):
    mpc = KinematicMPC(wheelbase=WHEELBASE, horizon=12, dt=0.08,
                       v_max=float(rv.max()) + 0.5)
    if not mpc.available:
        print('FAIL: osqp not available'); sys.exit(1)
    mpc.set_raceline(rx, ry, rh, rc, rv)
    return mpc


def naive_fn(rx, ry, rh, rc, rv):
    mpc = make_mpc(rx, ry, rh, rc, rv)

    def control(px, py, yaw, v, j):
        out = mpc.solve((px, py, yaw, v), j)
        return out if out is not None else (0.0, max(v, 1.0))
    return control


def compensated_fn(rx, ry, rh, rc, rv, delay, v0=2.0, ctrl_dt=0.02):
    """Mirror raceline_mpc._loop: predict through pipeline -> renearest -> solve.

    `st['buf']` holds the commands still in flight (oldest first, one per
    control tick over the delay window), exactly like the node's _cmd_buf;
    it is prefilled with (0.0, v0) to match run_lap's own pipeline prefill.
    """
    mpc = make_mpc(rx, ry, rh, rc, rv)
    ticks = int(round(delay / ctrl_dt))
    st = {'buf': [(0.0, v0)] * ticks, 'nearest': 0}

    def control(px, py, yaw, v, j):
        if delay > 0.0 and st['buf']:
            px, py, yaw, v = predict_state(
                px, py, yaw, v,
                [c[0] for c in st['buf']], [c[1] for c in st['buf']],
                delay, WHEELBASE)
        st['nearest'] = find_nearest(px, py, rx, ry, st['nearest'])
        out = mpc.solve((px, py, yaw, v), st['nearest'])
        steer, v_t = out if out is not None else (0.0, max(v, 1.0))
        if ticks:
            st['buf'].append((steer, v_t))
            st['buf'].pop(0)
        return steer, v_t
    return control


def lap_metrics(control, rx, ry, rh, delay):
    r = run_lap(control, rx, ry, rh, wheelbase=WHEELBASE, actuator_delay=delay)
    return dict(completed=bool(r['completed']),
                lap_time=round(float(r['lap_time']), 2),
                xte_mean=round(float(r['xte_mean']), 3),
                xte_max=round(float(r['xte_max']), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--json', nargs='?', const='-', default=None,
                    metavar='PATH', help='emit JSON results (to PATH, or '
                    'stdout when PATH is omitted)')
    ap.add_argument('--raceline',
                    default=os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    args = ap.parse_args()

    rx, ry, rh, rc, rv = load_raceline(args.raceline)
    print(f'raceline: {len(rx)} pts, {os.path.basename(args.raceline)}, '
          f'v {rv.min():.1f}-{rv.max():.1f} m/s')
    print(f'{"delay":>6} | {"naive lap":>9} {"xte_mean":>8} {"xte_max":>7} | '
          f'{"comp lap":>9} {"xte_mean":>8} {"xte_max":>7}')
    print('-' * 66)

    results = []
    ok = True
    for delay in DELAYS:
        nv = lap_metrics(naive_fn(rx, ry, rh, rc, rv), rx, ry, rh, delay)
        cp = lap_metrics(compensated_fn(rx, ry, rh, rc, rv, delay),
                         rx, ry, rh, delay)
        ok &= nv['completed'] and cp['completed']
        results.append(dict(delay=delay, naive=nv, compensated=cp))
        print(f'{delay:6.2f} | {nv["lap_time"]:8.2f}s {nv["xte_mean"]:8.3f} '
              f'{nv["xte_max"]:7.3f} | {cp["lap_time"]:8.2f}s '
              f'{cp["xte_mean"]:8.3f} {cp["xte_max"]:7.3f}')

    if args.json is not None:
        payload = json.dumps(dict(raceline=os.path.basename(args.raceline),
                                  wheelbase=WHEELBASE, results=results),
                             indent=2)
        if args.json == '-':
            print(payload)
        else:
            with open(args.json, 'w') as f:
                f.write(payload + '\n')
            print(f'json written to {args.json}')

    if not ok:
        print('FAIL: at least one configuration did not complete a lap')
        sys.exit(1)
    print('all laps completed')


if __name__ == '__main__':
    main()
