#!/usr/bin/env python3
"""
Benchmark: MPCC vs the tracking KinematicMPC under the SAME physics budget.
===========================================================================

Both controllers lap the shared closed-loop harness (tests/closed_loop.py,
kinematic bicycle @ 50 Hz, accel/brake plant limits 4/8 m/s^2) on the same
raceline with the same budget: |a_lat| <= 6.5 m/s^2, v <= 7 m/s, friction
ellipse per velocity_profiler.  The harness plant has NO grip limit, so the
budget is only real if a controller self-enforces it — which is exactly what
this table makes visible.  Rows per raceline:

  tracking        KinematicMPC, speeds re-profiled by velocity_profiler at
                  the 6.5/4/8 budget (the stack's shipping configuration).
                  Its driven a_lat shows how far closed-loop tracking
                  corrections push past the budget the profile assumed.
  tracking+gov    the same controller behind the same output governor the
                  MPCC uses (steer clamped to the lateral budget at the
                  measured speed, accel clamped to the friction ellipse) —
                  the budget-true version of the tracking stack.
  tracking+gov.80 governed tracking with the speed profile at 0.80 of the
                  lateral budget — the most aggressive profile fraction at
                  which the governed tracker still tracks its line on this
                  map (found by sweep; at the full-budget profile the
                  governor leaves no steering authority for feedback and the
                  car runs off line).  This is the baseline's best *valid*
                  lap under the budget.
  mpcc            the LTV-MPCC (mpcc_controller.MPCC): corridor-constrained,
                  curvature-integrated speed caps, progress-maximizing
                  (plans at 0.95 of the lateral budget; the predictive
                  curvature integration is what lets it run a higher
                  planning fraction than the tracker).

All metrics are measured on the DRIVEN trajectory: lap time, min wall
clearance (EDT of maps/comp_track.yaml), max & p99.5 |v*yaw_rate|, solver
failures, solve-time stats.  Racelines are loaded fresh at run time;
comp_raceline_unrefined.csv is benchmarked too when present.

    python3 tools/benchmark_mpcc.py
"""

import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.join(REPO, 'tests'))

from mpc_controller import KinematicMPC                            # noqa: E402
from mpcc_controller import (MPCC, TrackCorridor, build_reference,  # noqa: E402
                             ellipse_ax_avail)
from closed_loop import load_raceline, run_lap                     # noqa: E402
from velocity_profiler import velocity_profile, segment_lengths    # noqa: E402

MAP_YAML = os.path.join(REPO, 'maps', 'comp_track.yaml')
A_LAT = 6.5
A_ACC = 4.0
A_BRK = 8.0
V_MAX = 7.0
WHEELBASE = 0.33
DT = 0.02


def lap_metrics(control_fn, rx, ry, rh, corr):
    """Run one lap; measure everything on the driven trajectory."""
    log = dict(px=[], py=[], yaw=[], v=[])

    def wrapped(px, py, yaw, v, j):
        log['px'].append(px); log['py'].append(py)
        log['yaw'].append(yaw); log['v'].append(v)
        return control_fn(px, py, yaw, v, j)

    res = run_lap(wrapped, rx, ry, rh, dt=DT)
    v = np.array(log['v']); yaw = np.array(log['yaw'])
    clearance = corr.clearance(np.array(log['px']), np.array(log['py']))
    a_lat = v[:-1] * np.diff(yaw) / DT
    return dict(completed=res['completed'], lap=res['lap_time'],
                clr_min=float(clearance.min()),
                alat_max=float(np.abs(a_lat).max()),
                alat_p995=float(np.percentile(np.abs(a_lat), 99.5)),
                xte_mean=res['xte_mean'])


def governor(steer, v_t, v):
    """The MPCC's output governor, applied to any controller's command."""
    d_cap = min(0.41, math.atan(A_LAT * WHEELBASE / max(v * v, 1e-6)))
    steer = float(np.clip(steer, -d_cap, d_cap))
    a_lat = v * v * abs(math.tan(steer)) / WHEELBASE
    a = float(np.clip((v_t - v) / DT,
                      -ellipse_ax_avail(a_lat, A_LAT, A_BRK),
                      ellipse_ax_avail(a_lat, A_LAT, A_ACC)))
    return steer, v + a * DT


def run_tracking(rx, ry, rh, rc, corr, governed, profile_frac=1.0):
    vp = velocity_profile(rc, segment_lengths(rx, ry),
                          a_lat_max=profile_frac * A_LAT,
                          a_accel_max=A_ACC, a_brake_max=A_BRK, v_max=V_MAX)
    mpc = KinematicMPC(wheelbase=WHEELBASE, v_max=V_MAX)
    mpc.set_raceline(rx, ry, rh, rc, vp)
    solve_ms, fails = [], [0]

    def control(px, py, yaw, v, j):
        t0 = time.perf_counter()
        out = mpc.solve((px, py, yaw, v), j)
        solve_ms.append((time.perf_counter() - t0) * 1e3)
        if out is None:
            fails[0] += 1
            return 0.0, max(v - A_BRK * DT, 0.0)
        steer, v_t = out
        return governor(steer, v_t, v) if governed else (steer, v_t)

    m = lap_metrics(control, rx, ry, rh, corr)
    sm = np.array(solve_ms)
    m.update(fails=fails[0], ticks=len(sm), ms_mean=float(sm.mean()),
             ms_p95=float(np.percentile(sm, 95)), ms_max=float(sm.max()))
    return m


def run_mpcc(rx, ry, corr):
    gx, gy, gh, gk, blo, bhi = build_reference(corr, rx, ry)
    mpcc = MPCC(wheelbase=WHEELBASE, a_lat_max=A_LAT, a_accel=A_ACC,
                a_brake=A_BRK, v_max=V_MAX, ctrl_dt=DT)
    mpcc.set_raceline(gx, gy, gh, gk, band_lo=blo, band_hi=bhi)
    # the MPCC laps its corridor-legal guide line (same closed circuit)
    m = lap_metrics(mpcc.control, gx, gy, gh, corr)
    sm = np.array(mpcc.solve_ms)
    m.update(fails=mpcc.fail_count, ticks=len(sm), ms_mean=float(sm.mean()),
             ms_p95=float(np.percentile(sm, 95)), ms_max=float(sm.max()))
    return m


def main():
    corr = TrackCorridor(MAP_YAML)
    paths = [os.path.join(REPO, 'racelines', 'comp_raceline.csv')]
    unrefined = os.path.join(REPO, 'racelines', 'comp_raceline_unrefined.csv')
    if os.path.exists(unrefined):
        paths.append(unrefined)

    hdr = (f"{'controller':<16} {'done':>4} {'lap_s':>7} {'clr_min':>8} "
           f"{'alat_max':>8} {'p99.5':>6} {'xte_m':>6} {'fail':>5} "
           f"{'ms_mean':>7} {'ms_p95':>6} {'ms_max':>6}")
    for path in paths:
        rx, ry, rh, rc, rv = load_raceline(path)   # fresh load every run
        # common start: roll the loop so the harness spawn (index 0 + the
        # deliberate pose error) is in a wide part of the corridor
        roll = int(np.argmax(corr.clearance(rx, ry)))
        rx, ry, rh, rc, rv = (np.roll(a, -roll) for a in (rx, ry, rh, rc, rv))

        print(f'\n=== {os.path.basename(path)} '
              f'(budget: a_lat {A_LAT}, accel {A_ACC}, brake {A_BRK}, '
              f'v_max {V_MAX}) ===')
        print(hdr)
        print('-' * len(hdr))
        rows = [('tracking', run_tracking(rx, ry, rh, rc, corr, False)),
                ('tracking+gov', run_tracking(rx, ry, rh, rc, corr, True)),
                ('tracking+gov.80',
                 run_tracking(rx, ry, rh, rc, corr, True, profile_frac=0.80)),
                ('mpcc', run_mpcc(rx, ry, corr))]
        for name, m in rows:
            print(f"{name:<16} {str(m['completed'])[:1]:>4} {m['lap']:>7.2f} "
                  f"{m['clr_min']:>8.3f} {m['alat_max']:>8.2f} "
                  f"{m['alat_p995']:>6.2f} {m['xte_mean']:>6.3f} "
                  f"{m['fails']:>5d} {m['ms_mean']:>7.2f} "
                  f"{m['ms_p95']:>6.2f} {m['ms_max']:>6.2f}")
        print('  clr_min: min wall clearance of the DRIVEN trajectory (m); '
              'the plant has no walls,')
        print('  so clr_min < ~0.15 means the "lap time" was bought by '
              'driving through/along a wall.')
        print('  alat_max/p99.5: |v*yaw_rate| of the driven trajectory '
              f'(budget {A_LAT} m/s^2).')


if __name__ == '__main__':
    main()
