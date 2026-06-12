"""
MAP controller steer LUT — speed-axis interpolation.
====================================================

The steer lookup inverts a (steer x speed) lateral-acceleration LUT.  The
lateral-acceleration axis was always interpolated; the speed axis used to be
nearest-column (argmin |lut_speed - v|), which made steering jump by up to
~0.12 rad whenever the speed crossed a column midpoint.  These tests pin the
fixed behavior:

  - steering is continuous in v (no column-boundary jumps);
  - at the LUT's exact sample speeds the output is bit-identical to the
    original nearest-column inversion;
  - outside the table's speed range the lookup clamps to the edge columns;
  - invariants: odd symmetry in a_lat, zero maps to zero, grip-limit clamp,
    monotonicity in |a_lat|, and the low-speed kinematic branch.

    python3 -m pytest tests/test_map_controller.py -q
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
from map_controller import MAPController, build_lat_accel_lut    # noqa: E402

LUT = build_lat_accel_lut()


def _ctl():
    return MAPController(lut=LUT)


def _nearest_column_steer(ctl, a_lat, v):
    """Reference oracle: the ORIGINAL nearest-speed-column inversion."""
    a = abs(float(a_lat))
    if v < 1.0:
        steer = math.atan(ctl.L_wb * a / max(v, 0.5) ** 2)
        return math.copysign(min(steer, ctl.max_steer), a_lat)
    j = int(np.argmin(np.abs(ctl.lut_speed - v)))
    col = ctl.lut_alat[:, j]
    idx = np.flatnonzero(np.isfinite(col))
    if len(idx) < 2:
        steer = math.atan(ctl.L_wb * a / v ** 2)
        return math.copysign(min(steer, ctl.max_steer), a_lat)
    rising = np.maximum.accumulate(col[idx])
    last = int(np.argmax(rising)) + 1
    xs, ys = col[idx[:last]], ctl.lut_steer[idx[:last]]
    steer = ys[-1] if a >= xs[-1] else float(np.interp(a, xs, ys))
    return math.copysign(min(steer, ctl.max_steer), a_lat)


# ── continuity across the speed axis ─────────────────────────────────────────

def test_steer_continuous_in_speed():
    # fine sweep over the LUT speed range at constant lateral acceleration:
    # adjacent samples must never jump (the old nearest-column lookup jumped
    # by 0.05-0.12 rad at every column midpoint)
    ctl = _ctl()
    vs = np.arange(1.0, 8.4, 0.002)
    for a in (1.0, 2.0, 4.0, 6.0, 8.0):
        s = np.array([ctl.steer_from_lat_accel(a, float(v)) for v in vs])
        assert np.abs(np.diff(s)).max() < 0.005


def test_interpolation_stays_between_bracketing_columns():
    # a lerp can never overshoot its endpoints: steer at any v between two
    # sample speeds lies between the steers at those sample speeds
    ctl = _ctl()
    sp = ctl.lut_speed
    for a in (1.5, 4.0, 7.0):
        for j in range(4, len(sp) - 1, 6):
            s0 = ctl.steer_from_lat_accel(a, float(sp[j]))
            s1 = ctl.steer_from_lat_accel(a, float(sp[j + 1]))
            for w in (0.25, 0.5, 0.75):
                v = float((1 - w) * sp[j] + w * sp[j + 1])
                s = ctl.steer_from_lat_accel(a, v)
                assert min(s0, s1) - 1e-12 <= s <= max(s0, s1) + 1e-12


# ── exactness at sample speeds, clamping outside the table ──────────────────

def test_exact_at_lut_sample_speeds():
    # at the LUT's own speed columns the interpolated lookup must reproduce
    # the original nearest-column result bit-for-bit
    ctl = _ctl()
    for v in ctl.lut_speed:
        if v < 1.0:                  # below the kinematic handoff
            continue
        for a in np.linspace(-12.0, 12.0, 49):
            assert (ctl.steer_from_lat_accel(float(a), float(v))
                    == _nearest_column_steer(ctl, float(a), float(v)))


def test_clamps_above_table_speed_range():
    ctl = _ctl()
    v_top = float(ctl.lut_speed[-1])
    for a in (1.0, 4.0, 9.0):
        assert (ctl.steer_from_lat_accel(a, v_top + 4.0)
                == ctl.steer_from_lat_accel(a, v_top))


def test_clamps_below_table_speed_range():
    # custom LUT starting at 2 m/s so v in [1, 2) exercises the low clamp
    # (the default table starts below the 1 m/s kinematic handoff)
    lut = build_lat_accel_lut(v_min=2.0, v_max=6.0, n_speed=9)
    ctl = MAPController(lut=lut)
    for a in (1.0, 3.0):
        assert (ctl.steer_from_lat_accel(a, 1.2)
                == ctl.steer_from_lat_accel(a, 2.0))


# ── invariants ───────────────────────────────────────────────────────────────

def test_sign_symmetry_and_zero():
    ctl = _ctl()
    for v in (1.3, 2.7, 4.0, 6.55, 9.0):
        assert ctl.steer_from_lat_accel(0.0, v) == 0.0
        for a in (0.5, 2.0, 5.0, 20.0):
            assert (ctl.steer_from_lat_accel(-a, v)
                    == -ctl.steer_from_lat_accel(a, v))


def test_grip_limit_clamp_never_exceeds_max_steer():
    ctl = _ctl()
    for v in (1.1, 3.3, 5.0, 7.9, 11.0):
        s = ctl.steer_from_lat_accel(80.0, v)    # far beyond any grip limit
        assert 0.0 < s <= ctl.max_steer


def test_monotonic_in_lat_accel_at_fixed_speed():
    ctl = _ctl()
    for v in (1.5, 3.05, 5.5, 7.7):
        prev = -1.0
        for a in np.linspace(0.0, 15.0, 60):
            s = ctl.steer_from_lat_accel(float(a), v)
            assert s >= prev - 1e-12
            prev = s


def test_low_speed_kinematic_branch_unchanged():
    ctl = _ctl()
    a, v = 2.0, 0.8
    assert (ctl.steer_from_lat_accel(a, v)
            == min(math.atan(ctl.L_wb * a / v ** 2), ctl.max_steer))


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
