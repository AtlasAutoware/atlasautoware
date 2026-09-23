"""
Multi-car avoidance + C1 lidar regrid — pure-logic tests, no ROS, no hardware.
=============================================================================

    python3 -m pytest tests/test_avoidance.py -q
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
from rplidar_node import regrid_scan, bin_scan                       # noqa: E402
from opponent_avoidance import (AvoidanceLayer, follow_speed_cap,    # noqa: E402
                                room_lr)


# ─────────────────────────────────────────────────────────────────────────────
# regrid_scan: SDK driver scan -> fixed 720-bin grid (same contract as bin_scan)
# ─────────────────────────────────────────────────────────────────────────────

def _sdk_scan(n=504, r=None):
    """Irregular-ish C1-style scan: n beams over [-pi, pi)."""
    inc = 2 * math.pi / n
    ranges = np.full(n, 5.0) if r is None else r
    return ranges, -math.pi, inc


def test_regrid_grid_contract():
    ranges, amin, inc = _sdk_scan()
    out, ages = regrid_scan(ranges, amin, inc, 720)
    assert out.shape == (720,) and out.dtype == np.float32
    assert ages.shape == (720,)
    # 504 beams into 720 bins -> some bins empty (inf), the rest carry 5.0
    filled = np.isfinite(out)
    assert 400 <= filled.sum() <= 504
    assert np.allclose(out[filled], 5.0)


def test_regrid_places_beam_at_correct_angle():
    n = 360
    ranges = np.full(n, 10.0)
    ranges[n // 2 + 45] = 1.0           # beam at +45 deg (index n/2 = 0 rad)
    out, _ = regrid_scan(ranges, -math.pi, 2 * math.pi / n, 720)
    k = int(np.argmin(out))
    theta = -math.pi + k * (2 * math.pi / 720)
    assert abs(theta - math.radians(45)) < math.radians(1.0)


def test_regrid_angle_offset_rotates_ccw():
    n = 360
    ranges = np.full(n, 10.0)
    ranges[n // 2] = 1.0                # straight ahead
    out, _ = regrid_scan(ranges, -math.pi, 2 * math.pi / n, 720,
                         angle_offset=math.radians(90))
    k = int(np.argmin(out))
    theta = -math.pi + k * (2 * math.pi / 720)
    assert abs(theta - math.radians(90)) < math.radians(1.0)


def test_regrid_nearest_wins_and_filters_invalid():
    n = 1440                             # 2 beams per output bin
    ranges = np.full(n, 3.0)
    ranges[0] = 2.0                      # two beams in bin 0: 2.0 and 3.0
    ranges[10] = float('inf'); ranges[11] = float('nan'); ranges[12] = 0.02
    out, _ = regrid_scan(ranges, -math.pi, 2 * math.pi / n, 720)
    assert out[0] == pytest.approx(2.0)
    assert out[5] == pytest.approx(3.0) or out[6] == pytest.approx(3.0)


def test_regrid_ages_are_nonpositive_and_ordered():
    n = 504
    ranges = np.full(n, 4.0)
    out, ages = regrid_scan(ranges, -math.pi, 2 * math.pi / n, 720,
                            time_increment=0.1 / n)
    filled = np.isfinite(out)
    assert (ages[filled] <= 1e-9).all()
    assert ages[filled].min() < -0.09    # first beam is ~one sweep old


def test_regrid_matches_bin_scan_geometry():
    """Same physical world through both paths -> same grid (within a bin)."""
    # world: a wall at 2 m straight ahead, 6 m everywhere else
    n = 720
    ros_ranges = np.full(n, 6.0); ros_ranges[n // 2] = 2.0
    grid_a, _ = regrid_scan(ros_ranges, -math.pi, 2 * math.pi / n, 720)
    # pip path: (quality, angle_deg CW from connector, dist_mm); 0 deg = ahead
    meas = [(15, float(a), 6000.0) for a in range(0, 360)]
    meas[0] = (15, 0.0, 2000.0)
    grid_b = bin_scan(meas, 720)
    assert int(np.argmin(grid_a)) == int(np.argmin(grid_b))


# ─────────────────────────────────────────────────────────────────────────────
# follow governor
# ─────────────────────────────────────────────────────────────────────────────

def test_follow_cap_inactive_when_far():
    assert follow_speed_cap(gap=10.0, opp_speed_along=2.0, ego_speed=3.0) == float('inf')


def test_follow_cap_converges_on_opponent_speed_at_gap():
    cap = follow_speed_cap(gap=1.0, opp_speed_along=2.0, ego_speed=3.0, follow_gap=1.0)
    assert cap == pytest.approx(2.0)


def test_follow_cap_backs_off_inside_gap_and_for_stopped_car():
    assert follow_speed_cap(gap=0.5, opp_speed_along=2.0, ego_speed=3.0) < 2.0
    # a stopped car very close -> cap goes to zero (approach, never hit)
    assert follow_speed_cap(gap=0.3, opp_speed_along=0.0, ego_speed=2.0) == 0.0


def test_follow_cap_is_monotone_in_gap():
    caps = [follow_speed_cap(g, 1.5, 3.0) for g in (0.6, 1.0, 1.5, 2.0)]
    assert caps == sorted(caps)


# ─────────────────────────────────────────────────────────────────────────────
# AvoidanceLayer end-to-end on synthetic scans (straight corridor)
# ─────────────────────────────────────────────────────────────────────────────

def _corridor_scan(half_width=1.2, car=None, n=720):
    """Ego at origin heading +x in a corridor; optional car (x, y, w) ahead."""
    inc = 2 * math.pi / n
    ang = -math.pi + np.arange(n) * inc
    r = np.full(n, 12.0)
    s = np.sin(ang)
    side = np.abs(s) > 1e-6
    r[side] = np.minimum(r[side], half_width / np.abs(s[side]))
    if car is not None:
        cx, cy, w = car
        for a_i in np.nonzero(np.abs(ang) < 1.0)[0]:   # only forward rays can hit it
            a = ang[a_i]
            # ray-box test against a w x w box centred at (cx, cy)
            dx, dy = math.cos(a), math.sin(a)
            for t in np.linspace(0.2, 12.0, 600):
                x, y = t * dx, t * dy
                if abs(x - cx) < w / 2 and abs(y - cy) < w / 2:
                    r[a_i] = min(r[a_i], t)
                    break
    return r, -math.pi, inc


def _straight_raceline(n=200, length=40.0):
    x = np.linspace(-10.0, -10.0 + length, n)
    return x, np.zeros(n), np.full(n, 4.0)


def test_room_lr_reads_corridor_walls():
    r, amin, inc = _corridor_scan(half_width=1.2)
    room_l, room_r = room_lr(r, amin, inc, margin=0.2)
    assert room_l == pytest.approx(1.0, abs=0.05)
    assert room_r == pytest.approx(1.0, abs=0.05)


def test_clear_track_is_cruise_with_zero_offset():
    lay = AvoidanceLayer()
    rl_x, rl_y, rl_v = _straight_raceline()
    r, amin, inc = _corridor_scan()
    for k in range(10):
        cmd = lay.update(k, r, amin, inc, (0.0, 0.0, 0.0), 3.0, 50, rl_x, rl_y, rl_v, 0.1 * k)
    assert cmd.mode == 'CRUISE' and cmd.offset == 0.0
    assert cmd.speed_cap == float('inf') and cmd.opponents == []


def test_car_ahead_is_detected_offset_moves_away_and_speed_is_capped():
    lay = AvoidanceLayer(max_offset=0.5, offset_rate=0.05, follow_gap=1.0)
    rl_x, rl_y, rl_v = _straight_raceline()
    # slow car 2.5 m ahead, slightly left of our line
    r, amin, inc = _corridor_scan(car=(2.5, 0.2, 0.3))
    cmd = None
    for k in range(40):
        cmd = lay.update(k, r, amin, inc, (0.0, 0.0, 0.0), 3.0, 50, rl_x, rl_y, rl_v, 0.1 * k)
    assert len(cmd.opponents) == 1, cmd.thought
    assert cmd.mode in ('ATTACK', 'EVADE'), cmd.thought
    assert cmd.offset < -0.2                       # passes on the RIGHT (opp on left)
    assert abs(cmd.offset) <= 0.5 + 1e-9
    assert math.isfinite(cmd.speed_cap) and cmd.speed_cap < 3.0   # follow governor active
    assert cmd.speed_factor <= 1.0                 # no boost on hardware defaults


def test_offset_is_rate_limited_and_never_exceeds_room():
    lay = AvoidanceLayer(max_offset=0.6, offset_rate=0.05, wall_margin=0.2)
    rl_x, rl_y, rl_v = _straight_raceline()
    # narrow corridor: only 0.4 m of room each side after the margin
    r, amin, inc = _corridor_scan(half_width=0.6, car=(2.0, 0.15, 0.3))
    prev = 0.0
    for k in range(60):
        cmd = lay.update(k, r, amin, inc, (0.0, 0.0, 0.0), 2.0, 50, rl_x, rl_y, rl_v, 0.1 * k)
        assert abs(cmd.offset - prev) <= 0.05 + 1e-9
        assert -cmd.room_right - 1e-6 <= cmd.offset <= cmd.room_left + 1e-6
        prev = cmd.offset


def test_wall_is_not_an_opponent():
    lay = AvoidanceLayer()
    rl_x, rl_y, rl_v = _straight_raceline()
    # dead-end wall 3 m ahead spanning the corridor: long continuous return
    r, amin, inc = _corridor_scan()
    ang = -math.pi + np.arange(720) * inc
    front = np.abs(ang) < math.radians(25)
    r[front] = np.minimum(r[front], 3.0 / np.cos(ang[front]))
    for k in range(10):
        cmd = lay.update(k, r, amin, inc, (0.0, 0.0, 0.0), 2.0, 50, rl_x, rl_y, rl_v, 0.1 * k)
    assert cmd.opponents == [] and cmd.mode == 'CRUISE'


def test_perception_runs_once_per_scan():
    lay = AvoidanceLayer()
    rl_x, rl_y, rl_v = _straight_raceline()
    r, amin, inc = _corridor_scan(car=(3.0, 0.0, 0.3))
    calls = {'n': 0}
    real = lay.det.detect

    def counting(*a, **k):
        calls['n'] += 1
        return real(*a, **k)
    lay.det.detect = counting
    for _ in range(5):                    # 5 control ticks, same scan id
        lay.update(7, r, amin, inc, (0.0, 0.0, 0.0), 2.0, 50, rl_x, rl_y, rl_v, 1.0)
    assert calls['n'] == 1
