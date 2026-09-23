"""Regression tests for the safety layer used by manual and autonomous drive."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
from drive_safety import (CommandArbiter, CommandLimiter, EmergencyBrake,  # noqa: E402
                          finite_clip, forward_clearance, stopping_distance)


def test_finite_clip_rejects_nan_and_infinity():
    assert finite_clip(float('nan'), -1, 1) == 0.0
    assert finite_clip(float('inf'), -1, 1, fallback=-0.2) == -0.2
    assert finite_clip(5, -1, 1) == 1.0


def test_forward_clearance_uses_wrapped_forward_cone():
    angles = np.linspace(-math.pi, math.pi, 9, endpoint=False)
    ranges = np.full(9, 8.0)
    forward = int(np.argmin(np.abs(angles)))
    ranges[forward] = 0.42
    got = forward_clearance(ranges, angles[0], angles[1] - angles[0],
                            half_angle=0.5)
    assert math.isclose(got, 0.42, abs_tol=1e-6)
    assert forward_clearance([], 0.0, 0.1) is None


def test_stopping_distance_includes_reaction_and_braking():
    assert math.isclose(stopping_distance(2.0, 0.35, 0.15, 4.0), 1.15)
    assert stopping_distance(float('nan')) == 0.35


def test_emergency_brake_latches_until_multiple_clear_scans():
    brake = EmergencyBrake(release_margin=0.2, release_samples=2)
    assert brake.update(0.4, 0.5)
    assert brake.update(0.65, 0.5)       # not outside the release margin
    assert brake.update(0.8, 0.5)        # first convincingly clear sample
    assert not brake.update(0.8, 0.5)    # second sample releases
    assert brake.update(None, 0.5)       # autonomy fails closed on invalid scan


def test_command_limiter_smooths_motion_but_not_emergency_stop():
    limiter = CommandLimiter(max_accel=2.0, max_decel=4.0,
                             max_steer_rate=1.0)
    limiter.reset(now=10.0)
    speed, steer = limiter.apply(5.0, 1.0, now=10.1)
    assert np.allclose((speed, steer), (0.2, 0.1))
    assert limiter.apply(5.0, 1.0, now=10.2, emergency=True) == (0.0, 0.0)


def test_command_arbiter_rejects_replay_and_second_driver():
    arbiter = CommandArbiter(lease_timeout=0.5)
    assert arbiter.offer('tab-a', 1, True, now=1.0)[0]
    assert not arbiter.offer('tab-a', 1, True, now=1.1)[0]
    ok, reason = arbiter.offer('tab-b', 1, True, now=1.2)
    assert not ok and 'tab-a' in reason
    assert arbiter.offer('tab-a', 2, False, now=1.3)[0]
    assert arbiter.offer('tab-b', 2, True, now=1.31)[0]
    assert arbiter.current_owner(now=2.0) is None
