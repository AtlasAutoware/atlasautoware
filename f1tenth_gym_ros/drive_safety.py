"""Small, ROS-free safety primitives shared by every AtlasCar drive path.

Keeping these functions independent of ROS makes the exact obstacle, command and
input-arbitration behavior cheap to regression-test.  Nodes remain responsible for
sensor freshness; these helpers make a fresh scan and a finite command safe to use.
"""

import math
import time

import numpy as np


def finite_clip(value, lower, upper, fallback=0.0):
    """Return a finite float inside ``[lower, upper]`` or ``fallback``."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if not math.isfinite(value):
        return float(fallback)
    return max(float(lower), min(float(upper), value))


def forward_clearance(ranges, angle_min, angle_increment, half_angle=0.20,
                      range_min=0.03, range_max=float('inf')):
    """Nearest valid return in a forward cone, or ``None`` if none is usable.

    Angles are wrapped before comparison, so scans whose angular interval crosses
    +/-pi work just as well as the usual ``[-pi, pi)`` lidar layout.
    """
    values = np.asarray(ranges, dtype=np.float32)
    if values.size == 0 or not math.isfinite(float(angle_increment)):
        return None
    angles = float(angle_min) + np.arange(values.size) * float(angle_increment)
    angles = np.arctan2(np.sin(angles), np.cos(angles))
    valid = (np.isfinite(values) & (values > float(range_min))
             & (values < float(range_max))
             & (np.abs(angles) <= abs(float(half_angle))))
    return float(values[valid].min()) if valid.any() else None


def stopping_distance(speed, margin=0.35, reaction_time=0.15, decel=4.0):
    """Conservative forward stopping envelope in metres."""
    speed = max(0.0, finite_clip(speed, 0.0, 1000.0))
    decel = max(0.1, finite_clip(decel, 0.1, 1000.0, 0.1))
    return max(0.0, float(margin)) + speed * max(0.0, float(reaction_time)) \
        + speed * speed / (2.0 * decel)


class EmergencyBrake:
    """Hysteretic AEB latch that cannot chatter at one noisy range boundary."""

    def __init__(self, release_margin=0.15, release_samples=3):
        self.release_margin = max(0.0, float(release_margin))
        self.release_samples = max(1, int(release_samples))
        self.engaged = False
        self._clear_samples = 0

    def reset(self):
        self.engaged = False
        self._clear_samples = 0

    def update(self, clearance, threshold, fail_closed=True):
        threshold = max(0.0, float(threshold))
        if clearance is None or not math.isfinite(float(clearance)):
            if fail_closed:
                self.engaged = True
            self._clear_samples = 0
            return self.engaged
        if float(clearance) <= threshold:
            self.engaged = True
            self._clear_samples = 0
        elif self.engaged:
            if float(clearance) >= threshold + self.release_margin:
                self._clear_samples += 1
                if self._clear_samples >= self.release_samples:
                    self.reset()
            else:
                self._clear_samples = 0
        return self.engaged


class CommandLimiter:
    """Rate-limit ordinary policy commands while preserving immediate stops."""

    def __init__(self, max_accel=2.5, max_decel=5.0, max_steer_rate=2.5):
        self.max_accel = max(0.01, float(max_accel))
        self.max_decel = max(0.01, float(max_decel))
        self.max_steer_rate = max(0.01, float(max_steer_rate))
        self.speed = 0.0
        self.steer = 0.0
        self.last_time = None

    @staticmethod
    def _step(current, target, limit):
        return current + max(-limit, min(limit, target - current))

    def reset(self, now=None):
        self.speed = self.steer = 0.0
        self.last_time = time.monotonic() if now is None else float(now)
        return self.speed, self.steer

    def apply(self, speed, steer, now=None, emergency=False):
        now = time.monotonic() if now is None else float(now)
        if emergency:
            return self.reset(now)
        dt = 0.0 if self.last_time is None else max(0.0, min(0.5, now - self.last_time))
        self.last_time = now
        speed = float(speed)
        steer = float(steer)
        rate = self.max_decel if speed < self.speed else self.max_accel
        self.speed = self._step(self.speed, speed, rate * dt)
        self.steer = self._step(self.steer, steer, self.max_steer_rate * dt)
        return self.speed, self.steer


class CommandArbiter:
    """Single-driver lease plus replay/out-of-order protection.

    The first armed client owns control until it explicitly releases the dead-man
    or its lease expires.  Sequence numbers are tracked per client, preventing a
    late HTTP request from re-applying an older command.
    """

    def __init__(self, lease_timeout=0.75, max_clients=64):
        self.lease_timeout = max(0.1, float(lease_timeout))
        self.max_clients = max(4, int(max_clients))
        self.owner = None
        self.owner_time = 0.0
        self.sequences = {}

    def _expire(self, now):
        if self.owner is not None and now - self.owner_time > self.lease_timeout:
            self.owner = None

    def offer(self, client, sequence, armed, now=None):
        now = time.monotonic() if now is None else float(now)
        client = str(client or 'legacy')[:96]
        self._expire(now)
        if sequence is not None:
            try:
                sequence = int(sequence)
            except (TypeError, ValueError):
                return False, 'bad sequence'
            previous = self.sequences.get(client)
            if previous is not None and sequence <= previous[0]:
                return False, 'stale command'
            self.sequences[client] = (sequence, now)
            if len(self.sequences) > self.max_clients:
                oldest = min(self.sequences, key=lambda key: self.sequences[key][1])
                self.sequences.pop(oldest, None)
        if self.owner is not None and self.owner != client:
            return False, f'control held by {self.owner}'
        if armed:
            self.owner = client
            self.owner_time = now
        elif self.owner == client:
            self.owner = None
        return True, 'accepted'

    def current_owner(self, now=None):
        now = time.monotonic() if now is None else float(now)
        self._expire(now)
        return self.owner
