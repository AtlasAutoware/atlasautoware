"""
Opponent avoidance layer — multi-car (up to 4-car) obstacle avoidance for the
competition raceline_mpc, pure logic (no ROS) so it is unit-testable.
=============================================================================

Sits between perception and the MPC.  Every control tick it turns the lidar
scan + ego pose into two numbers the tracking controller already understands:

    lateral offset  (m, +left)   -> MPC / MAP reference is shifted off the
                                    raceline to pass, cover or give room
    speed cap       (m/s)        -> ACC-style follow governor so a slower car
                                    ahead is *followed at a gap* instead of
                                    tripping the hard AEB, and passes happen
                                    with headroom

Pieces (all from race_brain, proven in the sim race agent):
    OpponentDetector  racers vs walls from the scan, tracked with velocity
    RaceStrategist    CRUISE / ATTACK / DEFEND / EVADE + target lateral offset
plus, new here:
    follow governor   constant-time-headway speed cap on the nearest car ahead
                      inside the travel cone (centred on the commanded steer)
    offset smoothing  rate-limited, hard-clamped to the lidar-measured room
                      so the strategic line can never point at a wall
    fail-safe         no scan / stale tracks -> offset decays to 0, cap lifts

The layer NEVER commands motion itself: raceline_mpc keeps its own AEB, sensor
watchdog and traction governor in front of the actuator.  This only shapes the
reference the controller tracks and the speed it is allowed.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from race_brain import OpponentDetector, RaceStrategist  # noqa: E402


class AvoidanceCommand:
    __slots__ = ('offset', 'speed_cap', 'speed_factor', 'mode', 'thought',
                 'opponents', 'room_left', 'room_right', 'opp_ahead')

    def __init__(self, offset=0.0, speed_cap=float('inf'), speed_factor=1.0,
                 mode='CRUISE', thought='', opponents=(), room_left=1.0,
                 room_right=1.0, opp_ahead=float('inf')):
        self.offset = offset
        self.speed_cap = speed_cap
        self.speed_factor = speed_factor
        self.mode = mode
        self.thought = thought
        self.opponents = list(opponents)
        self.room_left = room_left
        self.room_right = room_right
        self.opp_ahead = opp_ahead


def room_lr(ranges, angle_min, angle_increment, margin=0.20, default=1.0):
    """Free distance to the left / right walls (m) from beams around +/-90 deg."""
    r = np.asarray(ranges, np.float32)
    r = np.where(np.isfinite(r) & (r > 0.05), r, 30.0)
    ang = angle_min + np.arange(len(r)) * angle_increment
    ang = np.arctan2(np.sin(ang), np.cos(ang))
    left = r[(ang > 1.3) & (ang < 1.84)]          # ~ +90 deg
    right = r[(ang < -1.3) & (ang > -1.84)]
    room_l = float(left.min()) if len(left) else default
    room_r = float(right.min()) if len(right) else default
    return max(room_l - margin, 0.0), max(room_r - margin, 0.0)


def follow_speed_cap(gap, opp_speed_along, ego_speed, follow_gap=1.0,
                     time_headway=0.6, gain=1.5, min_speed=0.0):
    """Constant-time-headway follow law.

    gap              along-track distance to the car ahead (m, > 0)
    opp_speed_along  its speed along the track (m/s, may be 0 for a stopped car)
    Returns the speed cap (m/s); inf when the car is far enough not to matter.
    Desired gap grows with our speed (follow_gap + time_headway * v); inside it
    we converge on the opponent's speed, and inside 60 % of the standing gap we
    back off below it so a stopped car is approached, not hit.
    """
    if not math.isfinite(gap) or gap <= 0.0:
        return float('inf')
    desired = follow_gap + time_headway * max(ego_speed, 0.0)
    if gap >= desired + 1.0:                        # outside the envelope
        return float('inf')
    opp = max(0.0, opp_speed_along)
    cap = opp + gain * (gap - follow_gap)
    if gap < 0.6 * follow_gap:
        cap = min(cap, 0.8 * opp)
    return max(min_speed, cap)


class AvoidanceLayer:
    def __init__(self, max_offset=0.5, offset_rate=0.04, side_clearance=0.55,
                 attack_range=6.0, defend_range=5.0, contact_range=1.0,
                 follow_gap=1.0, time_headway=0.6, follow_gain=1.5,
                 follow_cone=0.5, min_follow_speed=0.0, allow_boost=False,
                 max_range=8.0, wall_margin=0.20, track_half=1.0,
                 detector_kw=None):
        self.max_offset = float(max_offset)
        self.offset_rate = float(offset_rate)       # m per update tick
        self.follow_gap = float(follow_gap)
        self.time_headway = float(time_headway)
        self.follow_gain = float(follow_gain)
        self.follow_cone = float(follow_cone)       # rad half-angle
        self.min_follow_speed = float(min_follow_speed)
        self.allow_boost = bool(allow_boost)
        self.wall_margin = float(wall_margin)
        self.det = OpponentDetector(max_range=max_range, **(detector_kw or {}))
        self.strat = RaceStrategist(attack_range=attack_range,
                                    defend_range=defend_range,
                                    contact_range=contact_range,
                                    side_clearance=side_clearance,
                                    track_half=track_half)
        self.applied_offset = 0.0
        self.last = AvoidanceCommand()
        self._scan_id = None
        self._opps = []

    # ── perception (only re-run when a new scan arrived) ────────────────────
    def perceive(self, scan_id, ranges, angle_min, angle_increment, ego, t):
        if scan_id != self._scan_id:
            self._scan_id = scan_id
            self._opps = self.det.detect(ranges, angle_min, angle_increment, ego, t)
        return self._opps

    def reset(self):
        self.applied_offset = 0.0
        self.strat.commit_side = 0
        self.last = AvoidanceCommand()

    # ── one tick ─────────────────────────────────────────────────────────────
    def update(self, scan_id, ranges, angle_min, angle_increment, ego, speed,
               nearest_idx, rl_x, rl_y, rl_speed, t, steer_cmd=0.0,
               overtake_idxs=()):
        """ego = (x, y, yaw).  Returns an AvoidanceCommand; also stored in .last."""
        opps = self.perceive(scan_id, ranges, angle_min, angle_increment, ego, t)
        room_l, room_r = room_lr(ranges, angle_min, angle_increment,
                                 margin=self.wall_margin)
        d = self.strat.decide(nearest_idx, speed, rl_x, rl_y, rl_speed,
                              room_l, room_r, opps, overtake_idxs)

        # target offset, hard-clamped by lidar room and the global max, then
        # rate-limited so the reference never jumps
        target = float(np.clip(d.offset, -min(self.max_offset, room_r),
                               min(self.max_offset, room_l)))
        step = np.clip(target - self.applied_offset,
                       -self.offset_rate, self.offset_rate)
        self.applied_offset = float(np.clip(self.applied_offset + step,
                                            -self.max_offset, self.max_offset))

        # follow governor on the nearest confirmed car ahead in the travel cone
        cap, opp_ahead = self._follow_cap(opps, ego, speed, steer_cmd)

        factor = float(d.speed_factor)
        if not self.allow_boost:
            factor = min(factor, 1.0)

        self.last = AvoidanceCommand(
            offset=self.applied_offset, speed_cap=cap, speed_factor=factor,
            mode=d.mode, thought=d.thought, opponents=opps,
            room_left=room_l, room_right=room_r, opp_ahead=opp_ahead)
        return self.last

    def _follow_cap(self, opps, ego, speed, steer_cmd):
        ex, ey, eyaw = ego
        best_cap, best_d = float('inf'), float('inf')
        for o in opps:
            dx, dy = o.x - ex, o.y - ey
            bearing = math.atan2(dy, dx) - eyaw
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            if abs(bearing - steer_cmd) > self.follow_cone:
                continue
            dist = math.hypot(dx, dy)
            # opponent speed component along OUR heading (what we'd close on)
            v_along = o.vx * math.cos(eyaw) + o.vy * math.sin(eyaw)
            cap = follow_speed_cap(dist, v_along, speed, self.follow_gap,
                                   self.time_headway, self.follow_gain,
                                   self.min_follow_speed)
            if dist < best_d:
                best_d = dist
            best_cap = min(best_cap, cap)
        return best_cap, best_d
