"""
Race brain — opponent perception + attack/defend strategy (pure logic, no ROS).
================================================================================

Two pieces, both framework-free so they can be unit-tested without a simulator:

  OpponentDetector  — tells **racers from walls** using only the lidar scan.
      Walls are long, continuous returns; a racer is a short, car-sized cluster
      that is isolated (free space jumps open on both angular sides).  Clusters
      are tracked frame-to-frame to estimate velocity.

  RaceStrategist    — the "thinking".  Given the ego's place on the raceline and
      the detected opponents, it picks a mode and a target line:
        CRUISE  — track clear, run the optimal raceline.
        ATTACK  — slower car ahead: close up, pick the side with room, and commit
                  to the pass (preferably out-braking into a corner / down a
                  straight, using the marked overtaking zones).
        DEFEND  — car behind & closing: cover the inside line into the next corner
                  so the pass is harder, without leaving the track.
        EVADE   — a car is alongside / contact imminent: hold a predictable line
                  and give room — finishing the race beats winning a corner.
      Every decision carries a plain-English `thought` so the reasoning is visible.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Perception: racers vs walls
# ─────────────────────────────────────────────────────────────────────────────

class Opponent:
    __slots__ = ('x', 'y', 'vx', 'vy', 'dist', 'width', 'angle',
                 's_idx', 'lateral', 'gap', 'closing', 'source')

    def __init__(self, x, y, dist, width, angle, source='lidar'):
        self.x, self.y = x, y
        self.vx = self.vy = 0.0
        self.dist, self.width, self.angle = dist, width, angle
        self.s_idx = 0          # nearest raceline index
        self.lateral = 0.0      # signed offset from raceline (+left)
        self.gap = 0.0          # along-track distance ahead(+)/behind(-) of ego (m)
        self.closing = 0.0      # closing speed along track (m/s, + = gap shrinking)
        self.source = source    # 'lidar' | 'camera' | 'fused'


class _Track:
    """One smoothed opponent, maintained across frames by an alpha-beta filter."""
    __slots__ = ('x', 'y', 'vx', 'vy', 'dist', 'width', 'angle',
                 'hits', 'misses', 'confirmed')

    def __init__(self, det):
        self.x, self.y = det.x, det.y
        self.vx = self.vy = 0.0
        self.dist, self.width, self.angle = det.dist, det.width, det.angle
        self.hits, self.misses, self.confirmed = 1, 0, False


class OpponentDetector:
    """
    Cluster the scan into car-sized isolated blobs (racers, not walls), then run
    a small multi-target tracker so the strategy sees a *stable*, *smooth* picture:

      - nearest-neighbour association inside a gate,
      - an alpha-beta filter smoothing each car's position AND velocity (so the
        reported closing speed is steady, not frame-to-frame noise),
      - confirm-before-report (a blob must persist `min_hits` frames) — kills
        one-frame false positives,
      - coast-on-miss for `max_miss` frames — survives momentary occlusion so the
        mode doesn't flicker back to CRUISE mid-overtake.
    """
    def __init__(self, max_range=8.0, jump=0.30, car_min=0.12, car_max=1.0,
                 gate=1.2, alpha=0.5, beta=0.20, min_hits=4, max_miss=12,
                 v_max=8.0, merge_dist=1.0):
        self.max_range, self.jump = max_range, jump
        self.car_min, self.car_max = car_min, car_max
        self.gate, self.alpha, self.beta = gate, alpha, beta
        self.min_hits, self.max_miss, self.v_max = min_hits, max_miss, v_max
        self.merge_dist = merge_dist     # collapse split clusters of one car
        self.tracks = []
        self.t_prev = None

    def detect(self, ranges, angle_min, angle_inc, ego, t):
        """ego = (x, y, yaw).  Returns confirmed, smoothed Opponents (world frame)."""
        raw = self._raw_detections(ranges, angle_min, angle_inc, ego)
        dt = 0.0 if self.t_prev is None else min(t - self.t_prev, 0.1)
        self.t_prev = t
        self._update_tracks(raw, dt)
        out = []
        for tr in self.tracks:
            if not tr.confirmed:
                continue
            o = Opponent(tr.x, tr.y, tr.dist, tr.width, tr.angle)
            o.vx, o.vy = tr.vx, tr.vy
            out.append(o)
        return out

    def _raw_detections(self, ranges, angle_min, angle_inc, ego):
        r = np.asarray(ranges, dtype=np.float32)
        valid = np.isfinite(r) & (r > 0.05) & (r < self.max_range)
        n = len(r)
        angles = angle_min + np.arange(n) * angle_inc
        segs, cur = [], []
        for i in range(n):
            if not valid[i]:
                if cur:
                    segs.append(cur); cur = []
                continue
            if cur and abs(r[i] - r[cur[-1]]) > self.jump:
                segs.append(cur); cur = [i]
            else:
                cur.append(i)
        if cur:
            segs.append(cur)

        ex, ey, eyaw = ego
        found = []
        for seg in segs:
            rmid = float(np.median(r[seg]))
            width = rmid * (angles[seg[-1]] - angles[seg[0]])      # arc width (m)
            if not (self.car_min < width < self.car_max):
                continue                                            # wall or noise
            li, ri = seg[0] - 1, seg[-1] + 1                        # isolation test
            left_free = li < 0 or not valid[li] or (r[li] - rmid) > self.jump
            right_free = ri >= n or not valid[ri] or (r[ri] - rmid) > self.jump
            if not (left_free and right_free):
                continue
            amid = float(angles[seg[len(seg) // 2]])
            lx, ly = rmid * np.cos(amid), rmid * np.sin(amid)       # car frame
            wx = ex + lx * np.cos(eyaw) - ly * np.sin(eyaw)
            wy = ey + lx * np.sin(eyaw) + ly * np.cos(eyaw)
            found.append(Opponent(wx, wy, rmid, width, amid))
        return found

    def _update_tracks(self, dets, dt):
        # Predict every existing track forward (constant velocity).
        for tr in self.tracks:
            tr.x += tr.vx * dt
            tr.y += tr.vy * dt

        # Greedy nearest-neighbour association within the gate.
        pairs = []
        for di, d in enumerate(dets):
            for ti, tr in enumerate(self.tracks):
                dist = np.hypot(d.x - tr.x, d.y - tr.y)
                if dist < self.gate:
                    pairs.append((dist, di, ti))
        pairs.sort()
        used_d, used_t = set(), set()
        for dist, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            used_d.add(di); used_t.add(ti)
            self._ab_update(self.tracks[ti], dets[di], dt)

        # Unmatched existing tracks -> coasted miss (count BEFORE adding new ones).
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.misses += 1
        # Unmatched detections -> new tentative tracks.
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks.append(_Track(d))

        # Confirm / delete.
        for tr in self.tracks:
            if tr.hits >= self.min_hits:
                tr.confirmed = True
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_miss]

        # Merge near-duplicate tracks: a single car at close range can split into
        # two lidar clusters -> two tracks. Keep the stronger (more hits) of any
        # pair within merge_dist.  This removes the residual close-range opp=2.
        self.tracks.sort(key=lambda t: -t.hits)
        kept = []
        for tr in self.tracks:
            if all(np.hypot(tr.x - k.x, tr.y - k.y) > self.merge_dist for k in kept):
                kept.append(tr)
        self.tracks = kept

    def _ab_update(self, tr, det, dt):
        if dt > 1e-3:
            rx, ry = det.x - tr.x, det.y - tr.y              # residual vs prediction
            tr.x += self.alpha * rx
            tr.y += self.alpha * ry
            tr.vx += (self.beta / dt) * rx
            tr.vy += (self.beta / dt) * ry
            tr.vx = float(np.clip(tr.vx, -self.v_max, self.v_max))
            tr.vy = float(np.clip(tr.vy, -self.v_max, self.v_max))
        else:
            tr.x, tr.y = det.x, det.y
        tr.dist, tr.width, tr.angle = det.dist, det.width, det.angle
        tr.hits += 1
        tr.misses = 0


# ─────────────────────────────────────────────────────────────────────────────
# Raceline geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def project_to_raceline(x, y, rl_x, rl_y):
    """Nearest raceline index + signed lateral offset (+left of travel)."""
    d2 = (rl_x - x) ** 2 + (rl_y - y) ** 2
    i = int(np.argmin(d2))
    n = len(rl_x)
    tx = rl_x[(i + 1) % n] - rl_x[(i - 1) % n]
    ty = rl_y[(i + 1) % n] - rl_y[(i - 1) % n]
    tn = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / tn, tx / tn                       # left normal
    lateral = (x - rl_x[i]) * nx + (y - rl_y[i]) * ny
    return i, lateral


# ─────────────────────────────────────────────────────────────────────────────
# Sensor fusion: lidar (precise range) + camera (class + bearing)
# ─────────────────────────────────────────────────────────────────────────────

def fuse_opponents(lidar, camera, ego, fov=1.2, match=0.8):
    """
    Merge lidar and camera opponent lists (both in world frame).

      - matched (both sensors see it): keep the LIDAR estimate (precise range +
        velocity), tagged 'fused' — high confidence.
      - lidar-only INSIDE the camera's field of view: the camera should have seen
        a real car there and didn't → almost certainly a wall artifact → DROP.
        (This is what removes the lidar false positives.)
      - lidar-only OUTSIDE the FOV (behind / far to the side): the camera can't
        see it → trust lidar, keep it.
      - camera-only (lidar missed it — e.g. the car merged with a wall in the
        scan): keep the camera estimate.

    ego = (x, y, yaw).  fov = camera half-angle (rad).  match = assoc gate (m).
    """
    fused, used_cam = [], set()
    for lo in lidar:
        best, bd = None, match
        for ci, co in enumerate(camera):
            if ci in used_cam:
                continue
            d = np.hypot(lo.x - co.x, lo.y - co.y)
            if d < bd:
                bd, best = d, ci
        if best is not None:
            used_cam.add(best)
            lo.source = 'fused'
            fused.append(lo)
        else:
            ang = np.arctan2(lo.y - ego[1], lo.x - ego[0]) - ego[2]
            bearing = np.arctan2(np.sin(ang), np.cos(ang))
            if abs(bearing) >= fov:               # camera can't see it -> trust lidar
                fused.append(lo)
            # else: in-FOV but camera saw no car here -> drop (wall artifact)
    for ci, co in enumerate(camera):
        if ci not in used_cam:
            co.source = 'camera'
            fused.append(co)
    return fused


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

class Decision:
    def __init__(self, mode, offset, speed_factor, thought, target=None):
        self.mode = mode                 # CRUISE / ATTACK / DEFEND / EVADE
        self.offset = offset             # target lateral offset from raceline (m, +left)
        self.speed_factor = speed_factor # multiplies raceline speed
        self.thought = thought           # human-readable reasoning
        self.target = target             # optional (x,y) point of interest for viz


class RaceStrategist:
    def __init__(self, attack_range=6.0, defend_range=5.0, contact_range=1.0,
                 side_clearance=0.55, track_half=1.0):
        self.attack_range = attack_range     # see a car this far ahead to attack
        self.defend_range = defend_range
        self.contact_range = contact_range   # alongside / contact bubble
        self.side_clearance = side_clearance # how far to sit beside the opponent (m)
        self.track_half = track_half         # usable half-width (m), for clamping
        self.commit_side = 0                 # remembered pass side (hysteresis)

    def decide(self, ego_idx, ego_speed, rl_x, rl_y, rl_speed, room_left, room_right,
               opponents, overtake_idxs=()):
        n = len(rl_x)
        spacing = self._spacing(rl_x, rl_y)

        # Place each opponent along the track relative to the ego.
        rel = []
        for o in opponents:
            o.s_idx, o.lateral = project_to_raceline(o.x, o.y, rl_x, rl_y)
            d_idx = (o.s_idx - ego_idx) % n
            gap = d_idx * spacing
            if gap > n * spacing / 2:                 # wrap: it's behind us
                gap -= n * spacing
            o.gap = gap
            # closing speed projected along track tangent
            ti = (o.s_idx + 1) % n
            tx, ty = rl_x[ti] - rl_x[o.s_idx], rl_y[ti] - rl_y[o.s_idx]
            tn = np.hypot(tx, ty) + 1e-9
            opp_v_along = (o.vx * tx + o.vy * ty) / tn
            o.closing = ego_speed - opp_v_along       # >0 we're catching up
            rel.append(o)

        ahead = [o for o in rel if 0.0 < o.gap < self.attack_range]
        behind = [o for o in rel if -self.defend_range < o.gap < 0.0]
        alongside = [o for o in rel if abs(o.gap) < self.contact_range]

        # ── Alongside a car: either COMPLETE the pass (we're faster) or EVADE. ──
        if alongside:
            o = min(alongside, key=lambda o: abs(o.gap))
            # Sit on the side away from the opponent; honour a committed pass side.
            if self.commit_side != 0:
                side = self.commit_side
            else:
                side = -np.sign(o.lateral) if abs(o.lateral) > 0.05 else 1
            offset = np.clip(side * self.side_clearance, -room_left, room_right)
            theirs = 'left' if o.lateral > 0 else 'right'
            ours = 'LEFT' if side > 0 else 'RIGHT'

            if o.closing > 0.3:
                # We're the quicker car drawing alongside a slower one — don't sit
                # there yielding (that deadlocks); hold our side and drive past.
                self.commit_side = side
                return Decision('ATTACK', offset, 1.05,
                                f'ATTACK: alongside on the {ours} and quicker '
                                f'({min(o.closing, 9.0):.1f} m/s) — completing the '
                                f'pass, holding my line', target=(o.x, o.y))
            # They are as quick or quicker (passing us) — give room, avoid contact.
            return Decision('EVADE', offset, 0.92,
                            f'EVADE: car alongside ({theirs}, {o.gap:+.1f} m) — '
                            f'giving room to avoid contact', target=(o.x, o.y))

        # ── ATTACK: slower car ahead within reach. ──
        if ahead:
            o = min(ahead, key=lambda o: o.gap)
            near_zone = any((o.s_idx - i) % n < 30 or (i - o.s_idx) % n < 8
                            for i in overtake_idxs) if overtake_idxs else False
            # Pick the side with more room, away from the opponent's position.
            if o.lateral >= 0:                        # opp on the left → pass right
                side, room = -1, room_right
            else:
                side, room = +1, room_left
            if self.commit_side != 0:                 # hysteresis once committed
                side = self.commit_side
                room = room_left if side > 0 else room_right
            offset = np.clip(side * (abs(o.lateral) + self.side_clearance),
                             -room_left, room_right)
            sidestr = 'LEFT' if side > 0 else 'RIGHT'

            if o.gap > self.attack_range * 0.7 and o.closing < 0.3:
                # Not closing yet — tuck into the slipstream for a run.
                self.commit_side = 0
                return Decision('ATTACK', np.clip(side*0.15, -room_left, room_right),
                                1.05,
                                f'ATTACK: tracking car {o.gap:.1f} m ahead — in the '
                                f'tow, building a run before committing',
                                target=(o.x, o.y))
            # Commit to the pass.
            self.commit_side = side
            zone = ' (overtaking zone — out-braking)' if near_zone else ''
            return Decision('ATTACK', offset, 1.08,
                            f'ATTACK: passing {sidestr} on car {o.gap:.1f} m ahead, '
                            f'closing {max(min(o.closing, 9.0), 0):.1f} m/s{zone}',
                            target=(o.x, o.y))
        self.commit_side = 0

        # ── DEFEND: car behind and closing → cover the line. ──
        if behind:
            o = max(behind, key=lambda o: o.gap)      # closest behind
            if o.closing > 0.2 or abs(o.gap) < self.defend_range * 0.6:
                # Move to cover the side they'd attack (their lateral side).
                side = np.sign(o.lateral) if abs(o.lateral) > 0.05 else 1
                offset = np.clip(side * 0.35, -room_left, room_right)
                sidestr = 'inside/left' if side > 0 else 'inside/right'
                return Decision('DEFEND', offset, 1.0,
                                f'DEFEND: car {abs(o.gap):.1f} m behind & closing — '
                                f'covering the {sidestr} line into the next corner',
                                target=(o.x, o.y))

        # ── CRUISE ──
        return Decision('CRUISE', 0.0, 1.0, 'CRUISE: track clear — optimal raceline')

    @staticmethod
    def _spacing(rl_x, rl_y):
        n = len(rl_x)
        return float(np.mean([np.hypot(rl_x[(i+1) % n]-rl_x[i],
                                       rl_y[(i+1) % n]-rl_y[i]) for i in range(n)]))
