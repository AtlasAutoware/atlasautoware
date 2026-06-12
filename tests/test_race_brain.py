"""
Triple-check the race brain: tracker, fusion, camera geometry, and every
strategist branch.  Pure-Python (no ROS) so it runs anywhere numpy is present.

    python3 tests/test_race_brain.py        # prints PASS/FAIL, exits nonzero on fail
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'f1tenth_gym_ros'))
from race_brain import (Opponent, OpponentDetector, RaceStrategist,        # noqa
                        fuse_opponents, project_to_raceline)
import camera_perception as cp                                              # noqa

N = 1080
AMIN = -2.356
AINC = 4.712 / N
ANG = AMIN + np.arange(N) * AINC

_fails = []


def check(name, cond):
    print(f'  [{"PASS" if cond else "FAIL"}] {name}')
    if not cond:
        _fails.append(name)


def scan_with_cars(specs):
    """specs = list of (angle_rad, dist). Returns a ranges array (~0.4 m cars)."""
    r = np.full(N, 6.0, np.float32)
    for a, d in specs:
        hb = max(1, int((0.2 / d) / AINC))
        c = int(np.argmin(np.abs(ANG - a)))
        r[max(0, c - hb):c + hb] = d
    return r


# ─────────────────────────────────────────────────────────────────────────────
def test_tracker():
    print('tracker:')
    det = OpponentDetector()
    # 1) builds confirmation, then reports a single smooth track
    conf = []
    for k in range(6):
        opps = det.detect(scan_with_cars([(0.0, 3.0 - 0.1 * k)]), AMIN, AINC,
                          (0, 0, 0), t=k * 0.02)
        conf.append(len(opps))
    check('confirms after min_hits (not before)', conf[0] == 0 and conf[-1] == 1)
    check('reports exactly one car', conf[-1] == 1)

    # 2) one-frame false blob never confirms
    det2 = OpponentDetector()
    for k in range(4):
        det2.detect(scan_with_cars([(0.0, 3.0)]), AMIN, AINC, (0, 0, 0), k * 0.02)
    spike = det2.detect(scan_with_cars([(0.0, 3.0), (1.2, 2.0)]), AMIN, AINC,
                        (0, 0, 0), 4 * 0.02)
    check('rejects one-frame false blob (still 1)', len(spike) == 1)

    # 3) coasts through a few missed frames
    coast = det2.detect(scan_with_cars([]), AMIN, AINC, (0, 0, 0), 5 * 0.02)
    check('coasts through occlusion', len(coast) == 1)

    # 4) close-range split (two clusters 0.4 m apart) merges to one
    det3 = OpponentDetector()
    out = 0
    for k in range(6):
        out = len(det3.detect(scan_with_cars([(-0.05, 2.0), (0.05, 2.0)]),
                              AMIN, AINC, (0, 0, 0), k * 0.02))
    check('merges split clusters to one car', out == 1)


def test_fusion():
    print('fusion:')
    ego = (0.0, 0.0, 0.0)

    def mk(x, y, s='lidar'):
        return Opponent(x, y, np.hypot(x, y), 0.3, np.arctan2(y, x), s)

    lidar = [mk(3, 0), mk(2, 1.0), mk(-2, 0.5)]
    camera = [mk(3.1, 0.05, 'camera'), mk(2.5, -1.0, 'camera')]
    fused = fuse_opponents(lidar, camera, ego, fov=1.2, match=0.8)
    srcs = sorted((round(f.x, 1), f.source) for f in fused)
    check('matched -> fused', (3.0, 'fused') in srcs)
    check('behind (out of FOV) -> kept lidar', (-2.0, 'lidar') in srcs)
    check('camera-only -> kept', (2.5, 'camera') in srcs)
    check('in-FOV lidar w/o camera -> dropped', (2.0, 'lidar') not in srcs)
    check('count is 3', len(fused) == 3)

    # No camera at all => fusion not invoked by agent; pure call keeps non-FOV only.
    # (Agent guards with cam_live; here we just confirm function purity.)
    only = fuse_opponents([mk(-3, 0)], [], ego, fov=1.2)
    check('lidar-only behind kept when camera empty', len(only) == 1)


def test_camera_geometry():
    print('camera geometry:')
    xf, yl, rng = cp.box_to_relative((270, 200, 100, 120), fx=600, cx_img=320)
    check('depth ~1.8 m for 0.3 m car @100 px', abs(xf - 1.8) < 0.05)
    check('centred -> ~0 lateral', abs(yl) < 0.05)
    _, yr, _ = cp.box_to_relative((470, 200, 100, 120), 600, 320)
    check('right of centre -> negative (right) lateral', yr < -0.3)
    wx, wy = cp.relative_to_world(xf, yl, (10.0, 5.0, 0.0))
    check('world transform correct', abs(wx - 11.8) < 0.05 and abs(wy - 5.0) < 0.05)


def _circle(nn=120, R=12.0):
    th = np.linspace(0, 2 * np.pi, nn, endpoint=False)
    return R * np.cos(th), R * np.sin(th), np.full(nn, 6.0)


def test_strategist():
    print('strategist:')
    rlx, rly, rls = _circle()
    n = len(rlx)
    sp = float(np.mean([np.hypot(rlx[(i+1) % n]-rlx[i], rly[(i+1) % n]-rly[i])
                        for i in range(n)]))
    ego_idx = 0

    def opp_at(idx, lateral=0.0, vx=0.0, vy=0.0):
        # place an opponent `lateral` m left of the raceline at index idx
        i = idx % n
        tx, ty = rlx[(i+1) % n]-rlx[(i-1) % n], rly[(i+1) % n]-rly[(i-1) % n]
        tn = np.hypot(tx, ty)
        nx, ny = -ty/tn, tx/tn
        o = Opponent(rlx[i]+lateral*nx, rly[i]+lateral*ny, 2.0, 0.3, 0.0)
        o.vx, o.vy = vx, vy
        return o

    # CRUISE
    s = RaceStrategist()
    d = s.decide(ego_idx, 6.0, rlx, rly, rls, 1.0, 1.0, [])
    check('no cars -> CRUISE', d.mode == 'CRUISE')

    # ATTACK: slower car a few idx ahead, ego quick
    s = RaceStrategist()
    d = s.decide(ego_idx, 6.0, rlx, rly, rls, 1.0, 1.0, [opp_at(5, lateral=0.4)])
    check('slower car ahead -> ATTACK', d.mode == 'ATTACK')
    check('ATTACK offset within track', -1.0 <= d.offset <= 1.0)

    # DEFEND: car just behind
    s = RaceStrategist()
    d = s.decide(ego_idx, 5.0, rlx, rly, rls, 1.0, 1.0, [opp_at(n-4, lateral=0.3)])
    check('car close behind -> DEFEND', d.mode == 'DEFEND')

    # EVADE: alongside and they are quicker (big forward velocity)
    s = RaceStrategist()
    fast = opp_at(1, lateral=0.3, vx=-50.0, vy=50.0)   # large opp speed -> closing<0
    d = s.decide(ego_idx, 2.0, rlx, rly, rls, 1.0, 1.0, [fast])
    check('alongside & being passed -> EVADE', d.mode == 'EVADE')
    check('EVADE eases speed', d.speed_factor < 1.0)

    # COMPLETE PASS: alongside and we are quicker (opp ~stationary)
    s = RaceStrategist()
    d = s.decide(ego_idx, 6.0, rlx, rly, rls, 1.0, 1.0, [opp_at(1, lateral=0.3)])
    check('alongside & quicker -> ATTACK (complete pass)', d.mode == 'ATTACK')
    check('complete-pass keeps speed up', d.speed_factor >= 1.0)

    # REGRESSION: EVADE must move AWAY from a passing car even when a pass
    # side was committed earlier (a stale commit used to steer INTO them).
    s = RaceStrategist()
    d = s.decide(ego_idx, 6.0, rlx, rly, rls, 1.0, 1.0,
                 [opp_at(5, lateral=-0.4)])              # opp right -> commit LEFT
    check('pass committed left', d.mode == 'ATTACK' and s.commit_side == 1)
    passer = opp_at(1, lateral=0.3, vx=-50.0, vy=50.0)   # now alongside on LEFT
    d = s.decide(ego_idx, 2.0, rlx, rly, rls, 1.0, 1.0, [passer])
    check('passed-while-committed -> EVADE', d.mode == 'EVADE')
    check('EVADE moves away (right), not into the passer', d.offset < 0.0)
    check('EVADE clears the stale committed side', s.commit_side == 0)

    # REGRESSION: lateral offsets are clamped by the wall on THEIR side
    # (+left offsets by room_left); the bounds used to be swapped.
    s = RaceStrategist()
    d = s.decide(ego_idx, 5.0, rlx, rly, rls, 0.1, 2.0,  # left wall 0.1 m away
                 [opp_at(n - 4, lateral=0.3)])           # DEFEND covers the left
    check('DEFEND clamped by the left wall', d.mode == 'DEFEND'
          and 0.0 < d.offset <= 0.1 + 1e-9)
    s = RaceStrategist()
    fast = opp_at(1, lateral=0.3, vx=-50.0, vy=50.0)     # EVADE right
    d = s.decide(ego_idx, 2.0, rlx, rly, rls, 2.0, 0.05, [fast])
    check('EVADE clamped by the right wall', d.mode == 'EVADE'
          and -0.05 - 1e-9 <= d.offset < 0.0)


if __name__ == '__main__':
    test_tracker()
    test_fusion()
    test_camera_geometry()
    test_strategist()
    print()
    if _fails:
        print(f'FAILED: {len(_fails)} -> {_fails}')
        sys.exit(1)
    print('ALL TESTS PASSED')
