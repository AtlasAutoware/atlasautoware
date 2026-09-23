#!/usr/bin/env python3
"""policy_io: the ONE definition of the student policy's inputs and outputs (ROS-free).

Training (ml/train_student.py, ml/train_policy.py), closed-loop evaluation in the sim
(ml/sim_rollout.py) and the car (policy_bridge.py) all import from here, so the three can
no longer drift apart. They had: the dataset stores actions as (speed, steer) while
policy_bridge unpacked the network output as (steer, speed), which on the car turned the
predicted speed (~1.4 m/s) into a steering command (clamped to full left lock) and the
predicted steering angle (~0 rad) into the speed. See docs/ml_audit_2026-09-23.md.
"""
import hashlib
import numpy as np

FRONT_HW = (96, 128)          # front camera, resized (H, W), BGR uint8
BEV_HW = (96, 96)             # lidar bird's-eye raster
BEV_EXTENT = 6.0              # metres from the car to the raster edge
MAX_TOK = 24
TXT_VOCAB = 4096
# Order of the 2-vector the network predicts. Every dataset in this repo was written by
# EpisodeWriter.set_action(speed, steer) -> act = (speed, steer); the student learns that.
ACTION_ORDER = ('speed', 'steer')
IDX_SPEED, IDX_STEER = 0, 1


def text_ids(s, n=TXT_VOCAB, max_tok=MAX_TOK):
    """Hashed bag-of-words token ids (no downloads, deterministic across machines)."""
    toks = ''.join(c if c.isalnum() else ' ' for c in s.lower()).split()[:max_tok]
    ids = [1 + int(hashlib.md5(t.encode()).hexdigest(), 16) % (n - 1) for t in toks]
    return ids + [0] * (max_tok - len(ids))


def bev_image(ranges, angle_min, angle_inc, size=BEV_HW[0], extent=BEV_EXTENT):
    """Rasterise a scan to a size x size image: car at the centre, forward = up, left = left."""
    img = np.zeros((size, size), np.uint8)
    r = np.asarray(ranges, np.float32); n = len(r)
    ang = angle_min + angle_inc * np.arange(n)
    ok = np.isfinite(r) & (r > 0.05) & (r < extent)
    x, y = r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])
    px = (size / 2 - x / extent * size / 2).astype(int)
    py = (size / 2 - y / extent * size / 2).astype(int)
    m = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    img[px[m], py[m]] = 255
    img[size // 2 - 1:size // 2 + 2, size // 2 - 1:size // 2 + 2] = 128
    return img


def front_image(bgr):
    """Full-size BGR frame -> the (96, 128, 3) uint8 the network sees."""
    import cv2
    return cv2.resize(bgr, (FRONT_HW[1], FRONT_HW[0]), interpolation=cv2.INTER_AREA)


def make_feed(front_bgr_small, bev, state, ids):
    """Numpy inputs for the ONNX session (batch of 1)."""
    return {'front': (front_bgr_small.transpose(2, 0, 1)[None] / 255.0).astype(np.float32),
            'bev': (bev[None, None] / 255.0).astype(np.float32),
            'state': np.asarray(state, np.float32).reshape(1, 5),
            'ids': np.asarray(ids, np.int64).reshape(1, MAX_TOK)}


def action_order_of(session):
    """Read the output order from ONNX metadata; models exported before 2026-09-23 carry
    none, and every one of them was trained on (speed, steer)."""
    try:
        meta = session.get_modelmeta().custom_metadata_map or {}
    except Exception:
        meta = {}
    order = tuple(s.strip() for s in meta.get('action_order', 'speed,steer').split(','))
    if sorted(order) != ['speed', 'steer']:
        raise ValueError(f'bad action_order metadata {order!r}')
    return order


def split_action(vec, order=ACTION_ORDER):
    """Network output -> (speed, steer) regardless of the model's output order."""
    d = dict(zip(order, (float(vec[0]), float(vec[1]))))
    return d['speed'], d['steer']


def front_clear(ranges, angle_min, angle_inc, half_width_rad=0.2):
    """Nearest valid return inside +/- half_width of straight ahead (bearing 0), whatever the
    scan's angle convention (-pi..pi or 0..2pi)."""
    r = np.asarray(ranges, np.float32)
    ang = angle_min + angle_inc * np.arange(len(r))
    ang = (ang + np.pi) % (2 * np.pi) - np.pi
    m = (np.abs(ang) <= half_width_rad) & np.isfinite(r) & (r > 0.05)
    return float(r[m].min()) if m.any() else 99.0
