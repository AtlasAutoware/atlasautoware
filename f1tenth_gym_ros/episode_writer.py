#!/usr/bin/env python3
"""Episode writer: the ROS-free core of the demonstration logger.

An episode is one drive with one instruction attached ("follow the hallway to the
double doors and stop"). While recording, every camera frame becomes a row, and each
row carries the most recent value of every other stream plus that stream's own
timestamp, so a training script can decide how stale is too stale. The IMU and the
action stream are also kept at full rate in a side table, because 10 Hz is fine for
frames but not for a 200 Hz gyro or a 50 Hz command.

Layout on disk (one directory per episode, nothing exotic so the Jetson needs only
numpy and OpenCV; tools/episodes_to_lerobot.py repackages on the laptop):

    <root>/<episode_id>/
        meta.json          instruction, label, fps, counts, topics, car config
        frames/000000.jpg  colour frames (JPEG, quality 92)
        sync.npz           per-frame arrays, all length N (see FIELDS)
        hires.npz          imu (K,7) t ax ay az gx gy gz ; action (M,5) t speed steer src kind

Actions: `src` is 0 human (teleop), 1 policy (/drive); `kind` is 0 for an Ackermann
command (speed m/s, steer rad) and 1 for a raw joystick axis pair (throttle, steer in
-1..1). Both are logged when present; the Ackermann one is what a policy should imitate.
"""
import json, os, threading, time
import numpy as np

try:
    import cv2
except ImportError:                    # tests on a machine without OpenCV still run
    cv2 = None

FIELDS = {
    't':            'image stamp (s)',
    'scan':         'lidar ranges (m), 0 = invalid',
    'scan_t':       'lidar stamp',
    'pose':         'x y yaw (m, m, rad) from odom',
    'vel':          'vx wz (m/s, rad/s) from odom',
    'odom_t':       'odom stamp',
    'imu':          'ax ay az gx gy gz (body frame)',
    'imu_t':        'imu stamp',
    'act':          'speed steer (m/s, rad): last Ackermann command from any source',
    'act_src':      '0 human 1 policy',
    'act_t':        'command stamp',
    'joy':          'throttle steer axes (-1..1) from /joy',
    'volts':        'battery (V)',
}


class EpisodeWriter:
    def __init__(self, root, car_config=None, jpeg_quality=92, scan_len=None):
        self.root = os.path.expanduser(root)
        self.car = car_config or {}
        self.q = int(jpeg_quality)
        self.scan_len = scan_len
        self.lock = threading.Lock()
        self.latest = {'scan': None, 'scan_t': 0.0, 'pose': None, 'vel': None, 'odom_t': 0.0,
                       'imu': None, 'imu_t': 0.0, 'act': None, 'act_src': -1, 'act_t': 0.0,
                       'joy': None, 'volts': 0.0}
        self.ep = None                                   # active episode state

    # ── stream updates (call from any thread) ──────────────────────────────────
    def update(self, key, value, t=None):
        with self.lock:
            self.latest[key] = value
            if t is not None:
                self.latest[key + '_t'] = float(t)
            if self.ep is not None:
                if key == 'imu':
                    self.ep['imu_hi'].append((float(t), *[float(v) for v in value]))
                elif key == 'act':
                    self.ep['act_hi'].append((float(t), float(value[0]), float(value[1]),
                                              int(self.latest['act_src']), 0))
                elif key == 'joy':
                    self.ep['act_hi'].append((float(t), float(value[0]), float(value[1]), 0, 1))

    def stamp(self, key, t):
        with self.lock:
            self.latest[key + '_t'] = float(t)

    def set_action(self, speed, steer, src, t):
        with self.lock:
            self.latest['act_src'] = int(src)
        self.update('act', (speed, steer), t)

    # ── episode control ─────────────────────────────────────────────────────────
    def recording(self):
        return self.ep is not None

    def start(self, instruction, episode_id=None):
        if self.ep is not None:
            return False, 'already recording'
        # IDs must never collide: a stop+start inside one second, or a discard, would
        # otherwise reuse (and on discard, delete) the previous episode's directory.
        eid = episode_id or time.strftime('ep_%Y%m%d_%H%M%S')
        d = os.path.join(self.root, eid); k = 1
        while os.path.exists(d):
            d = os.path.join(self.root, f'{eid}_{k}'); k += 1
        eid = os.path.basename(d)
        os.makedirs(os.path.join(d, 'frames'))
        with self.lock:
            self.ep = {'id': eid, 'dir': d, 'instruction': str(instruction), 't0': time.time(),
                       'rows': [], 'imu_hi': [], 'act_hi': [], 'n': 0}
        return True, eid

    def add_frame(self, bgr, t):
        """Store one camera frame plus a snapshot of every other stream."""
        with self.lock:
            ep = self.ep
            if ep is None:
                return False
            L = dict(self.latest)
        i = ep['n']; ep['n'] += 1
        if cv2 is not None:
            cv2.imwrite(os.path.join(ep['dir'], 'frames', f'{i:06d}.jpg'), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, self.q])
        scan = L['scan']
        if scan is not None:
            if self.scan_len is None: self.scan_len = len(scan)
            s = np.zeros(self.scan_len, np.float32); m = min(self.scan_len, len(scan))
            s[:m] = np.asarray(scan[:m], np.float32)
        else:
            s = np.zeros(self.scan_len or 1, np.float32)
        row = {
            't': float(t), 'scan': s, 'scan_t': L['scan_t'],
            'pose': np.asarray(L['pose'] if L['pose'] is not None else (0, 0, 0), np.float32),
            'vel': np.asarray(L['vel'] if L['vel'] is not None else (0, 0), np.float32),
            'odom_t': L['odom_t'],
            'imu': np.asarray(L['imu'] if L['imu'] is not None else (0,) * 6, np.float32),
            'imu_t': L['imu_t'],
            'act': np.asarray(L['act'] if L['act'] is not None else (0, 0), np.float32),
            'act_src': int(L['act_src']), 'act_t': L['act_t'],
            'joy': np.asarray(L['joy'] if L['joy'] is not None else (0, 0), np.float32),
            'volts': float(L['volts'] or 0.0),
        }
        with self.lock:
            ep['rows'].append(row)
        return True

    def stop(self, label='unlabelled', discard=False):
        with self.lock:
            ep, self.ep = self.ep, None
        if ep is None:
            return False, 'not recording'
        if discard or not ep['rows']:
            import shutil
            shutil.rmtree(ep['dir'], ignore_errors=True)
            return True, f"{ep['id']} discarded ({len(ep['rows'])} frames)"
        rows = ep['rows']
        arr = {k: np.stack([r[k] for r in rows]) if isinstance(rows[0][k], np.ndarray)
               else np.asarray([r[k] for r in rows]) for k in rows[0]}
        np.savez_compressed(os.path.join(ep['dir'], 'sync.npz'), **arr)
        np.savez_compressed(os.path.join(ep['dir'], 'hires.npz'),
                            imu=np.asarray(ep['imu_hi'], np.float64).reshape(-1, 7),
                            action=np.asarray(ep['act_hi'], np.float64).reshape(-1, 5))
        dur = rows[-1]['t'] - rows[0]['t']
        meta = {'episode_id': ep['id'], 'instruction': ep['instruction'], 'label': label,
                't_start': rows[0]['t'], 't_end': rows[-1]['t'], 'duration_s': round(dur, 3),
                'n_frames': len(rows), 'fps': round(len(rows) / dur, 2) if dur > 0 else 0.0,
                'n_imu': len(ep['imu_hi']), 'n_actions': len(ep['act_hi']),
                'fields': FIELDS, 'car': self.car, 'wall_start': ep['t0'], 'wall_end': time.time()}
        with open(os.path.join(ep['dir'], 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        return True, f"{ep['id']}: {len(rows)} frames, {dur:.1f} s, {label}"

    def status(self):
        with self.lock:
            ep = self.ep
            ages = {k: (time.time() - self.latest[k + '_t']) if self.latest.get(k + '_t') else None
                    for k in ('scan', 'odom', 'imu', 'act')}
        st = {'recording': ep is not None, 'ages': ages}
        if ep is not None:
            st.update({'episode_id': ep['id'], 'instruction': ep['instruction'], 'frames': ep['n'],
                       'secs': round(time.time() - ep['t0'], 1)})
        return st


def list_episodes(root):
    root = os.path.expanduser(root)
    out = []
    if not os.path.isdir(root):
        return out
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, 'meta.json')
        if os.path.isfile(p):
            try:
                m = json.load(open(p))
                out.append({k: m.get(k) for k in ('episode_id', 'instruction', 'label', 'n_frames', 'duration_s')})
            except (OSError, ValueError):
                pass
    return out
