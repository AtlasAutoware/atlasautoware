#!/usr/bin/env python3
"""Repackage raw car episodes (episode_writer layout) into a LeRobot v2.1 dataset.

    python3 tools/episodes_to_lerobot.py ~/episodes  --out ~/lerobot/atlascar_hallway \
        [--only good] [--fps 10] [--bev 96]

LeRobot's on-disk format is what OpenVLA/π0/SmolVLA/GR00T fine-tuning scripts and
LeRobot's own trainers read, so producing it once here means every candidate policy
in the plan can consume the same data. The lidar has no standard slot in that
format; it is rasterised to a small bird's-eye occupancy image (`observation.images.bev`)
so a pretrained vision tower can consume it unchanged, and the raw ranges are kept in
`observation.lidar` for anyone who wants a learned encoder later.

Output:
    <out>/meta/info.json, episodes.jsonl, tasks.jsonl, stats.json (minimal)
    <out>/data/chunk-000/episode_XXXXXX.parquet
    <out>/videos/chunk-000/observation.images.front/episode_XXXXXX.mp4
    <out>/videos/chunk-000/observation.images.bev/episode_XXXXXX.mp4
Needs pandas + pyarrow + opencv on the laptop (not on the car).
"""
import argparse, glob, json, os, sys
import numpy as np

try:
    import cv2, pandas as pd
except ImportError as e:
    sys.exit(f'needs pandas, pyarrow, opencv: {e}')


def bev_image(scan, angle_min, angle_inc, size=96, extent=6.0):
    """Rasterise a lidar scan to a size×size occupancy image, car at the centre, x up."""
    img = np.zeros((size, size), np.uint8)
    n = len(scan)
    ang = angle_min + angle_inc * np.arange(n)
    ok = (scan > 0.05) & (scan < extent)
    x, y = scan[ok] * np.cos(ang[ok]), scan[ok] * np.sin(ang[ok])
    px = (size / 2 - x / extent * size / 2).astype(int)     # forward = up
    py = (size / 2 - y / extent * size / 2).astype(int)     # left = left
    m = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    img[px[m], py[m]] = 255
    img[size // 2 - 1:size // 2 + 2, size // 2 - 1:size // 2 + 2] = 128     # the car
    return img


def write_video(frames, path, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        vw.write(f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('--out', required=True)
    ap.add_argument('--only', default=None, help='keep episodes with this label (e.g. good)')
    ap.add_argument('--fps', type=float, default=None, help='override fps (default: from meta)')
    ap.add_argument('--bev', type=int, default=96)
    ap.add_argument('--angle-min', type=float, default=-np.pi)
    ap.add_argument('--angle-inc', type=float, default=None, help='default 2π/len(scan)')
    a = ap.parse_args()

    eps = sorted(d for d in glob.glob(os.path.join(os.path.expanduser(a.root), '*'))
                 if os.path.isfile(os.path.join(d, 'meta.json')))
    out = os.path.expanduser(a.out)
    for sub in ('meta', 'data/chunk-000', 'videos/chunk-000/observation.images.front',
                'videos/chunk-000/observation.images.bev'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    tasks, task_ids, ep_meta, total = {}, {}, [], 0
    fps_out = None
    for k, d in enumerate(eps):
        meta = json.load(open(os.path.join(d, 'meta.json')))
        if a.only and meta.get('label') != a.only:
            continue
        S = np.load(os.path.join(d, 'sync.npz'))
        n = len(S['t'])
        if n < 5:
            continue
        fps = a.fps or (meta.get('fps') or 10.0)
        fps_out = fps_out or fps
        instr = meta.get('instruction', '') or 'drive'
        if instr not in task_ids:
            task_ids[instr] = len(task_ids); tasks[task_ids[instr]] = instr
        ei = len(ep_meta)
        front = [cv2.imread(os.path.join(d, 'frames', f'{i:06d}.jpg')) for i in range(n)]
        front = [f for f in front if f is not None]
        n = min(n, len(front))
        scans = S['scan'][:n]
        inc = a.angle_inc or (2 * np.pi / scans.shape[1])
        bev = [bev_image(s, a.angle_min, inc, a.bev) for s in scans]
        write_video(front[:n], os.path.join(out, f'videos/chunk-000/observation.images.front/episode_{ei:06d}.mp4'), fps)
        write_video(bev, os.path.join(out, f'videos/chunk-000/observation.images.bev/episode_{ei:06d}.mp4'), fps)
        act = S['act'][:n].astype(np.float32)
        vel = S['vel'][:n].astype(np.float32); imu = S['imu'][:n].astype(np.float32)
        df = pd.DataFrame({
            'observation.state': list(np.concatenate([vel, imu[:, 3:6]], 1)),   # vx wz gx gy gz
            'observation.lidar': list(scans.astype(np.float32)),
            'action': list(act),                                                  # speed, steer
            'action.source': S['act_src'][:n].astype(np.int8),
            'timestamp': (S['t'][:n] - S['t'][0]).astype(np.float32),
            'frame_index': np.arange(n, dtype=np.int64),
            'episode_index': np.full(n, ei, np.int64),
            'index': np.arange(total, total + n, dtype=np.int64),
            'task_index': np.full(n, task_ids[instr], np.int64),
        })
        df.to_parquet(os.path.join(out, f'data/chunk-000/episode_{ei:06d}.parquet'))
        ep_meta.append({'episode_index': ei, 'tasks': [instr], 'length': int(n),
                        'source': meta.get('episode_id'), 'label': meta.get('label')})
        total += n
        print(f'  episode {ei:3d}  {n:4d} frames  {instr!r}')

    if not ep_meta:
        sys.exit('no episodes converted')
    h, w = front[0].shape[:2]
    info = {
        'codebase_version': 'v2.1', 'robot_type': 'atlascar_f1tenth', 'fps': fps_out,
        'total_episodes': len(ep_meta), 'total_frames': total, 'total_tasks': len(tasks),
        'total_chunks': 1, 'chunks_size': 1000,
        'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
        'features': {
            'observation.images.front': {'dtype': 'video', 'shape': [h, w, 3], 'names': ['height', 'width', 'channel'],
                                         'info': {'video.fps': fps_out, 'video.codec': 'mp4v'}},
            'observation.images.bev': {'dtype': 'video', 'shape': [a.bev, a.bev, 3], 'names': ['height', 'width', 'channel'],
                                       'info': {'video.fps': fps_out, 'video.codec': 'mp4v', 'extent_m': 6.0}},
            'observation.state': {'dtype': 'float32', 'shape': [5], 'names': ['vx', 'wz', 'gx', 'gy', 'gz']},
            'observation.lidar': {'dtype': 'float32', 'shape': [int(scans.shape[1])], 'names': None},
            'action': {'dtype': 'float32', 'shape': [2], 'names': ['speed', 'steer']},
            'action.source': {'dtype': 'int8', 'shape': [1], 'names': None},
            'timestamp': {'dtype': 'float32', 'shape': [1], 'names': None},
            'frame_index': {'dtype': 'int64', 'shape': [1], 'names': None},
            'episode_index': {'dtype': 'int64', 'shape': [1], 'names': None},
            'index': {'dtype': 'int64', 'shape': [1], 'names': None},
            'task_index': {'dtype': 'int64', 'shape': [1], 'names': None},
        },
    }
    json.dump(info, open(os.path.join(out, 'meta/info.json'), 'w'), indent=1)
    with open(os.path.join(out, 'meta/episodes.jsonl'), 'w') as f:
        for e in ep_meta: f.write(json.dumps(e) + '\n')
    with open(os.path.join(out, 'meta/tasks.jsonl'), 'w') as f:
        for i, t in tasks.items(): f.write(json.dumps({'task_index': i, 'task': t}) + '\n')
    print(f'wrote {out}: {len(ep_meta)} episodes, {total} frames, {len(tasks)} tasks')


if __name__ == '__main__':
    main()
