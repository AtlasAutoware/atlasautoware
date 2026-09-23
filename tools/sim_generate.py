#!/usr/bin/env python3
"""sim_generate: headless, ROS-free generation of goal-conditioned demonstration episodes.

Runs the whole loop in-process on any machine with numpy/scipy/PIL/OpenCV (no ROS, no
Jetson): sample (start, goal) on a map -> A* route -> language instruction -> pure-pursuit
expert drives the bicycle model at 50 Hz -> lidar ray-cast and camera render at 10 Hz ->
episode written in exactly the car's layout (frames/*.jpg, sync.npz, hires.npz, meta.json).
Faster than real time, so hundreds of episodes take minutes.

    python3 tools/sim_generate.py --maps levine,Spielberg_map --per-map 100 --out ~/sim_episodes
    python3 tools/episodes_to_lerobot.py ~/sim_episodes --out ~/lerobot/atlascar_sim --only good
"""
import argparse, json, math, os, random, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
from sim_core import SimMap, bicycle_step, render_fpv, load_map      # noqa: E402
from goal_core import Planner, pure_pursuit, describe_route         # noqa: E402
from episode_writer import EpisodeWriter                            # noqa: E402

PARA = [lambda s: s,
        lambda s: s.replace('Go straight', 'Drive straight ahead').replace('go straight', 'keep going straight'),
        lambda s: s.replace('to the end and stop', 'until you reach the goal, then stop'),
        lambda s: 'Please ' + s[0].lower() + s[1:],
        lambda s: s.replace('Turn', 'Take a').replace('turn ', 'take a ').replace('a left', 'left turn').replace('a right', 'right turn'),
        lambda s: s + '. Avoid the walls.']


def run_episode(w, sm, pl, path, instr, rng, dt=0.02, sensor_hz=10.0, timeout=45.0, beams=540,
                cam=(640, 480, 460.5), noise=0.01, wheelbase=0.33):
    sx, sy = path[0]; th0 = math.atan2(path[1][1] - sy, path[1][0] - sx)
    st = np.array([sx, sy, th0, 0.0]); t = 1000.0 + rng.random() * 100
    angles = -math.pi + 2 * math.pi * np.arange(beams) / beams
    next_sensor = t; coll = 0; done = False; dist = 0.0; prev_v = 0.0
    ok, eid = w.start(instr)
    steps = int(timeout / dt)
    for k in range(steps):
        v_cmd, steer, done, dg = pure_pursuit(st[:3], path, wheelbase=wheelbase)
        if done: break
        nxt = bicycle_step(st, v_cmd, steer, wheelbase, dt)
        if sm.occupied(nxt[0], nxt[1]):
            coll += 1; nxt[0], nxt[1], nxt[3] = st[0], st[1], 0.0
        dist += math.hypot(nxt[0] - st[0], nxt[1] - st[1]); st = nxt; t += dt
        # streams the real car would log
        w.set_action(v_cmd, steer, 1, t)                              # expert command (src=1 policy)
        ax = (st[3] - prev_v) / dt; prev_v = st[3]
        w.update('imu', (ax, 0.0, 9.81, 0.0, 0.0, st[3] / wheelbase * math.tan(steer)), t)
        w.update('pose', (st[0], st[1], st[2])); w.update('vel', (st[3], st[3] / wheelbase * math.tan(steer))); w.stamp('odom', t)
        if t >= next_sensor:
            next_sensor += 1.0 / sensor_hz
            r = sm.raycast(st[0], st[1], st[2] + angles, 16.0) + rng.normal(0, noise, beams).astype(np.float32)
            w.update('scan', np.clip(r, 0, 16.0).astype(np.float32), t)
            img = render_fpv(sm, st[0], st[1], st[2], cam[0], cam[1], cam[2])
            w.add_frame(img[:, :, ::-1].copy(), t)                     # writer expects BGR (cv2)
        if coll > 30: break                                           # stuck against a wall
    label = 'good' if done and coll == 0 else 'bad'
    ok, msg = w.stop(label)
    return label, done, coll, dist, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', default='levine')
    ap.add_argument('--per-map', type=int, default=50)
    ap.add_argument('--out', default=os.path.expanduser('~/sim_episodes'))
    ap.add_argument('--min-dist', type=float, default=3.0); ap.add_argument('--max-dist', type=float, default=25.0)
    ap.add_argument('--timeout', type=float, default=45.0)
    ap.add_argument('--cam-w', type=int, default=640); ap.add_argument('--cam-h', type=int, default=480)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); random.seed(a.seed)
    w = EpisodeWriter(a.out, {'wheelbase': 0.33, 'max_steer': 0.4, 'sim': True, 'camera': 'raycast render'})
    good = bad = 0; manifest = []; t0 = time.time()
    for map_name in a.maps.split(','):
        yaml_p = os.path.join(REPO, 'maps', f'{map_name}.yaml')
        occ, res, origin = load_map(yaml_p)
        sm = SimMap(occ, res, origin); pl = Planner(occ, res, origin)
        k = attempts = 0
        while k < a.per_map and attempts < a.per_map * 8:
            attempts += 1
            (sx, sy) = pl.sample_free(rng, 1)[0]
            g = pl.sample_free_near(rng, (sx, sy), a.min_dist, a.max_dist)
            if g is None: continue
            tx, ty = g; d = math.hypot(tx - sx, ty - sy)
            path = pl.plan((sx, sy), (tx, ty))
            if path is None or len(path) < 4: continue
            instr = random.choice(PARA)(describe_route(path))
            label, done, coll, dist, msg = run_episode(w, sm, pl, path, instr, rng, timeout=a.timeout,
                                                       cam=(a.cam_w, a.cam_h, 460.5 * a.cam_w / 640))
            good += label == 'good'; bad += label == 'bad'; k += 1
            manifest.append({'map': map_name, 'start': [sx, sy], 'goal': [tx, ty], 'route_m': round(d, 1),
                             'instruction': instr, 'label': label, 'reached': done, 'collisions': coll,
                             'driven_m': round(dist, 1), 'episode': msg.split(':')[0]})
            print(f'[{map_name} {k:3d}/{a.per_map}] {label:4s} {dist:5.1f} m  "{instr}"', flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'manifest.jsonl'), 'a') as f:
        for m in manifest: f.write(json.dumps(m) + '\n')
    print(f'done in {time.time()-t0:.0f} s: {good} good, {bad} bad -> {a.out}')


if __name__ == '__main__':
    main()
