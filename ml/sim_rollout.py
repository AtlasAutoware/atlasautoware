#!/usr/bin/env python3
"""sim_rollout: closed-loop evaluation and on-policy data collection for the student policy.

Everything runs in the ROS-free sim (f1tenth_gym_ros/sim_core.py + goal_core.py), with the
network fed through f1tenth_gym_ros/policy_io.py -- the same code the car's policy_bridge
uses -- so a number measured here is a number about the deployed input pipeline, not about
a training loader.

Modes
  eval    run a policy (ONNX file or 'expert') on a fixed, seeded task set; write per-episode
          results + a summary JSON.
  collect record (observation, expert label) pairs while a mixture of expert and student
          drives (DAgger; beta = probability the expert's action is executed at a tick).
          beta=1 is plain expert demonstration.

    python3 ml/sim_rollout.py eval --policy models/student.onnx --maps levine,Spielberg_map \
        --n 60 --seed 1000 --out runs/eval/baseline
    python3 ml/sim_rollout.py collect --policy expert --beta 1 --maps levine,Spielberg_map,comp_track \
        --n 300 --seed 0 --out data/demos
"""
import argparse, json, math, os, random, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros')); sys.path.insert(0, os.path.join(REPO, 'tools'))
from sim_core import SimMap, bicycle_step, render_fpv, load_map          # noqa: E402
from goal_core import Planner, pure_pursuit, describe_route              # noqa: E402
import policy_io as PIO                                                  # noqa: E402

PARA = [lambda s: s,
        lambda s: s.replace('Go straight', 'Drive straight ahead').replace('go straight', 'keep going straight'),
        lambda s: s.replace('to the end and stop', 'until you reach the goal, then stop'),
        lambda s: 'Please ' + s[0].lower() + s[1:],
        lambda s: s.replace('Turn', 'Take a').replace('turn ', 'take a ').replace('a left', 'left turn').replace('a right', 'right turn'),
        lambda s: s + '. Avoid the walls.']            # identical to tools/sim_generate.py

WHEELBASE, MAX_STEER, BEAMS, CAM = 0.33, 0.4, 540, (640, 480, 460.5)
SCAN_MAX = PIO.BEV_EXTENT + 1.0
GOAL_TOL = 0.5          # m: "reached"
STOP_V = 0.25           # m/s: "stopped" once inside GOAL_TOL
_MAPS = {}


def get_map(name):
    if name not in _MAPS:
        occ, res, origin = load_map(os.path.join(REPO, 'maps', f'{name}.yaml'))
        _MAPS[name] = (SimMap(occ, res, origin), Planner(occ, res, origin))
    return _MAPS[name]


def sample_tasks(maps, n_per_map, seed, min_dist=3.0, max_dist=25.0):
    """Deterministic (start, goal, path, instruction) set. Same seed -> same tasks."""
    tasks = []
    for mi, name in enumerate(maps):
        rng = np.random.default_rng(seed * 1000 + mi); prng = random.Random(seed * 1000 + mi)
        _, pl = get_map(name); k = tries = 0
        while k < n_per_map and tries < n_per_map * 10:
            tries += 1
            s = pl.sample_free(rng, 1)[0]; g = pl.sample_free_near(rng, s, min_dist, max_dist)
            if g is None: continue
            path = pl.plan(s, g)
            if path is None or len(path) < 4: continue
            instr = prng.choice(PARA)(describe_route(path))
            tasks.append({'map': name, 'start': list(s), 'goal': list(g), 'path': [list(p) for p in path],
                          'instruction': instr, 'task_id': f'{name}:{seed}:{k}'})
            k += 1
    return tasks


def path_len(P):
    P = np.asarray(P); return float(np.hypot(*np.diff(P, axis=0).T).sum())


def perturb_obs(front, scan, kind, rng):
    """Sim-side proxies for the sim-to-real gap. Not a substitute for the real car."""
    if kind in ('lidar', 'both'):
        scan = scan + rng.normal(0, 0.03, scan.shape).astype(np.float32)
        scan[rng.random(scan.shape) < 0.19] = 0.0          # C1 on the bench: 581/720 bins valid
    if kind in ('camera', 'both'):
        f = front.astype(np.float32)
        f = f * rng.uniform(0.6, 1.4) + rng.uniform(-40, 40, 3)[None, None, :]
        f += rng.normal(0, 12, f.shape)
        front = np.clip(f, 0, 255).astype(np.uint8)
    if kind == 'nocam':
        front = np.zeros_like(front)
    return front, scan


class OnnxPolicy:
    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
        self.order = PIO.action_order_of(self.sess)
        if os.environ.get('LEGACY_SWAP') == '1':            # reproduce the pre-9/23 policy_bridge bug
            self.order = self.order[::-1]

    def __call__(self, front, bev, state, ids):
        out = self.sess.run(['action'], PIO.make_feed(front, bev, state, ids))[0][0]
        return PIO.split_action(out, self.order)          # (speed, steer)


def rollout(task, policy, beta=0.0, record=False, seed=0, perturb=None, dt=0.02,
            sensor_hz=10.0, max_speed=1.5, time_mult=2.0):
    """One closed-loop episode. policy: OnnxPolicy or None (= expert). Returns (result, data)."""
    sm, _ = get_map(task['map']); path = [tuple(p) for p in task['path']]
    rng = np.random.default_rng(seed); P = np.asarray(path)
    cum = np.concatenate([[0.0], np.hypot(*np.diff(P, axis=0).T).cumsum()]); L = float(cum[-1])
    sx, sy = path[0]; th0 = math.atan2(path[1][1] - sy, path[1][0] - sx)
    st = np.array([sx, sy, th0, 0.0]); ids = PIO.text_ids(task['instruction'])
    angles = -math.pi + 2 * math.pi * np.arange(BEAMS) / BEAMS
    inc = 2 * math.pi / BEAMS
    timeout = time_mult * L / 0.8 + 8.0
    act = (0.0, 0.0); steer_applied = 0.0; t = 0.0; next_tick = 0.0
    rec = {'front': [], 'bev': [], 'state': [], 'act': [], 'scan': []}
    res = {'task_id': task['task_id'], 'map': task['map'], 'route_m': round(L, 2), 'collided': False,
           'reached': False, 'stopped_at_goal': False, 'progress': 0.0, 'time_s': 0.0,
           'steer_err': 0.0, 'speed_err': 0.0, 'ticks': 0, 'student_ticks': 0}
    se = sp = 0.0; n = 0; hold = 0.0
    while t < timeout:
        if t >= next_tick - 1e-9:
            next_tick += 1.0 / sensor_hz
            # 7 m is enough: the policy only sees the scan through the 6 m BEV raster
            scan = sm.raycast(st[0], st[1], st[2] + angles, SCAN_MAX)
            scan = np.clip(scan + rng.normal(0, 0.01, BEAMS), 0, SCAN_MAX).astype(np.float32)
            rgb = render_fpv(sm, st[0], st[1], st[2], CAM[0], CAM[1], CAM[2])
            front = PIO.front_image(np.ascontiguousarray(rgb[:, :, ::-1]))   # the car feeds BGR
            if perturb: front, scan = perturb_obs(front, scan, perturb, rng)
            bev = PIO.bev_image(scan, -math.pi, inc)
            wz = st[3] / WHEELBASE * math.tan(steer_applied)
            state = np.array([st[3], wz, 0.0, 0.0, wz], np.float32)
            ev, es, edone, _ = pure_pursuit(st[:3], path, wheelbase=WHEELBASE)
            if record:
                rec['front'].append(front); rec['bev'].append(bev); rec['state'].append(state)
                rec['scan'].append(scan.astype(np.float16))
                rec['act'].append((ev, es))
            use_expert = policy is None or rng.random() < beta
            if use_expert:
                act = (ev, es)
            else:
                v, s = policy(front, bev, state, ids)
                act = (float(np.clip(v, 0.0, max_speed)), float(np.clip(s, -MAX_STEER, MAX_STEER)))
                se += abs(act[1] - es); sp += abs(act[0] - ev); n += 1
                res['student_ticks'] += 1
            res['ticks'] += 1
        steer_applied = act[1]
        nxt = bicycle_step(st, act[0], act[1], WHEELBASE, dt)
        t += dt
        if sm.occupied(nxt[0], nxt[1]):
            res['collided'] = True; break
        st = nxt
        d = np.hypot(P[:, 0] - st[0], P[:, 1] - st[1]); i = int(np.argmin(d))
        if d[i] < 1.0: res['progress'] = max(res['progress'], float(cum[i] / max(L, 1e-6)))
        dg = math.hypot(P[-1, 0] - st[0], P[-1, 1] - st[1])
        if dg < GOAL_TOL:
            res['reached'] = True; res['progress'] = 1.0
            # keep going until the car has actually stopped and held it for a second: the
            # expert's stop (speed 0 at the goal) must be in the data, or "...and stop" is
            # an instruction the student has never seen demonstrated
            if abs(st[3]) < STOP_V:
                res['stopped_at_goal'] = True; hold += dt
                if hold > 1.0: break
    res['time_s'] = round(t, 2)
    res['success'] = bool(res['reached'] and not res['collided'])
    if n: res['steer_err'] = round(se / n, 4); res['speed_err'] = round(sp / n, 4)
    data = None
    if record and rec['act']:
        data = {'front': np.asarray(rec['front'], np.uint8), 'bev': np.asarray(rec['bev'], np.uint8),
                'state': np.asarray(rec['state'], np.float32), 'act': np.asarray(rec['act'], np.float32),
                'scan': np.asarray(rec['scan'], np.float16),
                'ids': np.repeat(np.asarray(ids, np.int64)[None], len(rec['act']), 0)}
    return res, data


_POLICY = None


def _init(policy_path):
    global _POLICY
    _POLICY = None if policy_path == 'expert' else OnnxPolicy(policy_path)


def _work(args):
    task, beta, record, seed, perturb, out_dir = args
    res, data = rollout(task, _POLICY, beta=beta, record=record, seed=seed, perturb=perturb)
    if data is not None and out_dir:
        fn = os.path.join(out_dir, task['task_id'].replace(':', '_') + f'_s{seed}.npz')
        np.savez(fn, **data)
        res['file'] = os.path.basename(fn); res['frames'] = int(len(data['act']))
    return res


def summarize(results):
    k = len(results) or 1
    f = lambda key: float(np.mean([r[key] for r in results])) if results else 0.0
    ci = lambda key: float(1.96 * np.std([float(r[key]) for r in results]) / math.sqrt(k))
    return {'episodes': len(results), 'success_rate': f('success'), 'success_ci95': ci('success'),
            'collision_rate': f('collided'), 'stopped_at_goal_rate': f('stopped_at_goal'),
            'mean_progress': f('progress'), 'mean_steer_err_rad': f('steer_err'),
            'mean_speed_err_mps': f('speed_err')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['eval', 'collect'])
    ap.add_argument('--policy', default='expert'); ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--maps', default='levine,Spielberg_map'); ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--seed', type=int, default=1000); ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--perturb', default=None, choices=[None, 'lidar', 'camera', 'both', 'nocam'])
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = sample_tasks(a.maps.split(','), a.n, a.seed)
    record = a.mode == 'collect'
    jobs = [(t, a.beta, record, a.seed * 7919 + r * 104729 + i, a.perturb, a.out if record else None)
            for r in range(a.repeats) for i, t in enumerate(tasks)]
    t0 = time.time()
    pol = os.path.abspath(a.policy) if a.policy != 'expert' else 'expert'
    with Pool(a.workers, initializer=_init, initargs=(pol,)) as p:
        results = list(p.imap_unordered(_work, jobs, chunksize=1))
    results.sort(key=lambda r: r['task_id'])
    with open(os.path.join(a.out, 'episodes.jsonl'), 'w') as f:
        for r in results: f.write(json.dumps(r) + '\n')
    summ = summarize(results); summ.update({'policy': a.policy, 'maps': a.maps, 'seed': a.seed, 'beta': a.beta,
                                            'perturb': a.perturb, 'secs': round(time.time() - t0, 1)})
    by_map = {m: summarize([r for r in results if r['map'] == m]) for m in a.maps.split(',')}
    summ['by_map'] = by_map
    json.dump(summ, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != 'by_map'}))
    for m, s in by_map.items():
        print(f"  {m:16s} success {s['success_rate']:.2f}  collide {s['collision_rate']:.2f}  progress {s['mean_progress']:.2f}")


if __name__ == '__main__':
    main()
