#!/usr/bin/env python3
"""Run a route-hint model in the sim with the hint built the way the car builds it.

Standard eval (sim_rollout) hands the policy a hint from the task's pre-planned path.
On the car policy_bridge only has a goal point, a noisy map-frame pose and the map, and
uses route_source.RouteSource: A* from the current pose, replanned every `replan` s,
zero speed when there is no route, stop inside goal_tol. This script does exactly that in
the sim so the car-side code is exercised closed-loop before the car is on the floor.

    python ml/car_route_check.py --policy runs/route/student.onnx --out /tmp/carcheck \
        --pose-noise 0.05 --yaw-noise 0.03 --pose-lag 0.1
The map is passed through an OccupancyGrid round trip (occ_from_grid) so the row flip
the bridge does on /map is tested too.
"""
import argparse, json, math, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sim_rollout as SR                                     # noqa: E402  (sets sys.path)
from route_source import RouteSource, occ_from_grid          # noqa: E402
from sim_core import load_map                                # noqa: E402


class CarHint:
    def __init__(self, task, a, seed):
        occ, res, origin = load_map(os.path.join(SR.REPO, 'maps', f"{task['map']}.yaml"))
        msg = (np.flipud(occ).astype(np.int8) * 100).ravel()          # what /map carries
        self.rs = RouteSource(goal_tol=a.goal_tol)
        self.rs.set_map(occ_from_grid(msg, occ.shape[1], occ.shape[0]), res, origin)
        self.rs.set_goal(task['goal'])
        self.a = a; self.rng = np.random.default_rng(seed + 17); self.hist = []; self.t_plan = -1e9
        self.plans = 0; self.fails = 0; self.no_route_ticks = 0; self.max_ms = 0.0

    def __call__(self, st, t):
        self.hist.append((t, st[:3].copy()))
        while len(self.hist) > 1 and self.hist[1][0] <= t - self.a.pose_lag: self.hist.pop(0)
        x, y, th = self.hist[0][1]                                    # pose from pose_lag ago
        pose = (x + self.rng.normal(0, self.a.pose_noise), y + self.rng.normal(0, self.a.pose_noise),
                th + self.rng.normal(0, self.a.yaw_noise))
        if t - self.t_plan >= self.a.replan:
            self.t_plan = t; self.plans += 1
            if not self.rs.replan(pose, t): self.fails += 1
            self.max_ms = max(self.max_ms, self.rs.plan_ms)   # first call includes building the planner
        h, st_ = self.rs.hint(pose, t)
        if h is None and st_['route'] != 'arrived': self.no_route_ticks += 1
        return h


A = None


def _work(args):
    task, seed = args
    ch = CarHint(task, A, seed)
    res, _ = SR.rollout(task, SR._POLICY, seed=seed, hint_fn=ch)
    res.update({'plans': ch.plans, 'plan_fails': ch.fails, 'no_route_ticks': ch.no_route_ticks,
                'plan_ms': round(ch.max_ms, 1)})
    return res


def _init(pol, a):
    global A
    A = a; SR._init(pol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--maps', default='levine,Spielberg_map,comp_track,my_track,uploadtest')
    ap.add_argument('--n', type=int, default=10); ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--replan', type=float, default=1.0); ap.add_argument('--goal-tol', type=float, default=0.4)
    ap.add_argument('--pose-noise', type=float, default=0.0); ap.add_argument('--yaw-noise', type=float, default=0.0)
    ap.add_argument('--pose-lag', type=float, default=0.0); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    tasks = SR.sample_tasks(a.maps.split(','), a.n, a.seed)
    jobs = [(t, a.seed * 7919 + i) for i, t in enumerate(tasks)]      # same episode seeds as sim_rollout eval
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(os.path.abspath(a.policy), a)) as p:
        results = sorted(p.imap_unordered(_work, jobs, chunksize=1), key=lambda r: r['task_id'])
    with open(os.path.join(a.out, 'episodes.jsonl'), 'w') as f:
        for r in results: f.write(json.dumps(r) + '\n')
    s = SR.summarize(results)
    s.update({k: v for k, v in vars(a).items() if k not in ('out', 'workers')})
    s['by_map'] = {m: SR.summarize([r for r in results if r['map'] == m]) for m in a.maps.split(',')}
    s['plan_fail_rate'] = float(np.sum([r['plan_fails'] for r in results]) / max(1, np.sum([r['plans'] for r in results])))
    s['max_plan_ms'] = float(max(r['plan_ms'] for r in results)); s['secs'] = round(time.time() - t0, 1)
    json.dump(s, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print(json.dumps({k: s[k] for k in ('success_rate', 'collision_rate', 'stopped_at_goal_rate', 'mean_progress',
                                        'plan_fail_rate', 'max_plan_ms', 'secs')}))
    for m, v in s['by_map'].items():
        print(f"  {m:16s} success {v['success_rate']:.2f}  collide {v['collision_rate']:.2f}  progress {v['mean_progress']:.2f}")


if __name__ == '__main__':
    main()
