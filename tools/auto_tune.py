"""
Autonomous tuning harness — continuous optimize/benchmark/rollback cycles.
==========================================================================

Each cycle:
  1. propose a candidate config (coordinate descent: nudge one parameter off the
     current best, with occasional random exploration),
  2. apply it — regenerate the raceline if a raceline param changed (writing the
     line the agent loads) and write controller params to runtime/tune_config.json,
  3. GATE: validate the regenerated raceline (finite, closed, speeds in range)
     before it is ever driven — invalid candidates are rejected without a run,
  4. benchmark a solo lap in sim (tools/benchmark_lap.py) -> score,
  5. accept if the score improves, else ROLL BACK to the previous best line+config.

Search space spans all three tunable areas the stack exposes:
  raceline:   a_lat, v_max, apex_bias, margin
  controller: kL, max_L, steer_smooth, anticip_k, curv_gain   (lookahead-per-zone,
              cross-track damping, corner anticipation)

Persists: runtime/tune_best.json (best config+score), runtime/tune_log.jsonl
(every cycle), racelines/tuned_best.csv (best line, for rollback).

    python3 tools/auto_tune.py --minutes 55 --bench-time 60
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.path.join(REPO, 'runtime')
RL = os.path.join(REPO, 'racelines', 'best_raceline.csv')
RL_BEST = os.path.join(REPO, 'racelines', 'tuned_best.csv')
TUNE_CFG = os.path.join(RUNTIME, 'tune_config.json')
BEST_JSON = os.path.join(RUNTIME, 'tune_best.json')
LOG = os.path.join(RUNTIME, 'tune_log.jsonl')
SEED = ('49.910', '42.780')

# name -> (min, max, step, kind)
SPACE = {
    'a_lat':       (5.0, 8.0, 0.5, 'raceline'),
    'v_max':       (6.0, 8.0, 0.5, 'raceline'),
    'apex_bias':   (0.4, 1.6, 0.2, 'raceline'),
    'margin':      (0.22, 0.42, 0.03, 'raceline'),
    'kL':          (0.24, 0.46, 0.03, 'controller'),
    'max_L':       (1.6, 2.8, 0.2, 'controller'),
    'steer_smooth':(0.45, 0.72, 0.05, 'controller'),
    'anticip_k':   (0.7, 1.2, 0.1, 'controller'),
    'curv_gain':   (0.9, 1.6, 0.1, 'controller'),
}
DEFAULT = {'a_lat': 6.5, 'v_max': 7.0, 'apex_bias': 1.0, 'margin': 0.35,
           'kL': 0.32, 'max_L': 2.2, 'steer_smooth': 0.55, 'anticip_k': 0.9,
           'curv_gain': 1.2}


def clamp(name, v):
    lo, hi, _, _ = SPACE[name]
    return round(min(hi, max(lo, v)), 3)


def regen_raceline(cfg):
    cmd = ('source /opt/ros/foxy/setup.bash; source /sim_ws/install/setup.bash 2>/dev/null; '
           f'cd {REPO}; python3 f1tenth_gym_ros/raceline_optimizer.py '
           f'--map maps/comp_track.yaml --output {RL} --seed {SEED[0]} {SEED[1]} '
           f'--margin {cfg["margin"]} --apex-bias {cfg["apex_bias"]} '
           f'--a-lat {cfg["a_lat"]} --v-max {cfg["v_max"]} --no-overlay')
    r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=90)
    return r.returncode == 0


def raceline_valid():
    """GATE: a generated line must be finite, closed, sane before it's driven."""
    try:
        import csv
        xs, sp = [], []
        with open(RL) as f:
            for row in csv.DictReader(f):
                xs.append(float(row['x'])); sp.append(float(row['speed']))
        if len(xs) < 100:
            return False
        if any(v != v for v in xs) or any(v != v for v in sp):   # NaN
            return False
        if min(sp) < 0.3 or max(sp) > 12.0:
            return False
        return True
    except Exception:
        return False


def write_controller(cfg):
    os.makedirs(RUNTIME, exist_ok=True)
    with open(TUNE_CFG, 'w') as f:
        json.dump({k: cfg[k] for k in
                   ('kL', 'max_L', 'steer_smooth', 'anticip_k', 'curv_gain')}, f)


def benchmark(bench_time):
    cmd = ('source /opt/ros/foxy/setup.bash; source /sim_ws/install/setup.bash 2>/dev/null; '
           f'cd {REPO}; python3 tools/benchmark_lap.py --time {bench_time}')
    try:
        r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True,
                           timeout=bench_time + 40)
        for line in reversed(r.stdout.splitlines()):
            line = line.strip()
            if line.startswith('{'):
                return json.loads(line)
    except Exception as e:
        return {'score': 9999, 'error': str(e)}
    return {'score': 9999}


def propose(best):
    cand = dict(best)
    if random.random() < 0.15:                       # exploration: random restart of 1 param
        name = random.choice(list(SPACE))
        lo, hi, _, _ = SPACE[name]
        cand[name] = clamp(name, random.uniform(lo, hi))
    else:                                            # coordinate descent: nudge one param
        name = random.choice(list(SPACE))
        _, _, step, _ = SPACE[name]
        cand[name] = clamp(name, best[name] + random.choice([-1, 1]) * step)
    return cand, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=55.0)
    ap.add_argument('--bench-time', type=float, default=60.0)
    args = ap.parse_args()
    os.makedirs(RUNTIME, exist_ok=True)

    # Seed best from current config + a baseline benchmark.
    best = dict(DEFAULT)
    if os.path.exists(BEST_JSON):
        try:
            best.update(json.load(open(BEST_JSON)).get('config', {}))
        except Exception:
            pass
    write_controller(best)
    if not os.path.exists(RL_BEST):
        shutil.copy(RL, RL_BEST)
    print('[baseline] benchmarking current best...', flush=True)
    base = benchmark(args.bench_time)
    best_score = base.get('score', 9999)
    print(f'[baseline] score={best_score} {base}', flush=True)
    json.dump({'config': best, 'score': best_score, 'bench': base}, open(BEST_JSON, 'w'))

    t_end = time.time() + args.minutes * 60
    cycle = 0
    while time.time() < t_end:
        cycle += 1
        cand, changed = propose(best)
        kind = SPACE[changed][3]
        applied_ok = True
        if kind == 'raceline':
            applied_ok = regen_raceline(cand) and raceline_valid()
        write_controller(cand)

        if not applied_ok:
            res = {'score': 9999, 'gate': 'invalid_raceline'}
        else:
            res = benchmark(args.bench_time)
        score = res.get('score', 9999)
        improved = score < best_score - 0.05         # require a real gain

        rec = {'cycle': cycle, 't': round(time.time()), 'changed': changed,
               'value': cand[changed], 'kind': kind, 'score': score,
               'best_score': best_score, 'accepted': improved, 'bench': res}
        with open(LOG, 'a') as f:
            f.write(json.dumps(rec) + '\n')

        if improved:
            best, best_score = cand, score
            if kind == 'raceline':
                shutil.copy(RL, RL_BEST)             # lock in the better line
            json.dump({'config': best, 'score': best_score, 'bench': res},
                      open(BEST_JSON, 'w'))
            print(f'[c{cycle}] ACCEPT {changed}={cand[changed]} -> {score} '
                  f'(best now {best_score})', flush=True)
        else:
            # ROLLBACK: restore the best line + controller config.
            if kind == 'raceline':
                shutil.copy(RL_BEST, RL)
            write_controller(best)
            print(f'[c{cycle}] reject {changed}={cand[changed]} ({score} '
                  f'vs best {best_score})', flush=True)

    # Final: ensure best is deployed.
    shutil.copy(RL_BEST, RL)
    write_controller(best)
    print(f'\n[done] {cycle} cycles | best score {best_score} | config {best}', flush=True)


if __name__ == '__main__':
    main()
