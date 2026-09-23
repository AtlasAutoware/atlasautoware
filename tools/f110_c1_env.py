#!/usr/bin/env python3
"""f110_c1_env: f1tenth_gym (f110_gym 0.2.1) configured for this car's RPLidar C1 (milestone T8).

The stock gym simulates a Hokuyo: 1080 beams over 270 deg every physics step. The C1 on the
car delivers 0.72 deg bins at 10 Hz, so over the same 270 deg front arc that is 375 beams, and
a policy should only see a new scan every 0.1 s. This module builds the env with that scan and
wraps it so observations are held between 10 Hz lidar frames.

Setup (the gym pins gym==0.19 / numpy<=1.22, so it needs Python 3.10):
    uv venv -p 3.10 f110_venv
    uv pip install -p f110_venv/bin/python "pip<24.1" setuptools==65.5.0 wheel==0.38.4
    f110_venv/bin/python -m pip install --no-build-isolation gym==0.19.0
    f110_venv/bin/python -m pip install --no-build-isolation -e <f1tenth_gym checkout>
    f110_venv/bin/python tools/f110_c1_env.py --map <f1tenth_gym>/gym/f110_gym/envs/../../../examples/example_map
"""
import argparse, math, os, time
import numpy as np

C1_RES_DEG = 0.72
C1_FOV = math.radians(270.0)
C1_BEAMS = int(round(270.0 / C1_RES_DEG))        # 375
C1_HZ = 10.0


def make_env(map_path, map_ext='.png', timestep=0.01, num_agents=1):
    import gym
    from f110_gym.envs import base_classes as bc
    # f110_gym builds every RaceCar with num_beams=1080, fov=4.7 and does not expose either;
    # patch the defaults before the env constructs its cars.
    orig = bc.RaceCar.__init__

    def init(self, *args, **kw):
        kw['num_beams'] = C1_BEAMS; kw['fov'] = C1_FOV
        orig(self, *args, **kw)
    bc.RaceCar.__init__ = init
    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext=map_ext, num_agents=num_agents, timestep=timestep)
    return env


class HoldScan:
    """Hold the scan between 10 Hz lidar frames (the physics runs at 1/timestep Hz)."""
    def __init__(self, env, timestep=0.01, hz=C1_HZ):
        self.env, self.every = env, max(1, int(round(1.0 / (hz * timestep)))); self.k = 0; self.scan = None

    def reset(self, poses):
        obs, r, d, info = self.env.reset(poses); self.k = 0; self.scan = obs['scans'][0].copy()
        return obs, r, d, info

    def step(self, action):
        obs, r, d, info = self.env.step(action); self.k += 1
        if self.k % self.every == 0: self.scan = obs['scans'][0].copy()
        obs['scans'] = [self.scan]
        return obs, r, d, info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--map', required=True); ap.add_argument('--ext', default='.png')
    ap.add_argument('--steps', type=int, default=500)
    a = ap.parse_args()
    env = HoldScan(make_env(a.map, a.ext))
    pose = [0.0, 0.0, 0.0]
    cfg = os.path.join(os.path.dirname(os.path.abspath(a.map)), 'config_' + os.path.basename(a.map) + '.yaml')
    if os.path.isfile(cfg):                       # the gym's own example start pose
        import yaml; c = yaml.safe_load(open(cfg)); pose = [c['sx'], c['sy'], c['stheta']]
    obs, _, _, _ = env.reset(np.array([pose]))
    n = len(obs['scans'][0]); changes = 0; last = obs['scans'][0].copy(); t0 = time.time(); done = False
    steps = 0
    for _ in range(a.steps):
        steps += 1
        # simple gap-follow on the C1 scan: steer toward the farthest beam in the front arc
        s = obs['scans'][0]; ang = -C1_FOV / 2 + C1_FOV * np.arange(n) / (n - 1)
        i = int(np.argmax(np.convolve(s, np.ones(15) / 15, mode='same')))
        obs, _, done, _ = env.step(np.array([[float(np.clip(ang[i], -0.4, 0.4)), 1.5]]))
        if not np.array_equal(obs['scans'][0], last): changes += 1; last = obs['scans'][0].copy()
        if done: break
    print(f'beams={n} fov_deg={math.degrees(C1_FOV):.0f} res_deg={math.degrees(C1_FOV) / (n - 1):.3f} '
          f'scan_updates={changes} over {steps} steps of 10 ms (expect ~{steps // 10}) '
          f'collision={bool(done)} lap_time={obs["lap_times"][0]:.1f}s wall={time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
