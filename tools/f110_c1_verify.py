#!/usr/bin/env python3
"""T8 check: the stock waypoint_follow.py pure-pursuit lap, headless, on the C1-configured gym.

    cd <f1tenth_gym>/examples && <f110_venv>/bin/python <repo>/tools/f110_c1_verify.py
"""
import os, sys, time, types
import numpy as np, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.modules.setdefault('pyglet.gl', types.SimpleNamespace(GL_POINTS=0))   # only used for rendering
sys.path.insert(0, os.getcwd())
from waypoint_follow import PurePursuitPlanner                              # noqa: E402
from f110_c1_env import make_env, HoldScan                                   # noqa: E402

conf = types.SimpleNamespace(**yaml.safe_load(open('config_example_map.yaml')))
work = {'tlad': 0.82461887897713965, 'vgain': 1.375}
planner = PurePursuitPlanner(conf, 0.17145 + 0.15875)
env = HoldScan(make_env(conf.map_path, conf.map_ext))
obs, r, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
lap, t0, n = 0.0, time.time(), 0
while not done and n < 20000:
    speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
    obs, r, done, info = env.step(np.array([[steer, speed]])); lap += r; n += 1
s = obs['scans'][0]
print(f"scan: {len(s)} beams (C1: 375 over 270 deg); laps completed: {int(obs['lap_counts'][0])}; "
      f"collision: {bool(obs['collisions'][0])}; sim time {lap:.2f} s; wall {time.time() - t0:.1f} s")
