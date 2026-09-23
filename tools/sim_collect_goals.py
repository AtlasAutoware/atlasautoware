#!/usr/bin/env python3
"""sim_collect_goals: generate GOAL-CONDITIONED demonstration episodes in the simulator.

Not laps. Each episode is a random (start, goal) on a map: the collector plans a route
(A*), turns the route into a language instruction, resets the sim car at the start, sets
the goal for the success metric, hands the route to goal_expert, and records with the
episode logger until the goal is reached or time runs out. Labels: good = reached with no
collision. Output is the car's episode layout -> tools/episodes_to_lerobot.py.

    ROS_DOMAIN_ID=7 python3 tools/sim_collect_goals.py --maps levine,Spielberg_map --per-map 30

Instruction variety comes from the route geometry (turns, straights) plus paraphrases,
so a language-conditioned policy sees many phrasings of many routes.
"""
import argparse, json, math, os, random, subprocess, sys, time
import numpy as np

REPO = os.path.expanduser('~/atlas_ws/src/atlasautoware')
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
from goal_core import Planner, describe_route                   # noqa: E402
from sim_core import load_map                                   # noqa: E402

PARA = [lambda s: s,
        lambda s: s.replace('Go straight', 'Drive straight ahead').replace('go straight', 'keep going straight'),
        lambda s: s.replace('to the end and stop', 'until you reach the goal, then stop'),
        lambda s: 'Please ' + s[0].lower() + s[1:],
        lambda s: s.replace('Turn', 'Take a').replace('turn ', 'take a ').replace('a left', 'left turn').replace('a right', 'right turn')]


def sh(cmd):
    return subprocess.Popen(cmd, shell=True, executable='/bin/bash')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', default='levine')
    ap.add_argument('--per-map', type=int, default=20)
    ap.add_argument('--timeout', type=float, default=45.0)
    ap.add_argument('--min-dist', type=float, default=3.0)
    ap.add_argument('--max-dist', type=float, default=25.0)
    ap.add_argument('--root', default=os.path.expanduser('~/sim_episodes'))
    ap.add_argument('--camera', default='true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); random.seed(a.seed)
    os.environ.setdefault('ROS_DOMAIN_ID', '7')

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped
    rclpy.init(); n = Node('sim_collect_goals')
    reset_pub = n.create_publisher(String, '/sim/reset', 5)
    goal_pub = n.create_publisher(String, '/sim/goal', 5)
    ep_pub = n.create_publisher(String, '/episode/cmd', 5)
    path_pub = n.create_publisher(Path, '/goal_expert/path', 5)
    state, gx = {}, {}
    n.create_subscription(String, '/sim/state', lambda m: state.update(json.loads(m.data)), 5)
    n.create_subscription(String, '/goal_expert/status', lambda m: gx.update(json.loads(m.data)), 5)

    def spin(sec):
        t = time.time()
        while time.time() - t < sec: rclpy.spin_once(n, timeout_sec=0.05)

    src = 'source /opt/ros/humble/setup.bash; source ~/atlas_ws/install/setup.bash; '
    logger = sh(src + f'ros2 run f1tenth_gym_ros episode_logger --ros-args -p root:={a.root} '
                      f'-p image_topic:=/camera/color/image_raw -p odom_topic:=/odom -p imu_topic:=/none '
                      f'> /tmp/sim_logger.log 2>&1')
    expert = sh(src + 'ros2 run f1tenth_gym_ros goal_expert --ros-args -p odom_topic:=/pf/pose/odom > /tmp/goal_expert.log 2>&1')
    good = bad = 0; manifest = []
    try:
        for map_name in a.maps.split(','):
            yaml_p = os.path.join(REPO, 'maps', f'{map_name}.yaml')
            occ, res, origin = load_map(yaml_p)
            pl = Planner(occ, res, origin)
            sx0, sy0 = pl.sample_free(rng, 1)[0]
            sim = sh(src + f'ros2 run f1tenth_gym_ros sim_env --ros-args -p map_yaml:={yaml_p} '
                           f'-p start_x:={sx0} -p start_y:={sy0} -p camera:={a.camera} > /tmp/sim_env.log 2>&1')
            spin(4.0)
            k = 0; attempts = 0
            while k < a.per_map and attempts < a.per_map * 6:
                attempts += 1
                (sx, sy), (tx, ty) = pl.sample_free(rng, 2)
                d = math.hypot(tx - sx, ty - sy)
                if not (a.min_dist <= d <= a.max_dist): continue
                path = pl.plan((sx, sy), (tx, ty))
                if path is None or len(path) < 4: continue
                instr = random.choice(PARA)(describe_route(path))
                th0 = math.atan2(path[1][1] - sy, path[1][0] - sx)
                # place the car, set the goal metric, hand the route to the expert
                reset_pub.publish(String(data=json.dumps({'x': sx, 'y': sy, 'theta': th0})))
                goal_pub.publish(String(data=json.dumps({'x': tx, 'y': ty, 'radius': 0.4})))
                spin(1.0)
                pm = Path(); pm.header.frame_id = 'map'
                for (px, py) in path:
                    ps = PoseStamped(); ps.pose.position.x = float(px); ps.pose.position.y = float(py); pm.poses.append(ps)
                ep_pub.publish(String(data=json.dumps({'action': 'start', 'instruction': instr})))
                spin(0.3)
                path_pub.publish(pm)
                t0 = time.time(); c0 = state.get('collisions', 0); reached = False
                while time.time() - t0 < a.timeout:
                    spin(0.2)
                    if state.get('reached') or gx.get('done'): reached = True; break
                coll = state.get('collisions', 0) - c0
                label = 'good' if reached and coll == 0 else 'bad'
                ep_pub.publish(String(data=json.dumps({'action': 'stop', 'label': label})))
                path_pub.publish(Path())                       # clear the route -> expert stops
                spin(1.0)
                good += label == 'good'; bad += label == 'bad'; k += 1
                manifest.append({'map': map_name, 'start': [sx, sy], 'goal': [tx, ty], 'route_m': round(d, 1),
                                 'instruction': instr, 'label': label, 'reached': reached, 'collisions': coll,
                                 'secs': round(time.time() - t0, 1)})
                print(f'[{map_name} {k}/{a.per_map}] {d:4.1f} m  {label:4s}  "{instr}"', flush=True)
            sim.terminate(); sim.wait(timeout=5); spin(1.0)
    finally:
        expert.terminate(); logger.terminate()
        n.destroy_node(); rclpy.shutdown()
    os.makedirs(a.root, exist_ok=True)
    with open(os.path.join(a.root, 'manifest.jsonl'), 'a') as f:
        for m in manifest: f.write(json.dumps(m) + '\n')
    print(f'done: {good} good, {bad} bad -> {a.root}')


if __name__ == '__main__':
    main()
