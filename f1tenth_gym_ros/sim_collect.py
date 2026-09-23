#!/usr/bin/env python3
"""sim_collect: generate demonstration episodes in the simulator, automatically.

The raceline MPC is the expert. For each map it starts sim_env + raceline_mpc once, then
for each episode: resets the sim car to a random raceline waypoint (heading along the
line), starts an episode on the logger with a varied instruction, lets the expert drive
for `duration` s, and stops it labelled good (no collision) or bad. Output is the same
per-episode layout the real car produces, so tools/episodes_to_lerobot.py packages it.

    ROS_DOMAIN_ID=7 python3 tools/sim_collect.py --maps levine --per-map 20 --duration 20

Runs anywhere ROS 2 + the atlasautoware package are installed (the Jetson works).
"""
import argparse, csv, json, math, os, random, subprocess, sys, time

REPO = os.path.expanduser('~/atlas_ws/src/atlasautoware')
INSTRUCTIONS = [
    'follow the racing line around the track',
    'drive the track as fast as is safe',
    'complete a lap without touching the walls',
    'follow the track',
    'keep driving along the line',
    'drive around the circuit',
]


def sh(cmd, **kw):
    return subprocess.Popen(cmd, shell=True, executable='/bin/bash', **kw)


def raceline_for(map_name):
    yaml_p = os.path.join(REPO, 'maps', f'{map_name}.yaml')
    csv_p = os.path.join(REPO, 'racelines', f'{map_name}_auto.csv')
    if not os.path.isfile(csv_p):
        print(f'[build] raceline for {map_name} ...', flush=True)
        r = subprocess.run([sys.executable, os.path.join(REPO, 'tools', 'build_raceline.py'), yaml_p,
                            '--out', csv_p, '--no-validate'], capture_output=True, text=True)
        if not os.path.isfile(csv_p):
            print(r.stdout[-800:], r.stderr[-400:]); raise SystemExit(f'no raceline for {map_name}')
    pts = []
    for row in csv.reader(open(csv_p)):
        if not row or row[0].lstrip().startswith('#'): continue
        try: pts.append((float(row[0]), float(row[1])))
        except ValueError: continue
    return yaml_p, csv_p, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', default='levine')
    ap.add_argument('--per-map', type=int, default=10)
    ap.add_argument('--duration', type=float, default=20.0)
    ap.add_argument('--v-scales', default='0.5,0.7')
    ap.add_argument('--root', default=os.path.expanduser('~/sim_episodes'))
    ap.add_argument('--camera', default='true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    os.environ.setdefault('ROS_DOMAIN_ID', '7')

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    rclpy.init(); n = Node('sim_collect')
    reset_pub = n.create_publisher(String, '/sim/reset', 5)
    ep_pub = n.create_publisher(String, '/episode/cmd', 5)
    state = {}
    n.create_subscription(String, '/sim/state', lambda m: state.update(json.loads(m.data)), 5)
    epst = {}
    n.create_subscription(String, '/episode/status', lambda m: epst.update(json.loads(m.data)), 5)

    def spin(sec):
        t = time.time()
        while time.time() - t < sec: rclpy.spin_once(n, timeout_sec=0.05)

    src = 'source /opt/ros/humble/setup.bash; source ~/atlas_ws/install/setup.bash; '
    logger = sh(src + f'ros2 run f1tenth_gym_ros episode_logger --ros-args -p root:={a.root} '
                      f'-p image_topic:=/camera/color/image_raw -p odom_topic:=/odom -p imu_topic:=/none '
                      f'> /tmp/sim_logger.log 2>&1')
    total_good = total_bad = 0
    try:
        for map_name in a.maps.split(','):
            yaml_p, csv_p, pts = raceline_for(map_name)
            x0, y0 = pts[0]
            sim = sh(src + f'ros2 run f1tenth_gym_ros sim_env --ros-args -p map_yaml:={yaml_p} '
                           f'-p start_x:={x0} -p start_y:={y0} -p camera:={a.camera} > /tmp/sim_env.log 2>&1')
            spin(4.0)
            vscales = [float(v) for v in a.v_scales.split(',')]
            for k in range(a.per_map):
                vs = random.choice(vscales)
                mpc = sh(src + f'ros2 run f1tenth_gym_ros raceline_mpc --ros-args -p raceline:={csv_p} '
                               f'-p odom_topic:=/pf/pose/odom -p scan_topic:=/scan -p v_scale:={vs} '
                               f'-p steer_offset:=0.0 > /tmp/sim_mpc.log 2>&1')
                i = random.randrange(0, len(pts) - 5)
                (x, y), (x2, y2) = pts[i], pts[i + 3]
                th = math.atan2(y2 - y, x2 - x)
                reset_pub.publish(String(data=json.dumps({'x': x, 'y': y, 'theta': th})))
                spin(2.5)
                instr = random.choice(INSTRUCTIONS)
                ep_pub.publish(String(data=json.dumps({'action': 'start', 'instruction': instr})))
                t0 = time.time(); c0 = state.get('collisions', 0)
                spin(a.duration)
                coll = state.get('collisions', 0) - c0
                dist = state.get('distance_m', 0.0)
                label = 'good' if coll == 0 and dist > 2.0 else 'bad'
                ep_pub.publish(String(data=json.dumps({'action': 'stop', 'label': label})))
                spin(1.0)
                mpc.terminate(); mpc.wait(timeout=5)
                total_good += label == 'good'; total_bad += label == 'bad'
                print(f'[{map_name} {k+1}/{a.per_map}] v={vs} "{instr}" -> {label} '
                      f'(collisions {coll}, {dist:.1f} m total)', flush=True)
            sim.terminate(); sim.wait(timeout=5)
            spin(1.0)
    finally:
        logger.terminate()
        n.destroy_node(); rclpy.shutdown()
    print(f'done: {total_good} good, {total_bad} bad -> {a.root}')


if __name__ == '__main__':
    main()
