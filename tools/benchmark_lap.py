"""
Benchmark one solo run on the current best_raceline + tune_config, in sim.

Runs inside the container (ROS sourced).  Resets the car to the start, launches
race_agent (solo), samples runtime/race_state.json, and prints a JSON score:

  { laps, best_lap, progress, mean_speed, stuck, score }

score (lower = better):
  - completed >=1 lap  -> best_lap seconds
  - otherwise          -> 1000 - 400*progress  (always worse than any real lap)

    python3 tools/benchmark_lap.py --time 70
"""

import argparse
import json
import os
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(REPO, 'runtime', 'race_state.json')
RL = os.path.join(REPO, 'racelines', 'best_raceline.csv')   # for the lap-length (n)
# competition-track start pose
RESET = (
    "ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "
    "'{header: {frame_id: map}, pose: {pose: {position: {x: 49.815, y: 62.230, z: 0.0}, "
    "orientation: {z: -0.9685, w: 0.249}}}}'")


def sh(cmd, timeout=20):
    subprocess.run(['bash', '-lc', cmd], capture_output=True, timeout=timeout)


def kill_agent():
    subprocess.run(['bash', '-lc',
                    "for p in $(ps -eo pid,args | grep race_agent.py | grep -v grep "
                    "| awk '{print $1}'); do kill $p 2>/dev/null; done"],
                   capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--time', type=float, default=70.0)
    args = ap.parse_args()

    kill_agent()
    time.sleep(1.0)
    if os.path.exists(STATE):
        os.remove(STATE)
    sh('source /opt/ros/foxy/setup.bash; source /sim_ws/install/setup.bash 2>/dev/null; '
       'timeout 3 ' + RESET + ' >/dev/null 2>&1')
    time.sleep(0.5)

    # launch agent (inherits ROS env from the caller)
    proc = subprocess.Popen(
        ['bash', '-lc',
         'source /opt/ros/foxy/setup.bash; source /sim_ws/install/setup.bash 2>/dev/null; '
         'cd %s; python3 f1tenth_gym_ros/race_agent.py' % REPO],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # raceline length (for unwrapped progress, decoupled from where index 0 is)
    n = 300
    try:
        import csv
        with open(RL) as f:
            n = sum(1 for _ in csv.DictReader(f))
    except Exception:
        pass

    t0 = time.time()
    prev_near = None
    cum = 0.0                                   # cumulative forward indices travelled
    lap_marks = [t0]                            # wall-times at each full-lap crossing
    speeds, poshist = [], []
    stuck = False
    try:
        while time.time() - t0 < args.time:
            time.sleep(0.3)
            if not os.path.exists(STATE):
                continue
            try:
                with open(STATE) as f:
                    st = json.load(f)
            except Exception:
                continue
            near = st.get('nearest', 0)
            if prev_near is not None:
                d = near - prev_near             # unwrap the index step
                if d < -n / 2:
                    d += n
                elif d > n / 2:
                    d -= n
                if 0 < d < n / 2:                # forward only (ignore jitter)
                    cum += d
            prev_near = near
            while cum >= len(lap_marks) * n:     # crossed a full lap
                lap_marks.append(time.time())
            speeds.append(st.get('speed', 0.0))
            # stuck: not advancing around the track for ~6 s (even if wiggling in
            # place) — ends the run early so a corner the car can't clear doesn't
            # waste the whole window.
            poshist.append((time.time(), cum))
            poshist = [p for p in poshist if p[0] > time.time() - 6.0]
            if len(poshist) > 14 and (poshist[-1][1] - poshist[0][1]) < 4:
                stuck = True
                break
    finally:
        proc.terminate()
        kill_agent()

    progress = cum / float(n)                   # laps (fractional), from the start
    laps = len(lap_marks) - 1
    lap_times = [lap_marks[i + 1] - lap_marks[i] for i in range(laps)]
    best_lap = min(lap_times) if lap_times else None
    mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
    if laps >= 1:
        score = round(best_lap, 2)
    else:
        score = round(1000 - 400 * progress, 2)
    print(json.dumps({'laps': laps, 'best_lap': round(best_lap, 2) if best_lap else None,
                      'progress': round(progress, 3), 'mean_speed': round(mean_speed, 2),
                      'stuck': stuck, 'score': score}))


if __name__ == '__main__':
    main()
