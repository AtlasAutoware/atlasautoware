"""
Race Control dashboard — backend.
=================================

A dependency-free (stdlib only) web server you run on the HOST:

    python ui/server.py        # then open http://localhost:8000

It drives the whole pipeline by shelling into the sim container
(`f1tenth_gym_ros-sim-1`) and reads result images + live telemetry straight from
the mounted repo, so the browser sees everything without a ROS bridge.

Endpoints
  GET  /                      -> dashboard
  GET  /api/state             -> live race telemetry (runtime/race_state.json)
  GET  /api/raceline          -> current raceline polyline (x,y,speed)
  GET  /api/image/<name>      -> a result PNG from racelines/
  POST /api/generate          -> regenerate raceline {margin,apex_bias,a_lat,v_max}
  POST /api/race/start        -> launch the 2-car opponent demo
  POST /api/race/stop         -> stop it
"""

import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTAINER = 'f1tenth_gym_ros-sim-1'
SEED = ('49.910', '42.780')          # competition-track corridor seed
PORT = 8000


def dx(cmd, timeout=120):
    """Run a bash command inside the sim container; return (ok, output)."""
    try:
        p = subprocess.run(['docker', 'exec', CONTAINER, 'bash', '-lc', cmd],
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout + p.stderr)
    except Exception as e:
        return False, str(e)


def ros_prefix():
    return ('source /opt/ros/foxy/setup.bash; '
            'source /sim_ws/install/setup.bash 2>/dev/null; '
            'cd /sim_ws/src/f1tenth_gym_ros; ')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):            # quiet
        pass

    # ── helpers ──────────────────────────────────────────────────────────────
    def _send(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get('Content-Length', 0))
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n))
        except Exception:
            return {}

    # ── routes ───────────────────────────────────────────────────────────────
    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/':
            return self._file('index.html', 'text/html')
        if path == '/api/state':
            return self._state()
        if path == '/api/raceline':
            return self._raceline()
        if path.startswith('/api/image/'):
            return self._image(path.rsplit('/', 1)[-1])
        return self._send(404, {'error': 'not found'})

    def do_POST(self):
        if self.path == '/api/generate':
            return self._generate(self._body())
        if self.path == '/api/race/start':
            return self._race_start()
        if self.path == '/api/race/stop':
            return self._race_stop()
        return self._send(404, {'error': 'not found'})

    # ── implementations ──────────────────────────────────────────────────────
    def _file(self, name, ctype):
        p = os.path.join(os.path.dirname(__file__), name)
        if not os.path.exists(p):
            return self._send(404, 'missing ' + name, 'text/plain')
        with open(p, 'rb') as f:
            self._send(200, f.read(), ctype)

    def _image(self, name):
        p = os.path.join(REPO, 'racelines', os.path.basename(name))
        if not os.path.exists(p):
            return self._send(404, b'', 'image/png')
        with open(p, 'rb') as f:
            self._send(200, f.read(), 'image/png')

    def _state(self):
        p = os.path.join(REPO, 'runtime', 'race_state.json')
        if not os.path.exists(p):
            return self._send(200, {'running': False})
        try:
            with open(p) as f:
                st = json.load(f)
            st['running'] = (time.time() - st.get('ts', 0)) < 1.5
            return self._send(200, st)
        except Exception:
            return self._send(200, {'running': False})

    def _raceline(self):
        p = os.path.join(REPO, 'racelines', 'best_raceline.csv')
        xs, ys, sp = [], [], []
        try:
            import csv
            with open(p) as f:
                for r in csv.DictReader(f):
                    xs.append(float(r['x'])); ys.append(float(r['y']))
                    sp.append(float(r['speed']))
        except Exception:
            pass
        return self._send(200, {'x': xs, 'y': ys, 'speed': sp})

    def _generate(self, body):
        m = float(body.get('margin', 0.35))
        a = float(body.get('apex_bias', 1.0))
        al = float(body.get('a_lat', 6.5))
        vm = float(body.get('v_max', 7.0))
        cmd = (ros_prefix() +
               f'python3 f1tenth_gym_ros/raceline_optimizer.py '
               f'--map maps/comp_track.yaml --output racelines/best_raceline.csv '
               f'--seed {SEED[0]} {SEED[1]} --margin {m} --apex-bias {a} '
               f'--a-lat {al} --v-max {vm} --no-overlay && '
               f'python3 tools/annotate_raceline.py --image racetrackForComp.png '
               f'--csv racelines/best_raceline.csv --yaml maps/comp_track.yaml '
               f'--out racelines/comp_raceline_annotated.png')
        ok, out = dx(cmd)
        stats = {}
        for line in out.splitlines():
            if line.startswith('[speed]') or line.startswith('[centerline]') \
                    or line.startswith('[optimize]'):
                stats[line.split(']')[0].strip('[')] = line.split(']', 1)[1].strip()
        return self._send(200, {'ok': ok, 'stats': stats, 'log': out[-800:],
                                'image': 'comp_raceline_annotated.png?t=' + str(int(time.time()))})

    def _race_start(self):
        # kill any old, reset poses, launch opponent + race agent
        kill = ("for p in $(ps -eo pid,args | grep -E 'race_agent.py|opponent_driver.py' "
                "| grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done; sleep 1; ")
        reset = (
            "timeout 3 ros2 topic pub --once /initialpose "
            "geometry_msgs/msg/PoseWithCovarianceStamped "
            "'{header: {frame_id: map}, pose: {pose: {position: {x: 49.815, y: 62.230, z: 0.0}, "
            "orientation: {z: -0.9685, w: 0.249}}}}' >/dev/null 2>&1; "
            "timeout 3 ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped "
            "'{header: {frame_id: map}, pose: {position: {x: 45.340, y: 55.192, z: 0.0}, "
            "orientation: {z: -0.9075, w: 0.42}}}' >/dev/null 2>&1; ")
        launch = ('nohup python3 f1tenth_gym_ros/opponent_driver.py --cap 3.0 '
                  '>/tmp/opp.log 2>&1 & sleep 1; '
                  'nohup python3 f1tenth_gym_ros/race_agent.py >/tmp/race.log 2>&1 & ')
        ok, out = dx(ros_prefix() + kill + reset + launch, timeout=30)
        return self._send(200, {'ok': ok})

    def _race_stop(self):
        cmd = ("for p in $(ps -eo pid,args | grep -E 'race_agent.py|opponent_driver.py' "
               "| grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done; "
               "rm -f /sim_ws/src/f1tenth_gym_ros/runtime/race_state.json")
        ok, out = dx(cmd, timeout=15)
        return self._send(200, {'ok': ok})


if __name__ == '__main__':
    print(f'Race Control dashboard on http://localhost:{PORT}  (repo: {REPO})')
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
