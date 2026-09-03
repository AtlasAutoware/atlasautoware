#!/usr/bin/env python3
"""Autonomy supervisor and track manager behind the web pilot page.

Self-driving here means running `raceline_mpc`, which publishes AckermannDrive on
/drive. The manual F1TENTH bring-up already muxes that: `navigation` (topic drive)
has priority 10 and `joystick` (topic teleop) has priority 100, so a human holding
the dead-man always outranks the policy, with no mode switch and nothing to unwind.
Autonomy is therefore an extra publisher, not a takeover.

Three independent things can stop it: the STOP button, the browser going quiet
(the page sends a heartbeat; no heartbeat for `hb_timeout` seconds kills the node),
and raceline_mpc exiting on its own. The car's own watchdogs sit underneath all of it.

Track pictures become racelines with the existing offline tools:
    image -> track_from_image.py -> maps/<name>.{pgm,yaml} -> build_raceline.py
          -> racelines/<name>_auto.csv (+ overlay PNG, + feasibility report)
Both steps run as subprocesses so a slow optimizer never blocks the video stream.
"""
import glob, json, os, re, shutil, signal, subprocess, threading, time

REPO_CANDIDATES = [os.path.expanduser('~/atlas_ws/src/atlasautoware'),
                   os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]


def find_repo():
    for p in REPO_CANDIDATES:
        if os.path.isdir(os.path.join(p, 'tools')) and os.path.isdir(os.path.join(p, 'maps')):
            return p
    return REPO_CANDIDATES[-1]


REPO = find_repo()
MAPS = os.path.join(REPO, 'maps')
RACELINES = os.path.join(REPO, 'racelines')
UPLOADS = os.path.expanduser('~/track_uploads')
SAFE = re.compile(r'^[A-Za-z0-9_-]{1,40}$')


def safe_name(n):
    n = (n or '').strip().replace(' ', '_')
    return n if SAFE.match(n) else ''


# ── tracks ──────────────────────────────────────────────────────────────────────
def list_tracks():
    maps = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(MAPS, '*.yaml')))
    lines = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RACELINES, '*.csv')))
    return {'maps': maps, 'racelines': lines, 'repo': REPO}


def save_upload(data, name, ext='.jpg'):
    os.makedirs(UPLOADS, exist_ok=True)
    path = os.path.join(UPLOADS, f'{name}{ext}')
    with open(path, 'wb') as f:
        f.write(data)
    return path


class Job:
    """A background subprocess whose log the page can poll."""

    def __init__(self):
        self.proc = None; self.log = []; self.name = ''; self.rc = None
        self.lock = threading.Lock(); self.started = 0.0

    def busy(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd, name, cwd=None, env=None):
        if self.busy():
            return False, 'a job is already running'
        with self.lock:
            self.log = [f'$ {" ".join(cmd)}']; self.name = name; self.rc = None; self.started = time.time()
        try:
            self.proc = subprocess.Popen(cmd, cwd=cwd or REPO, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as e:
            with self.lock: self.log.append(f'failed to start: {e}'); self.rc = -1
            return False, str(e)
        threading.Thread(target=self._pump, daemon=True).start()
        return True, 'started'

    def _pump(self):
        for line in self.proc.stdout:
            with self.lock:
                self.log.append(line.rstrip()[:400])
                if len(self.log) > 400: self.log = self.log[-400:]
        self.proc.wait()
        with self.lock:
            self.rc = self.proc.returncode
            self.log.append(f'[exit {self.rc}]')

    def status(self):
        with self.lock:
            return {'name': self.name, 'busy': self.busy(), 'rc': self.rc,
                    'secs': round(time.time() - self.started, 1) if self.started else 0,
                    'log': self.log[-40:]}


def convert_cmd(image_path, name, mode, track_width, resolution, py=None):
    script = os.path.join(REPO, 'f1tenth_gym_ros', 'track_from_image.py')
    cmd = [py or 'python3', script, image_path, '--name', name, '--maps-dir', MAPS, '--mode', mode]
    if resolution: cmd += ['--resolution', str(resolution)]
    else: cmd += ['--track-width', str(track_width)]
    return cmd


def build_cmd(name, a_lat=6.5, v_max=7.0, margin=0.2, py=None):
    return [py or 'python3', os.path.join(REPO, 'tools', 'build_raceline.py'),
            os.path.join(MAPS, f'{name}.yaml'),
            '--out', os.path.join(RACELINES, f'{name}_auto.csv'),
            '--a-lat', str(a_lat), '--v-max', str(v_max), '--margin', str(margin)]


# ── autonomy ────────────────────────────────────────────────────────────────────
class Supervisor:
    """Owns the raceline_mpc process and the reasons it may not start or must stop."""

    def __init__(self, hb_timeout=2.0, log_lines=200):
        self.proc = None; self.log = []; self.lock = threading.Lock()
        self.hb = 0.0; self.hb_timeout = hb_timeout; self.log_lines = log_lines
        self.engaged_at = 0.0; self.params = {}; self.last_stop = ''

    def engaged(self):
        return self.proc is not None and self.proc.poll() is None

    def heartbeat(self):
        self.hb = time.time()

    def preflight(self, ages, raceline, odom_topic, max_age=1.5):
        """ages: {topic: seconds since last message or None}. Returns (ok, checks)."""
        def chk(name, ok, detail):
            return {'name': name, 'ok': bool(ok), 'detail': detail}
        c = []
        for topic, label in (('scan', 'lidar /scan'), ('core', 'VESC telemetry')):
            a = ages.get(topic)
            c.append(chk(label, a is not None and a < max_age,
                         'no data' if a is None else f'{a*1000:.0f} ms ago'))
        a = ages.get('odom')
        c.append(chk(f'pose {odom_topic}', a is not None and a < max_age,
                     'not publishing (start localization or pick another topic)' if a is None
                     else f'{a*1000:.0f} ms ago'))
        p = os.path.join(RACELINES, raceline) if raceline and not os.path.isabs(raceline) else raceline
        c.append(chk('raceline', bool(p) and os.path.isfile(p), os.path.basename(p or '') or 'none selected'))
        c.append(chk('battery', ages.get('volts') is None or ages['volts'] > 13.0,
                     f"{ages.get('volts')} V" if ages.get('volts') else 'unknown'))
        return all(x['ok'] for x in c), c

    def engage(self, raceline, v_scale, odom_topic, scan_topic='/scan', imu_topic='/oakd/imu',
               extra=None, env=None):
        if self.engaged():
            return False, 'already engaged'
        path = raceline if os.path.isabs(raceline) else os.path.join(RACELINES, raceline)
        if not os.path.isfile(path):
            return False, f'no such raceline: {raceline}'
        v_scale = max(0.05, min(1.0, float(v_scale)))
        cmd = ['ros2', 'run', 'f1tenth_gym_ros', 'raceline_mpc', '--ros-args',
               '-p', f'raceline:={path}', '-p', f'v_scale:={v_scale}',
               '-p', f'odom_topic:={odom_topic}', '-p', f'scan_topic:={scan_topic}',
               '-p', f'imu_topic:={imu_topic}', '-p', 'drive_topic:=/drive']
        for k, v in (extra or {}).items():
            cmd += ['-p', f'{k}:={v}']
        return self._launch(cmd, {'mode': 'raceline', 'raceline': os.path.basename(path),
                                  'v_scale': v_scale, 'odom_topic': odom_topic}, env)

    def engage_policy(self, instruction, max_speed, scan_topic='/scan', model='models/student.onnx',
                      odom_topic='/odom', env=None):
        """Mode 3: the distilled goal-conditioned student (policy_bridge) drives via /drive."""
        if self.engaged():
            return False, 'already engaged'
        path = model if os.path.isabs(model) else os.path.join(REPO, model)
        if not os.path.isfile(path):
            return False, f'no model at {path} (train it on the cluster: ml/slurm/pipeline.sh)'
        max_speed = max(0.1, min(2.0, float(max_speed)))
        cmd = ['ros2', 'run', 'f1tenth_gym_ros', 'policy_bridge', '--ros-args',
               '-p', f'model:={path}', '-p', f'instruction:={instruction}',
               '-p', f'max_speed:={max_speed}', '-p', f'scan_topic:={scan_topic}',
               '-p', f'odom_topic:={odom_topic}', '-p', 'drive_topic:=/drive']
        return self._launch(cmd, {'mode': 'policy', 'instruction': instruction, 'max_speed': max_speed,
                                  'model': os.path.basename(path)}, env)

    def _launch(self, cmd, params, env=None):
        with self.lock:
            self.log = [f'$ {" ".join(cmd)}']
        try:
            self.proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                                         preexec_fn=os.setsid)
        except OSError as e:
            return False, f'could not start {cmd[3]}: {e}'
        self.hb = time.time(); self.engaged_at = time.time(); self.last_stop = ''
        self.params = params
        threading.Thread(target=self._pump, daemon=True).start()
        return True, 'engaged'

    def _pump(self):
        for line in self.proc.stdout:
            with self.lock:
                self.log.append(line.rstrip()[:300])
                if len(self.log) > self.log_lines: self.log = self.log[-self.log_lines:]
        self.proc.wait()

    def stop(self, why='stop button'):
        if self.proc is None:
            return False
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)   # let it publish a zero drive
                for _ in range(20):
                    if self.proc.poll() is not None: break
                    time.sleep(0.05)
                if self.proc.poll() is None:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.last_stop = why
        with self.lock: self.log.append(f'[stopped: {why}]')
        return True

    def tick(self):
        """Call at loop rate. Stops autonomy if the page went quiet or the node died."""
        if not self.engaged():
            return
        if time.time() - self.hb > self.hb_timeout:
            self.stop(f'no heartbeat from the browser for {self.hb_timeout:.1f} s')

    def status(self):
        with self.lock: log = self.log[-25:]
        return {'engaged': self.engaged(), 'params': self.params, 'log': log,
                'secs': round(time.time() - self.engaged_at, 1) if self.engaged() else 0,
                'last_stop': self.last_stop}


# ── multipart (one file + simple fields), enough for the upload form ────────────
def parse_multipart(body, content_type):
    m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type or '')
    if not m:
        return {}, None, None
    boundary = ('--' + (m.group(1) or m.group(2)).strip()).encode()
    fields, filename, filedata = {}, None, None
    for part in body.split(boundary):
        if not part.strip() or part.strip() == b'--':
            continue
        head, _, data = part.partition(b'\r\n\r\n')
        if not _:
            continue
        data = data.rstrip(b'\r\n')
        head_s = head.decode('utf-8', 'replace')
        nm = re.search(r'name="([^"]*)"', head_s)
        fn = re.search(r'filename="([^"]*)"', head_s)
        if not nm:
            continue
        if fn and fn.group(1):
            filename, filedata = fn.group(1), data
        else:
            fields[nm.group(1)] = data.decode('utf-8', 'replace')
    return fields, filename, filedata
