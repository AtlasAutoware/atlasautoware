#!/usr/bin/env python3
"""Web pilot: drive the car from any browser on the same WiFi. FPV + WASD + gamepad.

    http://<car>:8080/          the page (stream, keyboard, gamepad, telemetry)
    http://<car>:8080/stream    MJPEG only (VLC, another tab, ...)
    POST /cmd  {"axes":[6],"buttons":[11]}   ROS-joy layout, 20-50 Hz from the page
    UDP :5005  same JSON from tools/remote_pilot.py (optional python client)

Whatever source sent the most recent command wins; if nothing arrives for `timeout`
seconds (0.25) a neutral Joy is published and the car stops. /joy feeds the unchanged
F1TENTH chain: joy_teleop (F310 profile: button 4 = dead-man, axis 1 throttle, axis 3
steer) -> ackermann_mux -> ackermann_to_vesc -> vesc_driver.
"""
import json, os, socket, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np, cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Joy, LaserScan

N_AXES, N_BUTTONS = 6, 11
TRIM_FILE = os.path.expanduser('~/.atlascar_trim.json')   # steering trim survives restarts
S = {'jpg': None, 'jpg_t': 0.0, 'cmd': None, 'cmd_t': 0.0, 'src': '-', 'v': None, 'rx': 0, 'scan': None,
     'trim': 0.0, 'video': {'width': 480, 'quality': 60, 'fps': 15.0}, 'lock': threading.Lock()}


def load_trim():
    try:
        with open(TRIM_FILE) as f: return float(json.load(f).get('steer_trim', 0.0))
    except (OSError, ValueError): return 0.0


def save_trim(t):
    try:
        with open(TRIM_FILE, 'w') as f: json.dump({'steer_trim': t}, f)
    except OSError: pass

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>AtlasCar pilot</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{margin:0;background:#0b0b0b;color:#ddd;font:14px system-ui,sans-serif;overflow:hidden}
 #v{position:fixed;inset:0;width:100vw;height:100vh;object-fit:contain;background:#000}
 #hud{position:fixed;left:0;right:0;top:0;padding:8px 12px;background:rgba(0,0,0,.55);display:flex;gap:18px;align-items:center;flex-wrap:wrap}
 #arm{font-weight:700;padding:2px 10px;border-radius:6px;background:#552}
 #arm.on{background:#2a7}
 .bar{display:inline-block;width:120px;height:10px;background:#333;border-radius:5px;vertical-align:middle;position:relative}
 .bar i{position:absolute;top:0;bottom:0;background:#4c9;border-radius:5px}
 #help{position:fixed;bottom:10px;left:12px;background:rgba(0,0,0,.55);padding:6px 10px;border-radius:6px}
 input[type=range]{width:120px;vertical-align:middle}
 #lidar{position:fixed;right:12px;bottom:12px;width:260px;height:260px;background:rgba(0,0,0,.55);border-radius:8px}
 #lbl{position:fixed;right:16px;bottom:276px;font-size:12px;color:#9cf}
</style></head><body>
<img id="v" src="/stream">
<canvas id="lidar" width="260" height="260"></canvas><span id="lbl">lidar 6 m <label><input id="mirror" type="checkbox"> mirror</label></span>
<div id="hud">
 <span id="arm">STOPPED</span>
 <span>thr <span class="bar"><i id="tb"></i></span> <span id="tv">0.00</span></span>
 <span>steer <span class="bar"><i id="sb"></i></span> <span id="sv">0.00</span></span>
 <span>max <input id="max" type="range" min="0.1" max="1" step="0.05" value="0.4"> <span id="mv">40%</span></span>
 <span>trim <button id="tl">&lsaquo;</button> <span id="trim">0.00</span> <button id="tr">&rsaquo;</button></span>
 <span>video <select id="vq"><option value="normal">normal</option><option value="low">low (cellular)</option></select></span>
 <span id="pad">no gamepad</span>
 <span id="link">link: -</span>
 <span id="batt">batt: -</span>
</div>
<div id="help">W/S throttle &nbsp; A/D steer &nbsp; Q/E trim (car pulls right &rarr; press Q) &nbsp; (keys held = armed; release = stop) &nbsp;|&nbsp; gamepad: hold LB, left stick throttle, right stick steer</div>
<script>
const keys={}; let thr=0, str=0, lastSend=0, lastArmed=-1e9, rtt='-', padName=null;
const $=id=>document.getElementById(id);
addEventListener('keydown',e=>{if(['w','a','s','d','W','A','S','D'].includes(e.key)){keys[e.key.toLowerCase()]=1;e.preventDefault();}});
addEventListener('keyup',e=>{keys[e.key.toLowerCase()]=0;});
addEventListener('blur',()=>{for(const k in keys)keys[k]=0;});
$('max').oninput=()=>{$('mv').textContent=Math.round($('max').value*100)+'%';};
// ── steering trim: stored on the car, applied to every source (keys, pad, UDP) ──
let trim=0;
function showTrim(t){trim=t;$('trim').textContent=(t>=0?'+':'')+t.toFixed(2);
  $('trim').title='equivalent vesc.yaml steering_angle_to_servo_offset change: '+(-0.4126*t).toFixed(4);}
function setTrim(t){t=Math.max(-0.3,Math.min(0.3,t));
  fetch('/trim',{method:'POST',body:JSON.stringify({steer_trim:t})}).then(r=>r.json()).then(s=>showTrim(s.steer_trim)).catch(()=>{});}
$('tl').onclick=()=>setTrim(trim+0.01); $('tr').onclick=()=>setTrim(trim-0.01);   // ROS: left = +
addEventListener('keydown',e=>{if(e.key==='q'||e.key==='Q')setTrim(trim+0.01); if(e.key==='e'||e.key==='E')setTrim(trim-0.01);});
fetch('/trim').then(r=>r.json()).then(s=>showTrim(s.steer_trim)).catch(()=>{});
// ── video quality: normal (480 px, q60, 15 fps) or low for cellular (320 px, q45, 10 fps) ──
$('vq').onchange=()=>{const low=$('vq').value==='low';
  fetch('/video?w='+(low?320:480)+'&q='+(low?45:60)+'&fps='+(low?10:15),{method:'POST'}).catch(()=>{});};
function step(target,cur,rate){return cur+Math.max(-rate,Math.min(rate,target-cur));}
function tick(){
  const dt=0.02, max=parseFloat($('max').value);
  let axes=[0,0,1,0,0,1], buttons=new Array(11).fill(0), src='';
  const gp=(navigator.getGamepads?navigator.getGamepads():[])[0];
  if(gp){ padName=gp.id.slice(0,28);
    // browser standard mapping -> ROS joy F310 layout (left/up = +1, triggers rest +1)
    axes=[-gp.axes[0], -gp.axes[1], 1, -gp.axes[2], -gp.axes[3], 1];
    for(let i=0;i<Math.min(11,gp.buttons.length);i++) buttons[i]=gp.buttons[i].pressed?1:0;
    if(buttons[4]){ src='pad'; thr=-gp.axes[1]*max; str=-gp.axes[2]; axes[1]=thr; }
  }
  const kthr=(keys.w?1:0)-(keys.s?1:0), kstr=(keys.a?1:0)-(keys.d?1:0);   // ROS: left = +1
  if(kthr||kstr){ src='keys';
    thr=step(kthr*max,thr,1.5*dt); str=step(kstr,str,4*dt);
    axes=[0,thr,1,str,0,1]; buttons[4]=1;
  } else if(!(gp&&buttons[4])){ thr=step(0,thr,3*dt); str=step(0,str,6*dt); }
  const armed=buttons[4]===1;
  $('arm').textContent=armed?('ARMED ('+src+')'):'STOPPED'; $('arm').className=armed?'on':'';
  $('tv').textContent=thr.toFixed(2); $('sv').textContent=str.toFixed(2);
  $('tb').style.left=(50+Math.min(0,thr)*50)+'%'; $('tb').style.width=(Math.abs(thr)*50)+'%';
  $('sb').style.left=(50-Math.max(0,str)*50)+'%'; $('sb').style.width=(Math.abs(str)*50)+'%';
  $('pad').textContent=padName?('pad: '+padName):'no gamepad (press a button on it)';
  const now=performance.now();
  if(armed) lastArmed=now;
  // Only a tab that is actually driving sends commands (plus ~0.6 s of explicit neutral
  // after release). Idle tabs just watch, so a second viewer never stomps the driver.
  const sending=armed||(now-lastArmed<600);
  if(sending && now-lastSend>=33){ lastSend=now; const t0=now;
    fetch('/cmd',{method:'POST',body:JSON.stringify({axes,buttons}),keepalive:true})
      .then(r=>r.json()).then(s=>{rtt=Math.round(performance.now()-t0);
        $('link').textContent='link: '+rtt+' ms';
        $('batt').textContent='batt: '+(s.v?s.v.toFixed(1)+' V':'-');})
      .catch(()=>{$('link').textContent='link: DOWN';});
  } else if(!sending && now-lastSend>=500){ lastSend=now;
    fetch('/status').then(r=>r.json()).then(s=>{
      $('link').textContent='link: idle'+(s.link?' (driver: '+s.src+')':'');
      $('batt').textContent='batt: '+(s.v?s.v.toFixed(1)+' V':'-');}).catch(()=>{$('link').textContent='link: DOWN';});
  }
}
setInterval(tick,20);
addEventListener('gamepadconnected',e=>{padName=e.gamepad.id.slice(0,28);});
// ── lidar: top-down plot, car at centre, forward = up, 6 m radius ──
const cv=$('lidar'), ctx=cv.getContext('2d'), R=6.0, SC=124/R;
function drawScan(s){
  ctx.clearRect(0,0,260,260); ctx.strokeStyle='#345'; ctx.lineWidth=1;
  for(const m of [1,2,4]){ctx.beginPath();ctx.arc(130,130,m*SC,0,2*Math.PI);ctx.stroke();}
  ctx.beginPath();ctx.moveTo(130,130);ctx.lineTo(130,4);ctx.stroke();       // heading
  if(!s){ctx.fillStyle='#777';ctx.fillText('no /scan',100,134);return;}
  const mir=$('mirror').checked?-1:1; ctx.fillStyle='#4cf';
  for(let i=0;i<s.r.length;i++){ const r=s.r[i]; if(!(r>0.05&&r<R)) continue;
    const a=s.a0+i*s.da, x=130-mir*r*Math.sin(a)*SC, y=130-r*Math.cos(a)*SC; ctx.fillRect(x-1,y-1,2,2); }
  ctx.fillStyle='#e94'; ctx.fillRect(126,126,8,8);                              // the car
}
setInterval(()=>fetch('/scan').then(r=>r.json()).then(drawScan).catch(()=>drawScan(null)),200);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == '/':
            b = PAGE.encode(); self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b); return
        if self.path == '/status':
            self._json(status()); return
        if self.path == '/scan':
            with S['lock']: sc = S['scan']
            self._json(sc if sc else {}); return
        if self.path == '/trim':
            with S['lock']: t = S['trim']
            self._json({'steer_trim': t}); return
        if self.path != '/stream':
            self.send_response(404); self.send_header('Content-Length', '0'); self.end_headers(); return
        self.send_response(200); self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache'); self.end_headers()
        last = 0.0
        try:
            while True:
                with S['lock']: jpg, t = S['jpg'], S['jpg_t']
                if jpg is None or t == last: time.sleep(0.01); continue
                last = t
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
        except (BrokenPipeError, ConnectionResetError): pass
    def do_POST(self):
        u = urlparse(self.path); n = int(self.headers.get('Content-Length', 0)); body = self.rfile.read(n)
        if u.path == '/cmd':
            if accept(body, 'web:' + self.client_address[0]): self._json(status())
            else: self._json({'error': 'bad command'}, 400)
            return
        if u.path == '/trim':
            try: t = max(-0.3, min(0.3, float(json.loads(body.decode())['steer_trim'])))
            except (ValueError, KeyError, TypeError): self._json({'error': 'bad trim'}, 400); return
            with S['lock']: S['trim'] = t
            save_trim(t); self._json({'steer_trim': t}); return
        if u.path == '/video':
            q = parse_qs(u.query)
            with S['lock']:
                v = S['video']
                v['width'] = int(q.get('w', [v['width']])[0]); v['quality'] = int(q.get('q', [v['quality']])[0]); v['fps'] = float(q.get('fps', [v['fps']])[0])
                self._json(dict(v)); return
        self.send_response(404); self.send_header('Content-Length', '0'); self.end_headers()


def accept(raw, src):
    try:
        d = json.loads(raw.decode()); axes = [float(x) for x in d['axes']][:N_AXES]; btns = [int(b) for b in d['buttons']][:N_BUTTONS]
    except (ValueError, KeyError, TypeError): return False
    axes += [0.0] * (N_AXES - len(axes)); btns += [0] * (N_BUTTONS - len(btns))
    with S['lock']: S['cmd'], S['cmd_t'], S['src'] = (axes, btns), time.time(), src; S['rx'] += 1
    return True


def status():
    with S['lock']:
        age = None if S['cmd'] is None else time.time() - S['cmd_t']
        return {'link': age is not None and age < 0.25, 'age_ms': None if age is None else round(age * 1000), 'src': S['src'], 'v': S['v']}


class WebPilot(Node):
    def __init__(self):
        super().__init__('web_pilot')
        for k, v in (('http_port', 8080), ('udp_port', 5005), ('joy_topic', '/joy'), ('timeout', 0.25), ('publish_hz', 50.0),
                     ('image_topic', '/camera/color/image_raw'), ('width', 480), ('quality', 60), ('fps', 15.0), ('udp_invert_axes', True)):
            self.declare_parameter(k, v)
        p = lambda n: self.get_parameter(n).value
        self.timeout = float(p('timeout'))
        with S['lock']:
            S['video'] = {'width': int(p('width')), 'quality': int(p('quality')), 'fps': float(p('fps'))}
            S['trim'] = load_trim()
        self.udp_invert = bool(p('udp_invert_axes')); self.last_jpg = 0.0; self.link_up = False
        if S['trim']: self.get_logger().info(f"steering trim restored: {S['trim']:+.2f} ({TRIM_FILE})")
        self.pub = self.create_publisher(Joy, p('joy_topic'), 10)
        self.create_subscription(Image, p('image_topic'), self._img, qos_profile_sensor_data)
        self.declare_parameter('scan_topic', '/scan'); self.last_scan = 0.0
        self.create_subscription(LaserScan, p('scan_topic'), self._scan, qos_profile_sensor_data)
        try:
            from vesc_msgs.msg import VescStateStamped
            self.create_subscription(VescStateStamped, '/sensors/core', lambda m: S.__setitem__('v', round(float(m.state.voltage_input), 2)), 10)
        except Exception: pass
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', int(p('udp_port')))); self.sock.setblocking(False)
        srv = ThreadingHTTPServer(('0.0.0.0', int(p('http_port'))), H); srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.create_timer(1.0 / float(p('publish_hz')), self._tick)
        self.get_logger().info(f"web pilot: http://<car-ip>:{int(p('http_port'))}/  (UDP :{int(p('udp_port'))} also accepted); watchdog {self.timeout*1000:.0f} ms")

    def _img(self, m):
        now = time.time()
        with S['lock']: v = dict(S['video'])
        if now - self.last_jpg < 1.0 / max(1.0, v['fps']) or m.encoding not in ('rgb8', 'bgr8'): return
        self.last_jpg = now
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        if m.encoding == 'rgb8': a = a[:, :, ::-1]
        w = int(v['width'])
        if m.width != w: a = cv2.resize(a, (w, int(m.height * w / m.width)), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode('.jpg', a, [cv2.IMWRITE_JPEG_QUALITY, int(v['quality'])])
        if ok:
            with S['lock']: S['jpg'], S['jpg_t'] = jpg.tobytes(), now

    def _scan(self, m):
        now = time.time()
        if now - self.last_scan < 0.2: return                    # 5 Hz is plenty for a picture
        self.last_scan = now
        k = max(1, len(m.ranges) // 360)                          # ~360 points
        rs = [round(float(r), 2) if r == r and r < 1e5 else 0.0 for r in m.ranges[::k]]
        with S['lock']: S['scan'] = {'a0': float(m.angle_min), 'da': float(m.angle_increment) * k, 'r': rs, 'rmax': float(m.range_max)}

    def _drain_udp(self):
        while True:
            try: data, addr = self.sock.recvfrom(2048)
            except (BlockingIOError, OSError): return
            if self.udp_invert:                                   # pygame/SDL sign -> ROS joy sign
                try:
                    d = json.loads(data.decode()); d['axes'] = [-float(x) for x in d['axes']]; data = json.dumps(d).encode()
                except (ValueError, KeyError, TypeError): continue
            accept(data, 'udp:' + addr[0])

    def _tick(self):
        self._drain_udp()
        msg = Joy(); msg.header.stamp = self.get_clock().now().to_msg(); msg.header.frame_id = 'web_pilot'
        with S['lock']: cmd, age, src, trim = S['cmd'], time.time() - S['cmd_t'], S['src'], S['trim']
        if cmd is not None and age <= self.timeout:
            axes = list(cmd[0])
            axes[3] = max(-1.0, min(1.0, axes[3] + trim))        # steering trim (ROS: + = left)
            msg.axes, msg.buttons = axes, cmd[1]
            if not self.link_up: self.link_up = True; self.get_logger().info(f'commands from {src}')
        else:
            msg.axes, msg.buttons = [0.0] * N_AXES, [0] * N_BUTTONS
            if self.link_up: self.link_up = False; self.get_logger().warn('no commands: neutral (car stops)')
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args); n = WebPilot()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
