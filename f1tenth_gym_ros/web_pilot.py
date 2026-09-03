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
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import pilot_autonomy as PA

N_AXES, N_BUTTONS = 6, 11
TRIM_FILE = os.path.expanduser('~/.atlascar_trim.json')   # steering trim survives restarts
S = {'jpg': None, 'jpg_t': 0.0, 'cmd': None, 'cmd_t': 0.0, 'src': '-', 'v': None, 'rx': 0, 'scan': None,
     'trim': 0.0, 'video': {'width': 480, 'quality': 60, 'fps': 15.0},
     'seen': {}, 'odom_topic': '/pf/pose/odom', 'lock': threading.Lock()}
SUP = PA.Supervisor()          # raceline_mpc process + its stop conditions
JOB = PA.Job()                 # track conversion / raceline optimization
NODE = [None]                  # the running WebPilot, for handlers that need ROS


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
 #panel{position:fixed;left:12px;top:46px;width:330px;max-height:calc(100vh - 120px);overflow:auto;background:rgba(0,0,0,.72);border-radius:8px;padding:10px 12px}
 #panel h3{margin:2px 0 8px;font-size:13px;color:#9cf;text-transform:none}
 #panel section{border-top:1px solid #333;padding-top:8px;margin-top:8px}
 #panel label{display:block;margin:4px 0;font-size:12px}
 #panel input[type=text],#panel input[type=number],#panel select{width:150px;background:#111;color:#ddd;border:1px solid #444;border-radius:4px;padding:2px 4px}
 button{background:#234;color:#dfe;border:1px solid #567;border-radius:5px;padding:4px 10px;cursor:pointer}
 button:hover{background:#356}
 #engage{background:#264;border-color:#4a7}
 #estop{background:#722;border-color:#c55;font-weight:700}
 .chk{font-size:12px;margin:2px 0}.chk b{display:inline-block;width:14px}
 .ok{color:#6d9}.bad{color:#e87}
 #joblog{white-space:pre-wrap;font:11px ui-monospace,Menlo,monospace;color:#9c9;max-height:130px;overflow:auto;background:#0a0a0a;padding:4px;border-radius:4px}
 #tprev{max-width:100%;border-radius:4px;margin-top:6px;display:none}
 #autobadge{position:fixed;left:50%;transform:translateX(-50%);top:44px;background:#2a7;color:#031;font-weight:700;padding:4px 14px;border-radius:6px;display:none}
</style></head><body>
<div id="autobadge">SELF-DRIVING &mdash; press Space or Esc to stop</div>
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
<div id="help">W/S throttle &nbsp; A/D steer &nbsp; Q/E trim (car pulls right &rarr; press Q) &nbsp; (keys held = armed; release = stop) &nbsp;|&nbsp; gamepad: hold LB, left stick throttle, right stick steer &nbsp;|&nbsp; <a href="#" id="togglepanel" style="color:#9cf">panel</a></div>

<div id="panel">
<h3>Self-driving (raceline)</h3>
<label>raceline <select id="rl"></select></label>
<label>pose topic <select id="odom">
  <option value="/pf/pose/odom">/pf/pose/odom (localization)</option>
  <option value="/ekf/odom">/ekf/odom (dead reckoning)</option>
  <option value="/vesc/odom">/vesc/odom (wheel only)</option></select></label>
<label>speed cap <input id="vs" type="range" min="0.1" max="1" step="0.05" value="0.3"> <span id="vsv">30%</span></label>
<div id="checks"></div>
<div style="margin-top:6px">
  <button id="engage">ENGAGE</button>
  <button id="estop">STOP</button>
  <span id="autosecs" style="font-size:12px;color:#9c9"></span>
</div>
<div style="font-size:11px;color:#999;margin-top:6px">Holding a key or the gamepad dead-man always
overrides the policy (the mux gives teleop priority). Closing this tab stops the car.</div>

<section>
<h3>Track picture &rarr; raceline</h3>
<label>name <input id="tname" type="text" placeholder="hall_loop"></label>
<label>picture <input id="tfile" type="file" accept="image/*"></label>
<label>kind <select id="tmode">
  <option value="photo">photo / drawing (dark ink = wall)</option>
  <option value="map">occupancy map (SLAM .pgm/.png)</option></select></label>
<label>lane width <input id="twidth" type="number" step="0.05" value="1.0"> m (sets the scale)</label>
<div style="margin-top:6px">
  <button id="tup">Upload &amp; convert</button>
  <button id="tbuild">Build raceline</button>
</div>
<img id="tprev">
<div id="joblog"></div>
</section>
</div>
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

// ── self-driving ────────────────────────────────────────────────────────────
let auto=false;
$('togglepanel').onclick=e=>{e.preventDefault();const p=$('panel');p.style.display=p.style.display==='none'?'block':'none';};
$('vs').oninput=()=>{$('vsv').textContent=Math.round($('vs').value*100)+'%';};
function renderAuto(s){
  auto=s.engaged;
  $('autobadge').style.display=auto?'block':'none';
  $('autosecs').textContent=auto?(s.secs+' s  '+(s.params.raceline||'')):(s.last_stop||'');
  $('checks').innerHTML=(s.checks||[]).map(c=>
    `<div class="chk"><b class="${c.ok?'ok':'bad'}">${c.ok?'✓':'✗'}</b>${c.name}: <span style="color:#999">${c.detail}</span></div>`).join('');
  const sel=$('rl'), cur=sel.value;
  const opts=(s.racelines||[]).map(r=>`<option ${r===cur?'selected':''}>${r}</option>`).join('');
  if(sel.dataset.n!=String((s.racelines||[]).length)){sel.innerHTML=opts;sel.dataset.n=String((s.racelines||[]).length);}
  if(s.job&&s.job.log&&s.job.log.length){$('joblog').textContent=s.job.log.join('\n');$('joblog').scrollTop=1e6;}
}
function pollAuto(){
  const q=auto?'/auto/hb':'/auto?raceline='+encodeURIComponent($('rl').value||'');
  fetch(q,{method:auto?'POST':'GET'}).then(r=>r.json()).then(renderAuto).catch(()=>{});
}
setInterval(pollAuto,500);
$('engage').onclick=()=>{
  if(!confirm('Engage self-driving? The car will move on its own.\nSpace or Esc stops it; holding a key overrides it.'))return;
  fetch('/auto/engage',{method:'POST',body:JSON.stringify({raceline:$('rl').value,v_scale:parseFloat($('vs').value)})})
   .then(r=>r.json()).then(s=>{if(s.error){alert(s.error+(s.checks?'\n'+s.checks.filter(c=>!c.ok).map(c=>'- '+c.name+': '+c.detail).join('\n'):''));}else renderAuto(s);});
};
$('estop').onclick=()=>fetch('/auto/stop',{method:'POST'}).then(r=>r.json()).then(renderAuto);
$('odom').onchange=()=>fetch('/auto/odom',{method:'POST',body:JSON.stringify({odom_topic:$('odom').value})}).then(r=>r.json()).then(renderAuto);
addEventListener('keydown',e=>{if((e.key===' '||e.key==='Escape')&&auto){e.preventDefault();$('estop').onclick();}});

// ── track picture -> map -> raceline ────────────────────────────────────────
$('tup').onclick=()=>{
  const f=$('tfile').files[0], n=$('tname').value.trim();
  if(!f||!n){alert('pick a picture and give it a name');return;}
  const fd=new FormData(); fd.append('name',n); fd.append('mode',$('tmode').value);
  fd.append('track_width',$('twidth').value); fd.append('file',f);
  $('joblog').textContent='uploading '+Math.round(f.size/1024)+' kB...';
  fetch('/track/upload',{method:'POST',body:fd}).then(r=>r.json()).then(s=>{
    if(s.error){$('joblog').textContent=s.error;return;}
    setTimeout(()=>{const i=$('tprev');i.src='/preview?kind=map&name='+encodeURIComponent(n)+'&t='+Date.now();i.style.display='block';},2500);
  });
};
$('tbuild').onclick=()=>{
  const n=$('tname').value.trim(); if(!n){alert('name?');return;}
  $('joblog').textContent='optimizing raceline (this takes a minute)...';
  fetch('/track/build',{method:'POST',body:JSON.stringify({name:n})}).then(r=>r.json()).then(s=>{
    if(s.error)$('joblog').textContent=s.error;
    else setTimeout(()=>{const i=$('tprev');i.src='/preview?kind=raceline&name='+encodeURIComponent(n)+'&t='+Date.now();i.style.display='block';},60000);
  });
};
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
        u = urlparse(self.path)
        if u.path == '/auto':
            self._json(auto_status(parse_qs(u.query).get('raceline', [None])[0])); return
        if u.path == '/preview':                                  # map preview / raceline overlay PNG
            name = PA.safe_name(parse_qs(u.query).get('name', [''])[0])
            kind = parse_qs(u.query).get('kind', ['map'])[0]
            path = (os.path.join(PA.MAPS, f'{name}_preview.png') if kind == 'map'
                    else os.path.join(PA.RACELINES, f'{name}_auto_overlay.png'))
            if not name or not os.path.isfile(path):
                self.send_response(404); self.send_header('Content-Length', '0'); self.end_headers(); return
            b = open(path, 'rb').read()
            self.send_response(200); self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b); return
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

        # ── autonomy ────────────────────────────────────────────────────────────
        if u.path == '/auto/hb':
            SUP.heartbeat(); self._json(auto_status()); return
        if u.path == '/auto/stop':
            SUP.stop('stop button'); self._json(auto_status()); return
        if u.path == '/auto/odom':
            try: topic = str(json.loads(body.decode())['odom_topic'])[:80]
            except (ValueError, KeyError, TypeError): self._json({'error': 'bad topic'}, 400); return
            NODE[0]._watch_odom(topic); self._json(auto_status()); return
        if u.path == '/auto/engage':
            try: d = json.loads(body.decode())
            except ValueError: self._json({'error': 'bad json'}, 400); return
            rl = os.path.basename(str(d.get('raceline', '')))
            with S['lock']: odom = S['odom_topic']
            ok, checks = SUP.preflight(ages(), rl, odom)
            if not ok and not d.get('override'):
                self._json({'error': 'preflight failed', 'checks': checks}, 409); return
            SUP.heartbeat()
            ok, msg = SUP.engage(rl, d.get('v_scale', 0.3), odom, extra=d.get('extra') or {})
            self._json(auto_status(rl) if ok else {'error': msg}, 200 if ok else 400); return

        # ── track pictures ──────────────────────────────────────────────────────
        if u.path == '/track/upload':
            fields, filename, data = PA.parse_multipart(body, self.headers.get('Content-Type'))
            name = PA.safe_name(fields.get('name', ''))
            if not name or not data:
                self._json({'error': 'need a name (letters/digits/_/-) and an image file'}, 400); return
            ext = os.path.splitext(filename or '')[1].lower() or '.jpg'
            if ext not in ('.jpg', '.jpeg', '.png', '.pgm', '.bmp', '.webp'):
                self._json({'error': f'unsupported image type {ext}'}, 400); return
            path = PA.save_upload(data, name, ext)
            cmd = PA.convert_cmd(path, name, fields.get('mode', 'photo'),
                                 fields.get('track_width', '1.0'), fields.get('resolution') or None)
            ok, msg = JOB.start(cmd, f'convert {name}')
            self._json({'started': ok, 'msg': msg, 'name': name, 'bytes': len(data)},
                       200 if ok else 409); return
        if u.path == '/track/build':
            try: d = json.loads(body.decode())
            except ValueError: d = {}
            name = PA.safe_name(d.get('name', ''))
            if not name or not os.path.isfile(os.path.join(PA.MAPS, f'{name}.yaml')):
                self._json({'error': 'no such map; convert a picture first'}, 400); return
            ok, msg = JOB.start(PA.build_cmd(name, d.get('a_lat', 6.5), d.get('v_max', 7.0),
                                             d.get('margin', 0.2)), f'raceline {name}')
            self._json({'started': ok, 'msg': msg}, 200 if ok else 409); return
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
        return {'link': age is not None and age < 0.25, 'age_ms': None if age is None else round(age * 1000), 'src': S['src'], 'v': S['v'], 'auto': SUP.engaged()}


def ages():
    """Seconds since the last message on each preflight topic (None = never seen)."""
    now = time.time()
    with S['lock']: seen, volts = dict(S['seen']), S['v']
    a = {k: (None if t is None else now - t) for k, t in seen.items()}
    a['volts'] = volts
    return a


def auto_status(raceline=None):
    st = SUP.status()
    with S['lock']: odom = S['odom_topic']
    rl = raceline or st['params'].get('raceline', '')
    ok, checks = SUP.preflight(ages(), rl, odom)
    st.update({'checks': checks, 'ready': ok, 'odom_topic': odom, 'job': JOB.status(), **PA.list_tracks()})
    return st


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
        # Explicit "hold at zero" on the teleop topic while nobody is driving and no policy
        # is engaged. joy_teleop's deadman-less `default` block used to do this, but it also
        # masked autonomy in the mux (teleop outranks navigation), so it was removed and the
        # job moved here where it can be suppressed while self-driving.
        self.declare_parameter('teleop_topic', '/teleop')
        self.declare_parameter('hold_zero_teleop', True)
        self.hold_zero = bool(p('hold_zero_teleop'))
        self.teleop_pub = self.create_publisher(AckermannDriveStamped, p('teleop_topic'), 10)
        self.last_hold = 0.0
        self.create_subscription(Image, p('image_topic'), self._img, qos_profile_sensor_data)
        self.declare_parameter('scan_topic', '/scan'); self.last_scan = 0.0
        self.create_subscription(LaserScan, p('scan_topic'), self._scan, qos_profile_sensor_data)
        try:
            from vesc_msgs.msg import VescStateStamped
            self.create_subscription(VescStateStamped, '/sensors/core', self._core, 10)
        except Exception: pass
        # preflight: watch the pose topic autonomy will use (switchable from the page)
        self.declare_parameter('odom_topic', '/pf/pose/odom')
        with S['lock']: S['odom_topic'] = p('odom_topic')
        self._odom_sub = None
        self._watch_odom(p('odom_topic'))
        NODE[0] = self
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', int(p('udp_port')))); self.sock.setblocking(False)
        srv = ThreadingHTTPServer(('0.0.0.0', int(p('http_port'))), H); srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.create_timer(1.0 / float(p('publish_hz')), self._tick)
        self.get_logger().info(f"web pilot: http://<car-ip>:{int(p('http_port'))}/  (UDP :{int(p('udp_port'))} also accepted); watchdog {self.timeout*1000:.0f} ms")

    def _core(self, m):
        with S['lock']:
            S['v'] = round(float(m.state.voltage_input), 2); S['seen']['core'] = time.time()

    def _watch_odom(self, topic):
        """(Re)subscribe to the map-frame pose topic used for the preflight check."""
        if self._odom_sub is not None:
            self.destroy_subscription(self._odom_sub); self._odom_sub = None
        with S['lock']:
            S['odom_topic'] = topic; S['seen'].pop('odom', None)
        if topic:
            self._odom_sub = self.create_subscription(
                Odometry, topic, lambda m: S['seen'].__setitem__('odom', time.time()), 10)

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
        with S['lock']: S['seen']['scan'] = now
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
        was = SUP.engaged(); SUP.tick()                          # heartbeat / process watchdog
        if was and not SUP.engaged():
            self.get_logger().warn(f'autonomy stopped: {SUP.last_stop}')
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
        # brake-to-zero on the teleop topic, but only when a remote pilot has been present,
        # nobody is holding the dead-man, and no policy is driving
        armed = bool(msg.buttons[4]) if len(msg.buttons) > 4 else False
        now = time.time()
        if (self.hold_zero and cmd is not None and not armed and not SUP.engaged()
                and now - self.last_hold >= 0.05):
            self.last_hold = now
            z = AckermannDriveStamped(); z.header.stamp = self.get_clock().now().to_msg()
            self.teleop_pub.publish(z)


def main(args=None):
    rclpy.init(args=args); n = WebPilot()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    SUP.stop('web_pilot shutting down')
    n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
