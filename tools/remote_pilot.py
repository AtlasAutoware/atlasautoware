#!/usr/bin/env python3
"""Remote pilot: gamepad on the laptop -> UDP -> the car's remote_joy_bridge, plus FPV.

    python3 remote_pilot.py                      # car at ubuntu.local (mDNS), pad on the laptop
    python3 remote_pilot.py --host 172.20.10.4   # explicit IP (hostname -I on the Jetson)
    python3 remote_pilot.py --fpv                # also open the camera stream in a window

Controls (Logitech F310, switch on X): hold LB = dead-man, left stick up/down = throttle,
right stick left/right = steering. Release LB or close this program and the car stops
(the bridge publishes neutral after 250 ms without packets).

The packet is the pad's raw SDL layout, which is the same numbering ROS `joy` uses, so
the F1TENTH teleop profile on the car applies unchanged. Needs: pip install pygame
(and opencv-python + numpy for --fpv).
"""
import argparse, json, socket, sys, threading, time, urllib.request

RATE_HZ = 50


def fpv_thread(url, state):
    try:
        import cv2, numpy as np
    except ImportError:
        print('[fpv] pip install opencv-python numpy'); return
    while not state['quit']:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
        except Exception as e:
            print(f'[fpv] connecting to {url}: {e}'); time.sleep(2); continue
        buf = b''
        while not state['quit']:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            a, b = buf.find(b'\xff\xd8'), buf.find(b'\xff\xd9')
            if a != -1 and b != -1 and b > a:
                jpg, buf = buf[a:b + 2], buf[b + 2:]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                s = state
                # overlay: dead-man, throttle/steer bars, link, battery
                col = (60, 200, 60) if s['deadman'] else (40, 40, 220)
                cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
                txt = (f"{'ARMED' if s['deadman'] else 'HOLD LB'}  thr {s['thr']:+.2f}  str {s['str']:+.2f}  "
                       f"tx {s['tx_hz']:.0f}Hz  link {s['age']}  batt {s['batt']}")
                cv2.putText(img, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                cx, cy = w // 2, h - 14
                cv2.line(img, (cx - 60, cy), (cx + 60, cy), (200, 200, 200), 1)
                cv2.circle(img, (int(cx + 60 * s['str']), cy), 5, (60, 200, 255), -1)
                cv2.rectangle(img, (w - 18, h - 10 - int(60 * max(0.0, s['thr']))), (w - 8, h - 10), (60, 200, 60), -1)
                cv2.imshow('FPV  (q to quit)', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    state['quit'] = True
        try:
            resp.close()
        except Exception:
            pass
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='ubuntu.local')
    ap.add_argument('--port', type=int, default=5005)
    ap.add_argument('--fpv', action='store_true', help='open the MJPEG camera stream from the car')
    ap.add_argument('--fpv-port', type=int, default=8080)
    ap.add_argument('--pad', type=int, default=0, help='pygame joystick index')
    a = ap.parse_args()

    try:
        import pygame
    except ImportError:
        sys.exit('pip install pygame')
    pygame.init(); pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        sys.exit('no gamepad found on this laptop (plug in the F310, switch on X)')
    js = pygame.joystick.Joystick(a.pad); js.init()
    print(f'pad: {js.get_name()}  axes {js.get_numaxes()}  buttons {js.get_numbuttons()}')

    try:
        addr = (socket.gethostbyname(a.host), a.port)
    except socket.gaierror:
        sys.exit(f"cannot resolve {a.host}: pass --host <ip from 'hostname -I' on the Jetson>")
    print(f'car: {a.host} = {addr[0]}:{addr[1]}   (hold LB to drive; Ctrl-C stops the car)')
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); sock.setblocking(False)

    state = {'quit': False, 'deadman': False, 'thr': 0.0, 'str': 0.0, 'tx_hz': 0.0, 'age': '-', 'batt': '-'}
    if a.fpv:
        threading.Thread(target=fpv_thread, args=(f'http://{addr[0]}:{a.fpv_port}/stream', state), daemon=True).start()

    seq = 0; t_win = time.time(); n_win = 0; last_print = 0.0; last_status = None
    try:
        while not state['quit']:
            t0 = time.time()
            pygame.event.pump()
            axes = [round(js.get_axis(i), 3) for i in range(min(6, js.get_numaxes()))]
            btns = [int(js.get_button(i)) for i in range(min(11, js.get_numbuttons()))]
            seq += 1
            sock.sendto(json.dumps({'seq': seq, 't': t0, 'axes': axes, 'buttons': btns}).encode(), addr)
            n_win += 1
            try:
                while True:
                    data, _ = sock.recvfrom(1024); last_status = json.loads(data.decode())
            except (BlockingIOError, ValueError):
                pass
            state['deadman'] = bool(btns[4]) if len(btns) > 4 else False
            state['thr'] = -axes[1] if len(axes) > 1 else 0.0        # SDL: stick up is negative -> show + for forward
            state['str'] = axes[3] if len(axes) > 3 else 0.0         # show + for right
            if t0 - t_win >= 1.0:
                state['tx_hz'] = n_win / (t0 - t_win); n_win = 0; t_win = t0
            if last_status:
                state['age'] = f"{last_status.get('age_ms')} ms" if last_status.get('link') else 'DOWN'
                state['batt'] = f"{last_status['v']:.1f} V" if last_status.get('v') else '-'
            if t0 - last_print >= 0.5:
                last_print = t0
                print(f"\r{'ARMED ' if state['deadman'] else 'hold LB'}  thr {state['thr']:+.2f}  str {state['str']:+.2f}  "
                      f"tx {state['tx_hz']:.0f} Hz  car-rx {last_status.get('rx_hz') if last_status else '-'} Hz  "
                      f"link {state['age']}  batt {state['batt']}   ", end='', flush=True)
            time.sleep(max(0.0, 1.0 / RATE_HZ - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        state['quit'] = True
        for _ in range(5):        # explicit neutral so the car stops before the watchdog would
            sock.sendto(json.dumps({'seq': seq + 1, 't': time.time(), 'axes': [0.0] * 6, 'buttons': [0] * 11}).encode(), addr)
            time.sleep(0.02)
        print('\nstopped')


if __name__ == '__main__':
    main()
