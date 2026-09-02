#!/usr/bin/env python3
"""Tiny MJPEG server for FPV: a ROS Image topic -> http://<car>:8080/stream .

Meant for a phone hotspot, so it is frugal: frames are resized to `width` px (default
480), JPEG-encoded at `quality` (default 60) and served at most `fps` (default 15).
That is roughly 1-2 Mbit/s. Any number of viewers can connect (browser, VLC, or
tools/remote_pilot.py --fpv). No dependencies beyond rclpy, numpy and cv2.
"""
import threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

LATEST = {'jpg': None, 't': 0.0, 'lock': threading.Lock()}
PAGE = b"""<html><body style="margin:0;background:#111"><img src="/stream" style="width:100vw;height:auto"></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):            # keep the console quiet
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
            self.wfile.write(PAGE); return
        if self.path != '/stream':
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache'); self.end_headers()
        last_t = 0.0
        try:
            while True:
                with LATEST['lock']:
                    jpg, t = LATEST['jpg'], LATEST['t']
                if jpg is None or t == last_t:
                    time.sleep(0.01); continue
                last_t = t
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                                 str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
        except (BrokenPipeError, ConnectionResetError):
            pass


class MjpegServer(Node):
    def __init__(self):
        super().__init__('mjpeg_server')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('port', 8080)
        self.declare_parameter('width', 480)
        self.declare_parameter('quality', 60)
        self.declare_parameter('fps', 15.0)
        p = lambda n: self.get_parameter(n).value
        self.width, self.q = int(p('width')), int(p('quality'))
        self.min_dt = 1.0 / float(p('fps')); self.last = 0.0
        self.create_subscription(Image, p('image_topic'), self._cb, qos_profile_sensor_data)
        srv = ThreadingHTTPServer(('0.0.0.0', int(p('port'))), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.get_logger().info(f"FPV: http://<car-ip>:{int(p('port'))}/  ({p('image_topic')} -> {self.width}px, q{self.q}, {float(p('fps')):.0f} fps max)")

    def _cb(self, m):
        now = time.time()
        if now - self.last < self.min_dt:
            return
        self.last = now
        if m.encoding not in ('rgb8', 'bgr8'):
            return
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        if m.encoding == 'rgb8':
            a = a[:, :, ::-1]
        if m.width != self.width:
            a = cv2.resize(a, (self.width, int(m.height * self.width / m.width)), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode('.jpg', a, [cv2.IMWRITE_JPEG_QUALITY, self.q])
        if ok:
            with LATEST['lock']:
                LATEST['jpg'], LATEST['t'] = jpg.tobytes(), now


def main(args=None):
    rclpy.init(args=args)
    n = MjpegServer()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
