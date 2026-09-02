#!/usr/bin/env python3
"""UDP gamepad bridge: laptop pad -> /joy on the car, with a dead-man watchdog.

The laptop runs tools/remote_pilot.py, which reads a gamepad with pygame and sends
small JSON datagrams at 50 Hz:  {"seq": n, "t": unix_time, "axes": [6], "buttons": [11]}
in the Logitech F310 (XInput) layout, i.e. exactly what ROS `joy` would publish if the
same pad were plugged into the Jetson:
    axes    0 LX  1 LY  2 LT  3 RX  4 RY  5 RT        buttons 4 LB (dead-man)  5 RB
This node republishes them as sensor_msgs/Joy at 50 Hz, so the unmodified F1TENTH
teleop chain (joy_teleop -> ackermann_mux -> ackermann_to_vesc -> vesc_driver) drives the
car. Launch the manual stack with F1TENTH_CONTROLLER=f310 so joy_teleop uses the F310
profile regardless of what is plugged into the Jetson.

Safety: if no packet arrives for `timeout` seconds (default 0.25 s) the node publishes a
neutral Joy (all axes 0, all buttons 0), which releases the dead-man and makes
joy_teleop command zero speed; the VESC's own 1 s command timeout is the last resort.
Every `status_period` seconds a status datagram is sent back to the last sender so the
laptop can display link age and, when available, battery voltage.
"""
import json, socket, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

N_AXES, N_BUTTONS = 6, 11


class RemoteJoyBridge(Node):
    def __init__(self):
        super().__init__('remote_joy_bridge')
        self.declare_parameter('port', 5005)
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('timeout', 0.25)
        self.declare_parameter('publish_hz', 50.0)
        self.declare_parameter('status_period', 0.5)
        self.declare_parameter('invert_axes', True)   # SDL -> ROS joy sign convention
        p = lambda n: self.get_parameter(n).value
        self.timeout = float(p('timeout'))
        self.invert = bool(p('invert_axes'))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', int(p('port'))))
        self.sock.setblocking(False)
        self.pub = self.create_publisher(Joy, p('joy_topic'), 10)
        self.last = None          # (recv_time, axes, buttons, seq)
        self.peer = None
        self.rx_count = 0; self.rx_window_t = time.time(); self.rx_hz = 0.0
        self.link_up = False
        self.voltage = None
        self._try_vesc_telemetry()
        self.create_timer(1.0 / float(p('publish_hz')), self._tick)
        self.create_timer(float(p('status_period')), self._status)
        self.get_logger().info(f"listening for the laptop pad on UDP :{int(p('port'))}, "
                               f"publishing {p('joy_topic')} at {float(p('publish_hz')):.0f} Hz, "
                               f"watchdog {self.timeout*1000:.0f} ms")

    def _try_vesc_telemetry(self):
        """Battery voltage for the laptop display, if the manual stack's VESC driver is up."""
        try:
            from vesc_msgs.msg import VescStateStamped
            self.create_subscription(VescStateStamped, '/sensors/core',
                                     lambda m: setattr(self, 'voltage', float(m.state.voltage_input)), 10)
        except Exception:
            pass

    def _drain(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(2048)
            except BlockingIOError:
                return
            except OSError:
                return
            try:
                d = json.loads(data.decode())
                # pygame/SDL reports right = +1, down = +1 and triggers at rest = -1;
                # ROS joy reports the opposite sign on every axis (left/up = +1, triggers
                # rest = +1). Negate so joy_teleop sees exactly what a local pad gives.
                sign = -1.0 if self.invert else 1.0
                axes = [sign * float(x) for x in d['axes']][:N_AXES]
                btns = [int(b) for b in d['buttons']][:N_BUTTONS]
            except (ValueError, KeyError, TypeError):
                continue
            axes += [0.0] * (N_AXES - len(axes)); btns += [0] * (N_BUTTONS - len(btns))
            self.last = (time.time(), axes, btns, int(d.get('seq', 0)))
            self.peer = addr; self.rx_count += 1

    def _tick(self):
        self._drain()
        now = time.time()
        msg = Joy(); msg.header.stamp = self.get_clock().now().to_msg(); msg.header.frame_id = 'remote_pad'
        if self.last and now - self.last[0] <= self.timeout:
            msg.axes, msg.buttons = self.last[1], self.last[2]
            if not self.link_up:
                self.link_up = True; self.get_logger().info(f'link up from {self.peer[0]}')
        else:
            msg.axes, msg.buttons = [0.0] * N_AXES, [0] * N_BUTTONS          # neutral: dead-man released
            if self.link_up:
                self.link_up = False; self.get_logger().warn('link lost: publishing neutral (car stops)')
        self.pub.publish(msg)
        if now - self.rx_window_t >= 1.0:
            self.rx_hz = self.rx_count / (now - self.rx_window_t); self.rx_count = 0; self.rx_window_t = now

    def _status(self):
        if not self.peer:
            return
        age_ms = (time.time() - self.last[0]) * 1000 if self.last else None
        st = {'rx_hz': round(self.rx_hz, 1), 'age_ms': None if age_ms is None else round(age_ms),
              'link': self.link_up, 'v': None if self.voltage is None else round(self.voltage, 2)}
        try:
            self.sock.sendto(json.dumps(st).encode(), self.peer)
        except OSError:
            pass


def main(args=None):
    rclpy.init(args=args)
    n = RemoteJoyBridge()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
