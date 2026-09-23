"""
Unified drive node — one /drive endpoint, two interchangeable backends.
=======================================================================

Subscribes to `AckermannDriveStamped` and actuates the car through whichever
hardware path it finds at startup:

  backend "pca9685"  I2C PWM board -> VESC PPM input (+ optional servo ch)
  backend "vesc"     direct VESC UART -> SET_RPM / SET_SERVO_POS,
                     plus free telemetry: GET_VALUES -> /vesc/odom (wheel
                     speed for the particle filter / MPC speed state)

With `backend: auto` (default) the node probes both: reads the PCA9685's
MODE1 register over I2C, and asks the serial port for the VESC firmware
version.  Whichever answers wins (PCA9685 first — if you wired the PWM board
you meant to use it); the same launch file therefore runs unchanged on either
wiring.  Set `backend` explicitly to skip probing.

Safety on both paths:
  - arming hold (neutral for `arm_time` s before commands are accepted),
  - command watchdog (`cmd_timeout` s without /drive -> neutral / zero
    current; the VESC's own app timeout is a second line of defence),
  - neutral on clean shutdown.

Pure pulse maths lives in pca9685.py, the UART protocol in vesc_protocol.py —
both unit-tested without hardware (tests/test_hardware.py).

Run:
    ros2 run f1tenth_gym_ros drive_node --ros-args --params-file config/hardware.yaml
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pca9685 as pca
import vesc_protocol as vp


def bounded_command(speed, steer, cfg):
    """Reject corrupt inputs before either actuator is written; bound valid ones."""
    speed, steer = float(speed), float(steer)
    if not math.isfinite(speed) or not math.isfinite(steer):
        raise ValueError('drive speed and steering must be finite')
    max_speed, max_steer = float(cfg['max_speed']), float(cfg['max_steer'])
    if not all(math.isfinite(v) and v > 0 for v in (max_speed, max_steer)):
        raise ValueError('max_speed and max_steer must be finite and positive')
    return (max(-max_speed, min(max_speed, speed)),
            max(-max_steer, min(max_steer, steer)))


# ─────────────────────────────────────────────────────────────────────────────
# Backends — same interface, picked at startup
# ─────────────────────────────────────────────────────────────────────────────

class PCA9685Backend:
    """Throttle pulses into the VESC PPM input, steering on a second channel."""

    name = 'pca9685'
    has_telemetry = False

    def __init__(self, bus, cfg):
        self.cfg = cfg
        self.dev = pca.PCA9685(bus, cfg['i2c_address'], cfg['pwm_hz'])
        self.ch_thr = int(cfg['throttle_channel'])
        self.ch_str = int(cfg['steer_channel'])

    def command(self, speed, steer):
        speed, steer = bounded_command(speed, steer, self.cfg)
        c = self.cfg
        self.dev.set_pulse_us(self.ch_thr, pca.speed_to_us(
            speed, c['max_speed'], c['neutral_us'],
            c['full_fwd_us'], c['full_rev_us']))
        if self.ch_str >= 0:
            self.dev.set_pulse_us(self.ch_str, pca.steer_to_us(
                steer, c['max_steer'], c['steer_center_us'],
                c['steer_half_range_us'], c['steer_invert'], c['steer_trim_us']))

    def neutral(self):
        self.dev.set_pulse_us(self.ch_thr, self.cfg['neutral_us'])
        if self.ch_str >= 0:
            self.dev.set_pulse_us(
                self.ch_str,
                self.cfg['steer_center_us'] + self.cfg['steer_trim_us'])

    def stop(self):
        self.neutral()
        time.sleep(0.05)
        self.dev.set_off(self.ch_thr)
        if self.ch_str >= 0:
            self.dev.set_off(self.ch_str)


class VescSerialBackend:
    """Direct UART: SET_RPM (closed-loop speed) or SET_CURRENT (direct torque),
    selected by `control_mode`, plus the VESC's own servo header."""

    name = 'vesc'
    has_telemetry = True

    def __init__(self, ser, cfg):
        self.ser = ser
        self.cfg = cfg
        self.erpm_gain = float(cfg['erpm_gain'])
        # current-control (torque) parameters — mirror f1tenth_stack vesc.yaml so the
        # self-driving path launches the same way the manual joystick path does.
        # Optional with defaults: control_mode 'speed' (the original behaviour) needs none
        # of these, and a config written before current mode existed must still load.
        self.control_mode = str(cfg.get('control_mode', 'speed')).lower()
        self.max_speed = float(cfg['max_speed'])
        self.max_current = float(cfg.get('max_current', 45.0))
        self.min_current = float(cfg.get('min_current', 10.0))
        self.brake_current = float(cfg.get('brake_current', 15.0))
        self.current_deadband = float(cfg.get('current_deadband', 0.05))
        self.parser = vp.PacketParser()

    def _write_throttle(self, speed):
        if self.control_mode == 'current':
            # Map the commanded speed (planner/MPC output, m/s) to motor current so the car
            # pulls off the line immediately instead of waiting for the VESC's internal RPM
            # loop to wind current up past stiction (the ~30-35 A dead zone). A min_current
            # feed-forward guarantees torque the instant a non-zero speed is requested.
            throttle = speed / self.max_speed if self.max_speed > 0.0 else 0.0
            throttle = max(-1.0, min(1.0, throttle))
            if abs(throttle) < self.current_deadband:
                self.ser.write(vp.pkt_set_current_brake(self.brake_current))
            else:
                mag = self.min_current + abs(throttle) * (self.max_current - self.min_current)
                self.ser.write(vp.pkt_set_current(math.copysign(mag, throttle)))
        else:
            self.ser.write(vp.pkt_set_rpm(speed * self.erpm_gain))

    def command(self, speed, steer):
        speed, steer = bounded_command(speed, steer, self.cfg)
        self._write_throttle(speed)
        c = self.cfg
        frac = max(-1.0, min(1.0, float(steer) / float(c['max_steer'])))
        if c['steer_invert']:
            frac = -frac
        # Mirror the PCA9685 pulse map (centre + trim + frac·half-range) on the
        # VESC's [0, 1] servo scale, treating it as a 1000 us span.  Using the
        # configured half-range — not the full ±0.5 span — leaves headroom so
        # trim shifts the centre instead of clipping one side's full lock
        # (the old `0.5 + 0.5·frac + trim` saturated full-left whenever
        # trim > 0, giving asymmetric steering on the UART backend only).
        pos = 0.5 + (float(c['steer_trim_us'])
                     + frac * float(c['steer_half_range_us'])) / 1000.0
        self.ser.write(vp.pkt_set_servo_pos(pos))

    def neutral(self):
        self.ser.write(vp.pkt_set_current(0.0))

    def stop(self):
        self.neutral()

    def poll_telemetry(self):
        """Request GET_VALUES, drain the port; returns latest dict or None."""
        self.ser.write(vp.pkt_request(vp.COMM_GET_VALUES))
        values = None
        waiting = self.ser.in_waiting
        if waiting:
            for payload in self.parser.feed(self.ser.read(waiting)):
                parsed = vp.parse_values(payload)
                if parsed is not None:
                    values = parsed
        return values


# ─────────────────────────────────────────────────────────────────────────────
# Detection — probe I2C for a PCA9685, the serial port for a VESC
# ─────────────────────────────────────────────────────────────────────────────

def probe_pca9685(cfg, log):
    try:
        bus = pca.open_i2c(cfg['i2c_bus'])
        bus.read_byte_data(int(cfg['i2c_address']), pca.PCA9685.MODE1)
        return bus
    except Exception as e:
        log(f"no PCA9685 on i2c-{cfg['i2c_bus']} @0x{int(cfg['i2c_address']):02x}: {e}")
        return None


def probe_vesc(cfg, log):
    try:
        import serial
        ser = serial.Serial(cfg['serial_port'], int(cfg['serial_baud']),
                            timeout=0.1, write_timeout=0.1)
        parser = vp.PacketParser()
        for _ in range(3):                       # fw-version handshake
            ser.write(vp.pkt_request(vp.COMM_FW_VERSION))
            time.sleep(0.1)
            for payload in parser.feed(ser.read(ser.in_waiting or 1)):
                if payload and payload[0] == vp.COMM_FW_VERSION:
                    return ser
        ser.close()
        log(f"no VESC reply on {cfg['serial_port']}")
    except Exception as e:
        log(f"no VESC on {cfg['serial_port']}: {e}")
    return None


def pick_backend(prefer, cfg, log):
    """prefer in ('auto', 'pca9685', 'vesc') -> backend instance or None."""
    if prefer in ('auto', 'pca9685'):
        bus = probe_pca9685(cfg, log)
        if bus is not None:
            return PCA9685Backend(bus, cfg)
        if prefer == 'pca9685':
            return None
    if prefer in ('auto', 'vesc'):
        ser = probe_vesc(cfg, log)
        if ser is not None:
            return VescSerialBackend(ser, cfg)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ROS node
# ─────────────────────────────────────────────────────────────────────────────

def _make_node():
    import rclpy
    from rclpy.node import Node
    from ackermann_msgs.msg import AckermannDriveStamped
    from nav_msgs.msg import Odometry

    class DriveNode(Node):
        def __init__(self):
            super().__init__('drive_node')
            self.declare_parameter('drive_topic', '/drive')
            self.declare_parameter('backend', 'auto')   # auto | pca9685 | vesc
            # shared actuation limits / calibration
            self.declare_parameter('max_speed', 7.0)    # m/s at full throttle
            self.declare_parameter('max_steer', 0.41)   # rad at full servo throw
            self.declare_parameter('steer_invert', False)
            self.declare_parameter('steer_trim_us', 0.0)
            self.declare_parameter('cmd_timeout', 0.5)
            self.declare_parameter('arm_time', 2.0)
            # pca9685 path
            self.declare_parameter('i2c_bus', 1)        # check `i2cdetect -l`
            self.declare_parameter('i2c_address', 0x40)
            self.declare_parameter('pwm_hz', 50.0)      # VESC PPM is happy <=200
            self.declare_parameter('throttle_channel', 0)
            self.declare_parameter('steer_channel', 1)  # -1 = no steering servo
            self.declare_parameter('neutral_us', 1500.0)
            self.declare_parameter('full_fwd_us', 2000.0)
            self.declare_parameter('full_rev_us', 1000.0)
            self.declare_parameter('steer_center_us', 1500.0)
            self.declare_parameter('steer_half_range_us', 400.0)
            # vesc-uart path
            self.declare_parameter('serial_port', '/dev/ttyACM0')
            self.declare_parameter('serial_baud', 115200)
            self.declare_parameter('erpm_gain', 4614.0)  # erpm per m/s
            # vesc-uart motor command mode (mirrors f1tenth_stack/config/vesc.yaml)
            self.declare_parameter('core_topic', '/sensors/core')
            self.declare_parameter('control_mode', 'speed')   # 'speed' (SET_RPM) | 'current'
            self.declare_parameter('max_current', 45.0)       # A at full-speed command
            self.declare_parameter('min_current', 10.0)       # A feed-forward to break stiction
            self.declare_parameter('brake_current', 15.0)     # A brake when ~zero speed requested
            self.declare_parameter('current_deadband', 0.05)  # |throttle| fraction treated as stop
            self.declare_parameter('odom_topic', '/vesc/odom')
            self.declare_parameter('odom_frame', 'odom')
            self.declare_parameter('base_frame', 'base_link')
            self.declare_parameter('telemetry_hz', 20.0)

            cfg = {n: self.get_parameter(n).value for n in (
                'max_speed', 'max_steer', 'steer_invert', 'steer_trim_us',
                'i2c_bus', 'i2c_address', 'pwm_hz', 'throttle_channel',
                'steer_channel', 'neutral_us', 'full_fwd_us', 'full_rev_us',
                'steer_center_us', 'steer_half_range_us',
                'serial_port', 'serial_baud', 'erpm_gain',
                'control_mode', 'max_current', 'min_current',
                'brake_current', 'current_deadband')}
            # Validate before opening hardware: invalid limits/timeouts must not
            # leave an already-open actuator outside the shutdown path.
            bounded_command(0.0, 0.0, cfg)
            self.timeout = float(self.get_parameter('cmd_timeout').value)
            arm_time = float(self.get_parameter('arm_time').value)
            if not math.isfinite(self.timeout) or self.timeout <= 0:
                raise ValueError('cmd_timeout must be finite and positive')
            if not math.isfinite(arm_time) or arm_time < 0:
                raise ValueError('arm_time must be finite and nonnegative')
            prefer = self.get_parameter('backend').value
            self.backend = pick_backend(
                prefer, cfg, lambda m: self.get_logger().info(m))
            if self.backend is None:
                raise RuntimeError(
                    f"no actuation hardware found (backend={prefer}) — "
                    f"checked PCA9685 on i2c-{cfg['i2c_bus']} and VESC on "
                    f"{cfg['serial_port']}")
            self.get_logger().info(f'backend: {self.backend.name}')

            self.arm_until = time.monotonic() + arm_time
            # Arm the watchdog from startup: initialising to 0.0 disabled it
            # (the `> 0.0` guard) until the first /drive message ever arrived.
            self.last_cmd = time.monotonic()
            self._wd_warned = False
            self.backend.neutral()
            self.create_subscription(
                AckermannDriveStamped,
                self.get_parameter('drive_topic').value, self._drive_cb, 1)
            self.create_timer(0.04, self._watchdog)

            if self.backend.has_telemetry:
                self.odom_pub = self.create_publisher(
                    Odometry, self.get_parameter('odom_topic').value, 10)
                # Full VESC state on /sensors/core, the topic f1tenth_stack's
                # vesc_driver publishes and that web_pilot / the dashboards read
                # for pack voltage, FET temp and fault codes. We cannot run
                # vesc_driver alongside this node (one owner per serial port),
                # so we republish the GET_VALUES frame we are already polling.
                self.core_pub = None
                try:
                    from vesc_msgs.msg import VescStateStamped
                    self._VescStateStamped = VescStateStamped
                    self.core_pub = self.create_publisher(
                        VescStateStamped,
                        self.get_parameter('core_topic').value, 10)
                except Exception as e:
                    self.get_logger().warning(
                        f'vesc_msgs unavailable — no /sensors/core telemetry: {e}')
                self.erpm_gain = float(cfg['erpm_gain'])
                self._telem_n = 0
                self.create_timer(
                    1.0 / float(self.get_parameter('telemetry_hz').value),
                    self._telemetry)

        def _drive_cb(self, msg):
            now = time.monotonic()
            if now < self.arm_until:                   # arming: hold neutral
                return
            try:
                speed, steer = float(msg.drive.speed), float(msg.drive.steering_angle)
                if not math.isfinite(speed) or not math.isfinite(steer):
                    raise ValueError('drive speed and steering must be finite')
                self.backend.command(speed, steer)
            except Exception as e:
                self.get_logger().error(f'actuation command failed: {e}')
                # A partial write may already have applied throttle. Stop now,
                # and leave the watchdog expired so it retries if neutral fails.
                self.last_cmd = float('-inf')
                self._wd_warned = False
                self._watchdog()
                return
            self.last_cmd = now
            self._wd_warned = False

        def _watchdog(self):
            if time.monotonic() - self.last_cmd <= self.timeout:
                return
            try:
                self.backend.neutral()
            except Exception as e:
                # A watchdog that fails silently is no watchdog: the operator
                # must know the car did NOT go neutral.
                if not self._wd_warned:
                    self.get_logger().error(f'watchdog neutral FAILED: {e}')
                    self._wd_warned = True
                return
            if not self._wd_warned:
                self.get_logger().warning(
                    f'no /drive for {self.timeout:.1f}s — neutral')
                self._wd_warned = True

        def _telemetry(self):
            try:
                values = self.backend.poll_telemetry()
            except Exception as e:
                self.get_logger().warning(f'telemetry read failed: {e}')
                return
            if values is None:
                return
            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = self.get_parameter('odom_frame').value
            odom.child_frame_id = self.get_parameter('base_frame').value
            odom.twist.twist.linear.x = values['erpm'] / self.erpm_gain
            self.odom_pub.publish(odom)

            if getattr(self, 'core_pub', None) is not None:
                core = self._VescStateStamped()
                core.header.stamp = odom.header.stamp
                core.header.frame_id = self.get_parameter('base_frame').value
                st = core.state
                st.voltage_input = float(values['v_in'])
                st.temp_fet = float(values['temp_fet'])
                st.temp_motor = float(values['temp_motor'])
                st.current_motor = float(values['current_motor'])
                st.current_input = float(values['current_input'])
                st.duty_cycle = float(values['duty'])
                st.speed = float(values['erpm'])
                st.displacement = int(values['tachometer'])
                st.fault_code = int(values['fault'])
                self.core_pub.publish(core)
            self._telem_n += 1
            if self._telem_n % 200 == 0:                 # ~every 10 s at 20 Hz
                self.get_logger().info(
                    f"vesc: {values['v_in']:.1f}V fet {values['temp_fet']:.0f}C "
                    f"fault {values['fault']}")
            if values['fault']:
                self.get_logger().error(f"VESC FAULT code {values['fault']}")

        def shutdown(self):
            try:
                self.backend.stop()
            except Exception:
                pass

    return rclpy, DriveNode


def main(args=None):
    rclpy, NodeCls = _make_node()
    rclpy.init(args=args)
    node = None
    try:
        node = NodeCls()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
