"""
Hardware diagnostics — what's actually plugged into the car, and is it talking?
==============================================================================

One command that probes every actuator and sensor on the F1TENTH car and prints
a PASS / WARN / FAIL table.  Two layers, so it is useful with or without ROS:

  device layer (always)   open the serial/I2C/USB device and get a real reply
                          - VESC      : UART FW_VERSION + GET_VALUES handshake
                          - RPLidar   : serial port present + descriptor probe
                          - OAK-D     : depthai device enumeration (USB)
                          - PCA9685   : I2C MODE1 read (alt. actuation board)
  ros layer (if up)       scan the live graph and measure publish rates on the
                          sensor/actuation topics (LaserScan, Image, Imu,
                          Odometry, AckermannDrive) — works for either stack
                          (f1tenth_stack manual, or atlasautoware self-drive)
                          because it classifies by message TYPE, not by name.

Steering servo — important: a VESC drives the steering servo OPEN-LOOP through
its PWM servo header (COMM_SET_SERVO_POS).  There is no feedback line, so no
amount of probing can tell you a servo is plugged in.  The only real test is
active: `--servo-sweep` commands a small left/centre/right motion through the
VESC so you can watch the wheels move (motor is held at zero current throughout).

Run:
    python3 hw_diag.py                 # device probes + ROS topic scan
    python3 hw_diag.py --no-ros        # device probes only (no rclpy needed)
    python3 hw_diag.py --servo-sweep   # ALSO wiggle the steering — WHEELS OFF GROUND
    ros2 run f1tenth_gym_ros hw_diag   # same thing, as a ROS entry point
"""

import argparse
import glob
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vesc_protocol as vp
try:
    import pca9685 as pca
except Exception:                       # smbus may be missing off-car
    pca = None

# ── result accounting ────────────────────────────────────────────────────────
PASS, WARN, FAIL = 'PASS', 'WARN', 'FAIL'
_GLYPH = {PASS: '✔', WARN: '⚠', FAIL: '✖'}      # ✔ ⚠ ✖
_results = []


def report(status, name, detail=''):
    _results.append((status, name, detail))
    print(f'  [{_GLYPH[status]}] {status:4}  {name:18} {detail}')


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# ── VESC (UART) ──────────────────────────────────────────────────────────────
VESC_PORTS = ['/dev/sensors/vesc', '/dev/ttyACM0', '/dev/ttyACM1']


def check_vesc(port_override=None):
    port = port_override or first_existing(VESC_PORTS)
    if port is None:
        report(FAIL, 'VESC', f'no serial device at any of {VESC_PORTS}')
        return None
    try:
        import serial
    except Exception as e:
        report(WARN, 'VESC', f'{port} present but pyserial missing ({e})')
        return None
    try:
        ser = serial.Serial(port, 115200, timeout=0.15)
    except Exception as e:
        report(FAIL, 'VESC', f'cannot open {port}: {e}')
        return None

    parser = vp.PacketParser()
    fw_ok = False
    for _ in range(5):                                   # FW handshake
        ser.write(vp.pkt_request(vp.COMM_FW_VERSION))
        time.sleep(0.1)
        for payload in parser.feed(ser.read(ser.in_waiting or 1)):
            if payload and payload[0] == vp.COMM_FW_VERSION:
                fw_ok = True
        if fw_ok:
            break
    if not fw_ok:
        report(FAIL, 'VESC', f'{port} opened but no FW reply (powered? right port?)')
        ser.close()
        return None

    # GET_VALUES gives a live health snapshot, and confirms the firmware layout.
    values = None
    for _ in range(5):
        ser.write(vp.pkt_request(vp.COMM_GET_VALUES))
        time.sleep(0.1)
        for payload in parser.feed(ser.read(ser.in_waiting or 1)):
            parsed = vp.parse_values(payload)
            if parsed is not None:
                values = parsed
        if values:
            break

    if values is None:
        report(WARN, 'VESC', f'{port}: FW ok but no telemetry (GET_VALUES)')
        return ser
    fault = values['fault']
    detail = (f"{port}: {values['v_in']:.1f}V  fet {values['temp_fet']:.0f}C  "
              f"erpm {values['erpm']:.0f}  fault {fault}")
    report(FAIL if fault else PASS, 'VESC', detail)
    # The servo is driven off this same VESC — say so, since it can't be sensed.
    report(WARN, 'Steering servo',
           'open-loop on VESC PWM header — cannot be sensed; use --servo-sweep to verify')
    return ser


def servo_sweep(ser):
    """Active steering test: small symmetric sweep, motor kept at zero current."""
    if ser is None:
        report(FAIL, 'Servo sweep', 'no VESC connection to drive the servo')
        return
    print('\n  Sweeping steering — WATCH THE FRONT WHEELS (motor held at 0 A)…')
    seq = [(0.50, 'centre'), (0.35, 'left'), (0.50, 'centre'),
           (0.65, 'right'), (0.50, 'centre')]
    try:
        for pos, label in seq:
            ser.write(vp.pkt_set_current(0.0))          # keep drive motor quiet
            ser.write(vp.pkt_set_servo_pos(pos))
            print(f'    servo -> {pos:.2f} ({label})')
            time.sleep(0.6)
        report(PASS, 'Servo sweep', 'commands sent — PASS only if the wheels actually moved')
    except Exception as e:
        report(FAIL, 'Servo sweep', f'write failed: {e}')


# ── RPLidar (UART) ───────────────────────────────────────────────────────────
LIDAR_PORTS = ['/dev/sensors/rplidar', '/dev/ttyUSB0', '/dev/ttyUSB1']


def check_lidar(port_override=None):
    port = port_override or first_existing(LIDAR_PORTS)
    if port is None:
        report(FAIL, 'RPLidar', f'no serial device at any of {LIDAR_PORTS}')
        return
    try:
        from rplidar import RPLidar
    except Exception:
        report(WARN, 'RPLidar', f'{port} present but `rplidar` pkg missing — '
                                'device there, run-time check via /scan topic')
        return
    lidar = None
    try:
        lidar = RPLidar(port, baudrate=115200, timeout=1.0)
        info = lidar.get_info()
        health = lidar.get_health()
        report(PASS if health[0].lower() == 'good' else WARN, 'RPLidar',
               f"{port}: {info.get('model','?')} fw{info.get('firmware','?')} "
               f"health={health[0]}")
    except Exception as e:
        report(WARN, 'RPLidar', f'{port}: present but probe failed ({e}); '
                                'try baud 256000 for A3/S1')
    finally:
        if lidar is not None:
            try:
                lidar.stop(); lidar.disconnect()
            except Exception:
                pass


# ── OAK-D camera (USB / depthai) ─────────────────────────────────────────────
def check_camera():
    # Best case: depthai can actually claim the device and read its serial.
    err = None
    try:
        import depthai as dai
        devices = dai.Device.getAllAvailableDevices()
        if devices:
            ids = ', '.join(d.getMxId() for d in devices)
            report(PASS, 'OAK-D camera', f'{len(devices)} device(s) via depthai: {ids}')
            return
    except Exception as e:
        err = e
    # depthai couldn't enumerate — is the camera at least on the USB bus?  This is
    # the common "device present but blocked by udev permissions" case, so prefer
    # a single WARN with the fix over a scary FAIL when the hardware is really there.
    try:
        out = subprocess.check_output(['lsusb'], text=True)
        # 03e7 = Intel/Movidius MyriadX (OAK runtime 2485 / bootloader f63b/f63c)
        hit = [ln for ln in out.splitlines() if '03e7:' in ln.lower()]
    except Exception:
        hit = []
    if hit:
        report(WARN, 'OAK-D camera',
               f'on USB but depthai not bound ({hit[0].strip()}); '
               'install udev rules: '
               'echo \'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"\' | '
               'sudo tee /etc/udev/rules.d/80-movidius.rules && sudo udevadm control --reload')
    elif err is not None:
        report(FAIL, 'OAK-D camera', f'depthai enumeration failed and no USB device: {err}')
    else:
        report(FAIL, 'OAK-D camera', 'no Movidius (03e7:*) device on USB')


# ── PCA9685 (alternate actuation board, I2C) ─────────────────────────────────
def check_pca9685(bus_no=1, addr=0x40):
    if pca is None:
        report(WARN, 'PCA9685', 'smbus not installed — skipping I2C probe (optional board)')
        return
    if not os.path.exists(f'/dev/i2c-{bus_no}'):
        report(WARN, 'PCA9685', f'/dev/i2c-{bus_no} absent — skipping (optional board)')
        return
    try:
        bus = pca.open_i2c(bus_no)
        bus.read_byte_data(addr, pca.PCA9685.MODE1)
        report(PASS, 'PCA9685', f'i2c-{bus_no} @0x{addr:02x} responding')
    except Exception as e:
        report(WARN, 'PCA9685', f'none on i2c-{bus_no} @0x{addr:02x} ({e}) — ok if using VESC UART')


# ── ROS2 graph layer ─────────────────────────────────────────────────────────
# Map message type -> the sensor/actuator it represents.  Classifying by type
# (not topic name) makes this stack-agnostic: it finds /scan or /ego_id/scan,
# /oakd/rgb or /camera/image_raw, etc., the same way.
ROS_TYPES = {
    'sensor_msgs/msg/LaserScan':          ('Lidar /scan',     2.0),
    'sensor_msgs/msg/Image':              ('Camera image',    5.0),
    'sensor_msgs/msg/Imu':                ('IMU',            20.0),
    'nav_msgs/msg/Odometry':              ('Odometry',        5.0),
    'ackermann_msgs/msg/AckermannDriveStamped': ('Drive cmd', 0.0),
}


def check_ros(window=3.0):
    try:
        import rclpy
        from rclpy.node import Node
    except Exception:
        report(WARN, 'ROS graph', 'rclpy not importable — source your ROS workspace first')
        return
    print(f'\n  Scanning ROS graph for {window:.0f}s …')
    rclpy.init()
    node = Node('hw_diag')
    time.sleep(0.5)                                     # let discovery settle
    topics = node.get_topic_names_and_types()
    watched = []                                        # (topic, label, min_hz)
    for name, types in topics:
        for t in types:
            if t in ROS_TYPES:
                label, min_hz = ROS_TYPES[t]
                watched.append((name, label, min_hz, t))

    if not watched:
        report(WARN, 'ROS graph', f'{len(topics)} topics up, none are sensor/drive types '
                                  '(is the stack launched?)')
        node.destroy_node(); rclpy.shutdown()
        return

    # Count messages per topic over the window to get a real rate.
    from rclpy.qos import qos_profile_sensor_data
    counts = {n: 0 for (n, _, _, _) in watched}

    def make_cb(topic):
        def _cb(_msg):
            counts[topic] += 1
        return _cb

    subs = []
    for name, label, min_hz, tstr in watched:
        # resolve the message class dynamically from its "pkg/msg/Type" string
        pkg, _, cls = tstr.split('/')
        mod = __import__(f'{pkg}.msg', fromlist=[cls])
        msg_cls = getattr(mod, cls)
        subs.append(node.create_subscription(
            msg_cls, name, make_cb(name), qos_profile_sensor_data))

    t_end = time.time() + window
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)

    for name, label, min_hz, _ in watched:
        hz = counts[name] / window
        if min_hz == 0.0:                              # event topic (drive cmd)
            status = PASS if hz > 0 else WARN
        else:
            status = PASS if hz >= min_hz else (WARN if hz > 0 else FAIL)
        report(status, label, f'{name}  {hz:.1f} Hz')

    node.destroy_node()
    rclpy.shutdown()


# ── main ─────────────────────────────────────────────────────────────────────
def main(args=None):
    ap = argparse.ArgumentParser(description='F1TENTH hardware diagnostics')
    ap.add_argument('--no-ros', action='store_true', help='skip the ROS topic scan')
    ap.add_argument('--servo-sweep', action='store_true',
                    help='actively wiggle the steering (WHEELS OFF GROUND)')
    ap.add_argument('--vesc-port', default=None, help='override VESC serial port')
    ap.add_argument('--lidar-port', default=None, help='override RPLidar serial port')
    ap.add_argument('--ros-window', type=float, default=3.0, help='ROS scan seconds')
    # tolerate ROS args (--ros-args …) when run via `ros2 run`
    ns, _ = ap.parse_known_args(args if args is not None else sys.argv[1:])

    print('\n──────────────  F1TENTH HARDWARE DIAGNOSTICS  ──────────────')
    print('Device layer (open the hardware and get a reply):')
    ser = check_vesc(ns.vesc_port)
    check_lidar(ns.lidar_port)
    check_camera()
    check_pca9685()
    if ns.servo_sweep:
        servo_sweep(ser)
    if ser is not None:
        try:
            ser.write(vp.pkt_set_current(0.0)); ser.close()
        except Exception:
            pass

    if not ns.no_ros:
        print('\nROS layer (live publish rates):')
        check_ros(ns.ros_window)

    n_fail = sum(1 for s, _, _ in _results if s == FAIL)
    n_warn = sum(1 for s, _, _ in _results if s == WARN)
    print('────────────────────────────────────────────────────────────')
    print(f'  summary: {len(_results)} checks  '
          f'{sum(1 for s,_,_ in _results if s==PASS)} pass  '
          f'{n_warn} warn  {n_fail} fail')
    print('────────────────────────────────────────────────────────────\n')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
