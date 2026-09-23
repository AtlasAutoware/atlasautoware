"""Hardware-free regressions for the final actuator boundary and watchdog."""
import math
import os
import struct
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'f1tenth_gym_ros'))
import drive_node as drive
import vesc_protocol as vp


class Serial:
    def __init__(self):
        self.tx = b''

    def write(self, data):
        self.tx += data


def config():
    return dict(max_speed=7.0, max_steer=0.41, erpm_gain=4614.0,
                steer_invert=False, steer_trim_us=0.0,
                steer_half_range_us=400.0)


@pytest.mark.parametrize('mode', ['speed', 'current'])
@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize('field', ['speed', 'steer'])
def test_nonfinite_command_never_writes_either_actuator(mode, value, field):
    ser = Serial()
    backend = drive.VescSerialBackend(ser, dict(config(), control_mode=mode))
    command = dict(speed=2.0, steer=0.0)
    command[field] = value
    with pytest.raises(ValueError):
        backend.command(**command)
    assert ser.tx == b''


@pytest.mark.parametrize('speed', [100.0, -100.0])
def test_uart_rpm_respects_configured_speed_limit(speed):
    ser = Serial()
    backend = drive.VescSerialBackend(ser, config())
    backend.command(speed, 0.0)
    throttle = vp.PacketParser().feed(ser.tx)[0]
    assert throttle[0] == vp.COMM_SET_RPM
    assert struct.unpack('>i', throttle[1:])[0] == math.copysign(7 * 4614, speed)


@pytest.fixture
def node(monkeypatch):
    # Exercise the actual callbacks without a ROS installation or hardware.
    for name in ('rclpy', 'rclpy.node', 'ackermann_msgs',
                 'ackermann_msgs.msg', 'nav_msgs', 'nav_msgs.msg'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['rclpy.node'].Node = object
    sys.modules['ackermann_msgs.msg'].AckermannDriveStamped = object
    sys.modules['nav_msgs.msg'].Odometry = object
    _, cls = drive._make_node()
    instance = cls.__new__(cls)
    instance.arm_until = 0.0
    instance.last_cmd = 90.0
    instance.timeout = 0.5
    instance._wd_warned = False
    instance.log = []
    instance.get_logger = lambda: SimpleNamespace(error=instance.log.append,
                                                  warning=instance.log.append)
    instance.commands = []
    instance.neutrals = []
    instance.backend = SimpleNamespace(
        command=lambda *args: instance.commands.append(args),
        neutral=lambda: instance.neutrals.append(True))
    monkeypatch.setattr(drive.time, 'monotonic', lambda: 100.0)
    return instance


def message(speed=2.0, steer=0.0):
    return SimpleNamespace(drive=SimpleNamespace(speed=speed, steering_angle=steer))


def test_invalid_command_stops_immediately_and_does_not_feed_watchdog(node):
    node._drive_cb(message(steer=math.nan))
    assert node.commands == []
    assert node.neutrals == [True]
    assert node.last_cmd == -math.inf
    node._watchdog()
    assert len(node.neutrals) == 2


def test_partial_actuation_failure_attempts_neutral_and_watchdog_retries(node):
    def failed_command(*args):
        raise OSError('steering write failed after throttle')
    def failed_neutral():
        raise OSError('disconnected')
    node.backend.command = failed_command
    node.backend.neutral = failed_neutral
    node._drive_cb(message())
    assert any('neutral FAILED' in item for item in node.log)
    node.backend.neutral = lambda: node.neutrals.append(True)
    node._watchdog()
    assert node.neutrals == [True]


def test_successful_command_refreshes_watchdog(node):
    node._drive_cb(message())
    assert node.commands == [(2.0, 0.0)]
    assert node.last_cmd == 100.0
    node._watchdog()
    assert node.neutrals == []


def test_arming_does_not_accept_command_or_refresh_watchdog(node):
    node.arm_until = 101.0
    node._drive_cb(message())
    assert node.commands == []
    assert node.last_cmd == 90.0
    node._watchdog()
    assert node.neutrals == [True]


def test_serial_probe_bounds_writes(monkeypatch):
    calls = []
    class ReplySerial(Serial):
        in_waiting = 1

        def read(self, count):
            return vp.frame(bytes([vp.COMM_FW_VERSION, 1, 0]))

    def open_serial(*args, **kwargs):
        calls.append(kwargs)
        return ReplySerial()

    monkeypatch.setitem(sys.modules, 'serial', SimpleNamespace(Serial=open_serial))
    monkeypatch.setattr(drive.time, 'sleep', lambda _: None)
    assert drive.probe_vesc(dict(serial_port='/fake', serial_baud=115200),
                           lambda _: None) is not None
    assert calls[0]['write_timeout'] == 0.1


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_pca_invalid_steering_does_not_apply_throttle(value):
    backend = drive.PCA9685Backend.__new__(drive.PCA9685Backend)
    backend.cfg = config()
    writes = []
    backend.dev = SimpleNamespace(set_pulse_us=lambda *args: writes.append(args))
    backend.ch_thr, backend.ch_str = 0, 1
    with pytest.raises(ValueError):
        backend.command(2.0, value)
    assert writes == []
