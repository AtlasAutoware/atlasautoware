"""Exercise actual controller callbacks/ticks with an in-memory ROS boundary.

No ROS installation, network graph, physical driver or raceline file is needed.
The planner is deterministic so faults and solve latency can be injected.
"""
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest


@pytest.fixture
def controller(monkeypatch):
    def module(name, **attrs):
        result = ModuleType(name)
        result.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    class Drive:
        def __init__(self):
            self.header = NS(stamp=None)
            self.drive = NS(speed=0.0, steering_angle=0.0)

    module('rclpy')
    module('rclpy.node', Node=object)
    module('rclpy.qos', qos_profile_sensor_data=object())
    for package, names in [('sensor_msgs', ['LaserScan', 'Imu']),
                           ('nav_msgs', ['Odometry'])]:
        module(package)
        module(package + '.msg', **dict.fromkeys(names, object))
    module('ackermann_msgs')
    module('ackermann_msgs.msg', AckermannDriveStamped=Drive)
    module('pursuit_agent', find_best_raceline=lambda: None,
           load_raceline=lambda _: None, find_nearest=lambda *args: 0)
    path = Path(__file__).resolve().parents[1] / 'f1tenth_gym_ros'
    monkeypatch.syspath_prepend(str(path))
    spec = importlib.util.spec_from_file_location('raceline_under_test', path / 'raceline_mpc.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    clock = NS(now=10.0)
    monkeypatch.setattr(mod, 'time', NS(monotonic=lambda: clock.now))
    node = mod.RacelineMPC.__new__(mod.RacelineMPC)
    node.x = node.y = node.yaw = 0.0
    node.speed = 1.0
    node.have_odom = True
    node.have_imu = False
    node.scan_time = node.odom_time = clock.now
    node.scan = NS(ranges=[8.0] * 7, angle_min=-0.3, angle_increment=0.1,
                   range_min=0.05, range_max=16.0)
    node.get_parameter = lambda _: NS(value=0.5)
    node.get_clock = lambda: NS(now=lambda: NS(to_msg=lambda: clock.now))
    node.get_logger = lambda: NS(warning=lambda _: None, info=lambda _: None)
    node.messages = []
    node.drive_pub = NS(publish=node.messages.append)
    node.delay = 0.0
    node._delay_ticks = 2
    node._last_cmd = (0.2, 2.0)
    node._cmd_buf = [(0.2, 2.0)] * 2
    node.nearest = node._prev_near = node._log = node.lap = 0
    node.n = 100
    node.rl_x = node.rl_y = [0.0] * 100
    node.aeb_dist, node.aeb_cone, node.aeb_decel = 0.45, 0.2, 6.0
    node.v_scale, node.v_max, node.max_steer = 1.0, 4.0, 0.41
    node.steer_offset = 0.08
    node.mpc = NS(available=True, solve=lambda *args: (0.1, 2.0))
    node.map_ctl = NS(control=lambda *args: (0.2, 1.0))
    node.governor = mod.TractionGovernor()
    return node, clock


def assert_stopped(node):
    assert node.messages[-1].drive.speed == 0.0
    assert node.messages[-1].drive.steering_angle == 0.0
    assert node._last_cmd == (0.0, 0.0)
    assert node._cmd_buf == [(0.0, 0.0)] * 2


@pytest.mark.parametrize('ranges', [[0.2] * 7, [float('nan')] * 7, []])
def test_aeb_stops_before_solver_with_no_steering_bias(controller, ranges):
    node, _ = controller
    node.scan.ranges = ranges
    node.mpc.solve = lambda *args: pytest.fail('AEB must precede the solver')
    node._loop()
    assert_stopped(node)


@pytest.mark.parametrize('sensor', ['scan_time', 'odom_time'])
def test_dropouts_stop_and_fresh_inputs_resume(controller, sensor):
    node, clock = controller
    setattr(node, sensor, clock.now - 0.5)
    node._loop()
    assert_stopped(node)
    setattr(node, sensor, clock.now)
    node._loop()
    assert node.messages[-1].drive.speed == 2.0


def test_solve_cannot_publish_after_sensor_deadline(controller):
    node, clock = controller
    def slow_solve(*args):
        clock.now += 0.51
        return 0.1, 2.0
    node.mpc.solve = slow_solve
    node._loop()
    assert_stopped(node)


@pytest.mark.parametrize('quaternion', [(0, 0, 0, 0), (float('nan'), 0, 0, 0),
                                         (1, 0, float('inf'), 0)])
def test_invalid_quaternion_invalidates_previous_pose(controller, quaternion):
    node, clock = controller
    w, x, y, z = quaternion
    msg = NS(pose=NS(pose=NS(position=NS(x=1.0, y=2.0),
                              orientation=NS(w=w, x=x, y=y, z=z))),
             twist=NS(twist=NS(linear=NS(x=0.0, y=0.0))))
    node._odom_cb(msg)
    node._loop()
    assert_stopped(node)
    msg.pose.pose.orientation = NS(w=1.0, x=0.0, y=0.0, z=0.0)
    node._odom_cb(msg)
    node._loop()
    assert node.messages[-1].drive.speed > 0


def test_invalid_imu_does_not_poison_governor(controller):
    node, _ = controller
    node._imu_cb(NS(angular_velocity=NS(z=float('nan'))))
    node._loop()
    assert math.isfinite(node.governor._lat)
    node._imu_cb(NS(angular_velocity=NS(z=20.0)))
    node._loop()
    assert math.isfinite(node.governor._lat)
    assert node.have_imu


@pytest.mark.parametrize('timeout', [0.0, -1.0, float('nan'), float('inf')])
def test_invalid_timeout_fails_closed(controller, timeout):
    node, _ = controller
    node.get_parameter = lambda _: NS(value=timeout)
    node._loop()
    assert_stopped(node)


def test_nonfinite_governor_output_stops(controller):
    node, clock = controller
    node.have_imu = True
    node.imu_time = clock.now
    node.yaw_rate = 0.0
    node.governor = NS(update=lambda *args: float('nan'))
    node._loop()
    assert_stopped(node)


def test_zero_speed_cap_stays_zero(controller):
    node, _ = controller
    node.v_scale = 0.0
    node._loop()
    assert node.messages[-1].drive.speed == 0.0
