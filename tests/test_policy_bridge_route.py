"""
policy_bridge with a route-hint model, exercised without ROS or the car.
========================================================================

The real callbacks and _tick run against stub ROS modules and a fake ONNX session that
records the state vector it was fed. Pins: no goal / no pose -> zero speed and the network
is not even run; with map + goal + pose the hint lands in state[2:4]; inside goal_tol the
car is held; older models keep the IMU in state[2:5] and ignore the route entirely.

    python3 -m pytest tests/test_policy_bridge_route.py -q
"""
import math, os, sys, time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
pytest.importorskip('cv2')
pytest.importorskip('yaml')
from sim_core import load_map          # noqa: E402
from goal_core import Planner          # noqa: E402


class _Msg:
    def __init__(self, **kw): self.__dict__.update(kw)


class _Drive:
    def __init__(self): self.drive = SimpleNamespace(speed=0.0, steering_angle=0.0)


@pytest.fixture
def bridge_mod(monkeypatch):
    names = ('rclpy', 'rclpy.node', 'rclpy.qos', 'sensor_msgs', 'sensor_msgs.msg', 'nav_msgs',
             'nav_msgs.msg', 'ackermann_msgs', 'ackermann_msgs.msg', 'std_msgs', 'std_msgs.msg',
             'geometry_msgs', 'geometry_msgs.msg')
    for n in names: monkeypatch.setitem(sys.modules, n, ModuleType(n))
    sys.modules['rclpy'].ok = lambda: False
    sys.modules['rclpy.node'].Node = object
    q = sys.modules['rclpy.qos']
    q.qos_profile_sensor_data = None; q.QoSProfile = _Msg
    q.DurabilityPolicy = SimpleNamespace(TRANSIENT_LOCAL=1); q.ReliabilityPolicy = SimpleNamespace(RELIABLE=1)
    for mod, cls in (('sensor_msgs.msg', ('Image', 'Imu', 'LaserScan')), ('nav_msgs.msg', ('Odometry', 'OccupancyGrid')),
                     ('std_msgs.msg', ('String',)), ('geometry_msgs.msg', ('PoseStamped',))):
        for c in cls: setattr(sys.modules[mod], c, _Msg)
    sys.modules['ackermann_msgs.msg'].AckermannDriveStamped = _Drive
    monkeypatch.delitem(sys.modules, 'policy_bridge', raising=False)
    import policy_bridge
    return policy_bridge


class _Sess:
    def __init__(self): self.fed = []
    def run(self, names, feed):
        self.fed.append(feed['state'].copy()); return [np.array([[0.8, 0.05]], np.float32)]   # speed, steer


def _make(mod, use_hint):
    B = mod.PolicyBridge.__new__(mod.PolicyBridge)
    B.p = {'instruction': 'go', 'stale': 0.5, 'max_speed': 1.0, 'max_steer': 0.4, 'aeb_dist': 0.35,
           'pose_stale': 0.5, 'replan': 1.0}
    B.sess = _Sess(); B.order = ('speed', 'steer'); B.ids = np.zeros((1, 24), np.int64)
    B.front = np.zeros((96, 128, 3), np.uint8); B.t_img = B.t_scan = time.time()
    B.scan = (np.full(540, 5.0, np.float32), -math.pi, 2 * math.pi / 540); B.front_clear = 5.0
    B.state = np.zeros(5, np.float32); B.n = 0; B.t0 = time.time() - 1
    B.use_hint = use_hint; B.route = mod.RouteSource(goal_tol=0.4); B.pose = None; B.t_pose = 0.0
    B.pub = []; B.status = []
    B.drive_pub = SimpleNamespace(publish=B.pub.append); B.st_pub = SimpleNamespace(publish=B.status.append)
    B.get_logger = lambda: SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None)
    return B


def _levine_task():
    occ, res, origin = load_map(os.path.join(REPO, 'maps', 'levine.yaml'))
    pl = Planner(occ, res, origin); rng = np.random.default_rng(3)
    while True:
        s = pl.sample_free(rng, 1)[0]; g = pl.sample_free_near(rng, s, 4.0, 12.0)
        path = pl.plan(s, g) if g else None
        if path and len(path) >= 4: return occ, res, origin, s, g, path


def _map_msg(occ, res, origin):
    info = _Msg(width=occ.shape[1], height=occ.shape[0], resolution=res,
                origin=_Msg(position=_Msg(x=origin[0], y=origin[1])))
    return _Msg(info=info, data=(np.flipud(occ).astype(np.int8) * 100).ravel().tolist())


def _odom(x, y, yaw):
    return _Msg(pose=_Msg(pose=_Msg(position=_Msg(x=x, y=y),
                                    orientation=_Msg(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)))))


def test_hint_model_holds_until_it_has_a_route(bridge_mod):
    import json
    B = _make(bridge_mod, True)
    B._tick()
    assert B.pub[-1].drive.speed == 0.0 and B.sess.fed == []          # no map/goal: network not run
    assert json.loads(B.status[-1].data)['route'] == 'no map'

    occ, res, origin, s, g, path = _levine_task()
    th = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
    B._map(_map_msg(occ, res, origin)); B._goal_str(_Msg(data=f'{g[0]},{g[1]}'))
    B._tick()
    assert json.loads(B.status[-1].data)['route'] == 'no pose' and B.pub[-1].drive.speed == 0.0

    B._pose(_odom(s[0], s[1], th))
    assert B.pose[2] == pytest.approx(th)
    B.route.replan(B._fresh_pose(), time.time())                        # what the replan thread does
    B._imu(_Msg(angular_velocity=_Msg(x=9.0, y=9.0, z=0.3)))
    B._tick()
    st = json.loads(B.status[-1].data)
    assert st['route'] == 'ok' and B.pub[-1].drive.speed == pytest.approx(0.8)
    fed = B.sess.fed[-1][0]
    assert fed[2:4] == pytest.approx(st['hint'], abs=0.01) and fed[2] > 0.5   # hint, not the IMU x/y
    assert fed[4] == pytest.approx(0.3)

    B._pose(_odom(g[0] + 0.1, g[1], th)); B._tick()
    assert json.loads(B.status[-1].data)['route'] == 'arrived' and B.pub[-1].drive.speed == 0.0

    B.t_pose -= 5.0; B._tick()                                          # localization went quiet
    assert json.loads(B.status[-1].data)['route'] == 'no pose' and B.pub[-1].drive.speed == 0.0


def test_old_model_ignores_route(bridge_mod):
    B = _make(bridge_mod, False)
    B._imu(_Msg(angular_velocity=_Msg(x=0.1, y=0.2, z=0.3)))
    B._tick()
    assert B.pub[-1].drive.speed == pytest.approx(0.8)                  # drives with no goal at all
    assert B.sess.fed[-1][0][2:5] == pytest.approx([0.1, 0.2, 0.3])
