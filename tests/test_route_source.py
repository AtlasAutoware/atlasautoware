"""
Car-side route hint (route_source.RouteSource, used by policy_bridge).
=====================================================================

Route-hint models get a point 2 m ahead on the planned route in state[2:4]. In the sim
it comes from the task's pre-planned path; on the car policy_bridge rebuilds it from the
map-frame pose, /map and a goal point. These tests pin:

  - /map (OccupancyGrid, row 0 at the bottom) is flipped into the planner's image rows,
    so a round trip reproduces the map loaded from the yaml exactly;
  - unknown cells are blocked by default;
  - no map / goal / pose / route, and arriving at the goal, all return no hint
    (the bridge then publishes zero speed);
  - a route planned on the car matches the sim task planner, and the hint at the start
    points along it;
  - a car pose inside the inflated wall margin is snapped to free space instead of failing;
  - a route planned for an old goal is thrown away when the goal changes mid-plan;
  - only models trained with the hint (config state_mask[2:4]) switch it on.

    python3 -m pytest tests/test_route_source.py -q
"""
import json, math, os, sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
from route_source import RouteSource, model_uses_hint, occ_from_grid, parse_goal   # noqa: E402
from goal_core import Planner                                                    # noqa: E402
from policy_io import route_hint                                                 # noqa: E402

yaml = pytest.importorskip('yaml')
from sim_core import load_map                                                    # noqa: E402

LEVINE = os.path.join(REPO, 'maps', 'levine.yaml')


def _grid_msg(occ):
    """What slam_toolbox / map_server put in OccupancyGrid.data for this map."""
    return (np.flipud(occ).astype(np.int8) * 100).ravel()


@pytest.fixture(scope='module')
def levine():
    occ, res, origin = load_map(LEVINE)
    return occ, res, origin, Planner(occ, res, origin)


def _task(pl, seed=3):
    rng = np.random.default_rng(seed)
    for _ in range(200):
        s = pl.sample_free(rng, 1)[0]; g = pl.sample_free_near(rng, s, 4.0, 12.0)
        if g is None: continue
        path = pl.plan(s, g)
        if path is not None and len(path) >= 4: return s, g, path
    raise RuntimeError('no task')


def test_occupancy_grid_round_trip(levine):
    occ, *_ = levine
    back = occ_from_grid(_grid_msg(occ), occ.shape[1], occ.shape[0])
    assert back.shape == occ.shape and np.array_equal(back, occ)


def test_unknown_is_blocked_by_default():
    data = np.array([[0, -1], [100, 0]], np.int8).ravel()        # row 0 = bottom
    assert occ_from_grid(data, 2, 2).tolist() == [[True, False], [False, True]]
    assert occ_from_grid(data, 2, 2, unknown_blocked=False).tolist() == [[True, False], [False, False]]


def test_parse_goal():
    assert parse_goal('3.5,-1.2') == (3.5, -1.2)
    assert parse_goal(' 1, 2 ') == (1.0, 2.0)
    for bad in ('', 'x,y', '1', None, 'nan,1'):
        assert parse_goal(bad) is None


def test_no_hint_without_inputs(levine):
    occ, res, origin, pl = levine
    s, g, _ = _task(pl)
    rs = RouteSource()
    assert rs.hint((*s, 0.0), 0.0)[1]['route'] == 'no map'
    rs.set_map(occ, res, origin)
    assert rs.hint((*s, 0.0), 0.0)[1]['route'] == 'no goal'
    rs.set_goal(g)
    assert rs.hint(None, 0.0)[1]['route'] == 'no pose'
    assert rs.hint((*s, 0.0), 0.0)[1]['route'].startswith('no route')     # not planned yet


def test_route_matches_sim_planner_and_hint_points_along_it(levine):
    occ, res, origin, pl = levine
    s, g, path = _task(pl)
    th = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
    rs = RouteSource(); rs.set_map(occ_from_grid(_grid_msg(occ), occ.shape[1], occ.shape[0]), res, origin)
    rs.set_goal(g)
    assert rs.replan((*s, th), 0.0)
    assert np.allclose(rs.path, path)                          # same planner, same map -> same route
    h, st = rs.hint((*s, th), 0.1)
    assert st['route'] == 'ok'
    assert h == pytest.approx(route_hint((*s, th), path))
    assert h[0] > 0.5                                          # ahead of the car
    assert rs.hint((*s, th), 10.0)[0] is None                  # route older than keep_s: stop


def test_arrived_means_no_hint(levine):
    occ, res, origin, pl = levine
    s, g, _ = _task(pl)
    rs = RouteSource(goal_tol=0.4); rs.set_map(occ, res, origin); rs.set_goal(g)
    assert rs.replan((*s, 0.0), 0.0)
    h, st = rs.hint((g[0] + 0.2, g[1], 0.0), 0.5)
    assert h is None and st['route'] == 'arrived'


def test_start_inside_wall_margin_is_snapped(levine):
    occ, res, origin, pl = levine
    s, g, _ = _task(pl)
    rs = RouteSource(); rs.set_map(occ, res, origin); rs.set_goal(g); rs.replan(None, 0.0)
    # a point within 0.4 m of free space that the planner counts as blocked (wall margin),
    # e.g. a SLAM pose a few cm off near a wall
    G = rs.planner.grid
    r, c = next((r, c) for r in range(G.shape[0]) for c in range(G.shape[1] - 3)
                if G[r, c] and not G[r, c + 1:c + 4].any())          # margin cell, free 0.1 m to its right
    blocked = rs.planner.c2w(r, c)
    assert pl.plan(blocked, g) is None                         # the raw planner refuses this start
    assert rs.replan((*blocked, 0.0), 1.0) and rs.path is not None


def test_goal_change_during_plan_discards_old_route(levine, monkeypatch):
    occ, res, origin, pl = levine
    s, g, _ = _task(pl, seed=3); _, g2, _ = _task(pl, seed=4)
    rs = RouteSource(); rs.set_map(occ, res, origin); rs.set_goal(g); rs.replan(None, 0.0)
    real = rs.planner.plan

    def plan_then_new_goal(a, b):
        out = real(a, b); rs.set_goal(g2); return out          # user clicks a new goal mid-A*
    monkeypatch.setattr(rs.planner, 'plan', plan_then_new_goal)
    assert not rs.replan((*s, 0.0), 0.0)
    assert rs.path is None and rs.goal == g2


class _Meta:
    def __init__(self, d): self.custom_metadata_map = d


class _Sess:
    def __init__(self, d): self.d = d
    def get_modelmeta(self): return _Meta(self.d)


def test_hint_only_for_models_trained_with_it():
    cfg = lambda m: {'config': json.dumps({'state_mask': m})}
    assert model_uses_hint(_Sess(cfg('0,0,1,1,0')))
    assert not model_uses_hint(_Sess(cfg('0,0,0,0,0')))        # 9/23-9/24 models
    assert not model_uses_hint(_Sess({}))                      # pre-metadata exports
    assert not model_uses_hint(_Sess({'config': 'not json'}))
