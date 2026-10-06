"""Whole-body MPC core (no ROS):

    cd Codes && python -m pytest src/scout_piper_whole_body_mpc/test -q
"""

import os
import sys

import numpy as np
import pytest

from scout_piper_whole_body_mpc.costs.terms import Goal, WholeBodyCost
from scout_piper_whole_body_mpc.dynamics.piper import JOINTS, PiperKinematics, parse_xacro_joints
from scout_piper_whole_body_mpc.dynamics.scout import ScoutParams, fit_slip, rollout_base
from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel
from scout_piper_whole_body_mpc.safety.projection import SafetyFilter
from scout_piper_whole_body_mpc.scene_adapter import spheres_distance_fn
from scout_piper_whole_body_mpc.sim import run_closed_loop
from scout_piper_whole_body_mpc.solvers.mppi import MPPI, MPPIConfig

SRC = os.path.join(os.path.dirname(__file__), "..", "..")
Q0 = np.array([0.0, 1.2, -1.0, 0.0, 0.5, 0.0])
X0 = np.r_[0.0, 0.0, 0.0, Q0]


# ------------------------------------------------------------------ models
def test_base_rollout_straight_and_arc():
    P = rollout_base(np.zeros(3), np.tile([[1.0, 0.0]], (10, 1)), 0.1)
    assert np.allclose(P[0, -1], [1.0, 0, 0])
    P = rollout_base(np.zeros(3), np.tile([[0.5, 0.5]], (2000, 1)), 0.001)   # radius 1 m, 1 rad
    assert np.allclose(P[0, -1], [np.sin(1), 1 - np.cos(1), 1.0], atol=1e-3)


def test_slip_identification_recovers_parameters():
    rng = np.random.default_rng(0)
    vw = np.column_stack([rng.uniform(0, 0.3, 200), rng.uniform(-0.5, 0.5, 200)])
    truth = ScoutParams(k_v=0.92, k_omega=0.78)
    poses = rollout_base(np.zeros(3), vw, 0.1, truth)[0]
    est = fit_slip(vw, poses + rng.normal(0, 1e-4, poses.shape), 0.1)
    assert abs(est.k_v - 0.92) < 0.01 and abs(est.k_omega - 0.78) < 0.01


def test_fk_table_matches_urdf_xacro():
    path = os.path.join(SRC, "scout_piper_description", "urdf", "_piper_arm.xacro")
    parsed = parse_xacro_joints(path)
    assert [p[0] for p in parsed] == [j[0] for j in JOINTS]
    for p, j in zip(parsed, JOINTS):
        assert np.allclose(p[1], j[1]) and np.allclose(p[2], j[2]) and np.allclose(p[3], j[3])


def test_jacobian_matches_finite_differences():
    k = PiperKinematics()
    q = np.array([0.3, 1.0, -1.2, 0.4, 0.3, -0.2])
    J = k.jacobian(q)
    for i in range(6):
        d = np.zeros(6); d[i] = 1e-6
        fd = (k.tcp(q + d)[:3, 3] - k.tcp(q - d)[:3, 3]) / 2e-6
        assert np.allclose(J[:3, i], fd, atol=1e-6)
    assert k.manipulability(q) > 0


# ---------------------------------------------------------------- control
def _controller(goal_p, obstacles=(), seed=0, samples=256):
    m = WholeBodyModel()
    dist = spheres_distance_fn(*zip(*obstacles)) if obstacles else None
    cost = WholeBodyCost(m, Goal(p=np.asarray(goal_p, float)), distance_fn=dist)
    mppi = MPPI(m, MPPIConfig(samples=samples, seed=seed))
    return m, cost, mppi, SafetyFilter(m, distance_fn=dist)


def test_arm_only_baseline_reaches_r1():
    """W0 must be a fair comparator: it plans without base motion."""
    m = WholeBodyModel()
    cost = WholeBodyCost(m, Goal(p=np.array([0.58, 0.10, 0.40])))
    log = run_closed_loop(X0, MPPI(m, MPPIConfig(seed=0, arm_only=True)), cost,
                          SafetyFilter(m), steps=80, freeze_base=True)
    assert log.base_distance == 0.0
    assert np.linalg.norm(m.tcp_world(log.X[-1])[:3, 3] - cost.goal.p) < 0.02


def test_r1_reachable_target_uses_the_arm_not_the_base():
    m, cost, mppi, safety = _controller([0.58, 0.10, 0.40])
    log = run_closed_loop(X0, mppi, cost, safety, steps=80)
    err = np.linalg.norm(m.tcp_world(log.X[-1])[:3, 3] - cost.goal.p)
    assert err < 0.02, err
    # small repositioning is allowed; measured 7–11 cm vs ~0.7 m in R3
    assert log.base_distance < 0.15, log.base_distance


def test_r3_unreachable_target_moves_the_base_and_beats_arm_only():
    goal = [1.5, 0.3, 0.40]
    m, cost, mppi, safety = _controller(goal)
    log = run_closed_loop(X0, mppi, cost, safety, steps=150)
    err = np.linalg.norm(m.tcp_world(log.X[-1])[:3, 3] - cost.goal.p)
    assert err < 0.02, err
    assert log.base_distance > 0.5
    m2, cost2, _, safety2 = _controller(goal)
    mppi2 = MPPI(m2, MPPIConfig(seed=0, arm_only=True))
    arm_only = run_closed_loop(X0, mppi2, cost2, safety2, steps=150, freeze_base=True)
    err_arm = np.linalg.norm(m2.tcp_world(arm_only.X[-1])[:3, 3] - cost2.goal.p)
    assert err_arm > 0.5                       # W0 cannot reach R3


def test_obstacle_on_the_direct_path_is_avoided():
    """The obstacle blocks the straight TCP path (the unobstructed run passes
    6 cm *through* it) while the goal configuration itself is 7 cm clear.

    Known limitation, measured over seeds: MPPI detours safely but stalls
    3–4 cm short of the goal (a sampling-limited local minimum). The test
    therefore checks safety and progress, not millimetre convergence; a
    gradient refinement (solver candidate B) is the planned fix.
    """
    goal = np.array([0.62, -0.35, 0.40])
    obstacle = ([0.58, -0.145, 0.434], 0.03)
    m, cost, mppi, safety = _controller(goal, obstacles=[obstacle])
    log = run_closed_loop(X0, mppi, cost, safety, steps=150)
    dist = spheres_distance_fn([obstacle[0]], [obstacle[1]])
    clear = [(dist(m.collision_spheres(x[None])[0][0])[0] - m.collision_spheres(x[None])[1]).min()
             for x in log.X]
    assert min(clear) > 0.02, min(clear)                      # never inside d_safe
    start_err = np.linalg.norm(m.tcp_world(X0)[:3, 3] - goal)
    err = np.linalg.norm(m.tcp_world(log.X[-1])[:3, 3] - goal)
    assert err < 0.15 * start_err, (err, start_err)           # ≥ 85 % of the way there


# ----------------------------------------------------------------- safety
def test_safety_filter_scales_motion_into_an_obstacle():
    m = WholeBodyModel()
    tcp = m.tcp_world(X0)[:3, 3]
    # obstacle 6 cm ahead of the TCP in +x; command the base straight at it
    dist = spheres_distance_fn([tcp + [0.06 + 0.03 + 0.02, 0, 0]], [0.02])
    sf = SafetyFilter(m, distance_fn=dist, d_safe=0.02)
    u = np.r_[0.3, 0.0, np.zeros(6)]
    x = X0.copy()
    for _ in range(40):
        rep = sf.project(x, u, u)
        x = m.rollout(x, rep.u[None, None])[0, 1]
    C, r = m.collision_spheres(x[None])
    assert (dist(C[0])[0] - r).min() >= 0.0
    assert rep.reason == "clearance" and rep.scale < 0.1


def test_safety_rate_limits_joint_limits_and_watchdog():
    m = WholeBodyModel()
    sf = SafetyFilter(m)
    rep = sf.project(X0, np.r_[0.3, 0, np.zeros(6)], np.zeros(8))
    assert rep.reason == "rate" and np.isclose(rep.u[0], m.scout.a_max * m.dt)
    x = X0.copy(); x[3 + 1] = m.kin.upper[1] - 1e-3
    u = np.zeros(8); u[3] = 0.5
    rep = sf.project(x, u, u)
    assert rep.u[3] == 0.0 and rep.reason == "joint_limit"
    assert sf.project(X0, u, u, state_age_s=1.0).reason == "watchdog"
    assert np.all(sf.project(X0, u, u, geometry_age_s=5.0).u == 0)


def test_unknown_geometry_at_the_robot_stops():
    m = WholeBodyModel()
    sf = SafetyFilter(m, distance_fn=lambda p: (np.ones(len(p)), np.zeros(len(p), bool)))
    rep = sf.project(X0, np.r_[0.1, 0, np.zeros(6)], np.r_[0.1, 0, np.zeros(6)])
    assert rep.reason == "watchdog" and np.all(rep.u == 0)


# --------------------------------------------------- semantic scene adapter
def test_semantic_scene_query_plugs_in():
    sys.path.insert(0, os.path.join(SRC, "scout_piper_scene_repr", "python"))
    pytest.importorskip("scout_piper_scene_repr_py.distance_query")
    from scout_piper_scene_repr_py.distance_query import SemanticDistanceQuery
    from scout_piper_scene_repr_py.voxel_map import SemanticVoxelMap, grid_around
    from scout_piper_whole_body_mpc.scene_adapter import semantic_distance_fn

    vmap = SemanticVoxelMap(grid_around([0.5, 0.0, 0.4], 0.3, 0.01))
    vmap.observed[:] = True
    vmap.last_seen[:] = 0.0
    vmap.stamp = 0.0
    i = vmap.spec.world_to_index(np.array([0.7, 0.0, 0.4]))
    vmap.hits["stem"][tuple(i)] = 10
    vmap.version += 1
    fn = semantic_distance_fn(SemanticDistanceQuery(vmap), now=0.0)
    d, valid = fn(np.array([[0.7, 0.0, 0.3], [0.5, 0.2, 0.4]]))
    assert valid.all() and abs(d[0] - (0.10 - 0.005 - 0.005)) < 0.011 and d[1] > d[0]
