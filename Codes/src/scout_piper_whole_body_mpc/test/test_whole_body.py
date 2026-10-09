"""Whole-body MPC core (no ROS):

    cd Codes/src/scout_piper_whole_body_mpc && PYTHONPATH=../scout_piper_scene_repr/python python -m pytest test -q
"""

import os
import sys

import numpy as np
import pytest

from scout_piper_whole_body_mpc.baselines.sequential import choose_base_pose, run_sequential
from scout_piper_whole_body_mpc.costs.terms import CostWeights, Goal, WholeBodyCost
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


def test_wrist_extension_depends_on_the_elbow_only():
    k = PiperKinematics()
    Q = np.random.default_rng(1).uniform(k.lower, k.upper, (500, 6))
    F = k.link_frames(Q)
    ref = np.linalg.norm(F[..., 4, :3, 3] - F[..., 2, :3, 3], axis=-1)
    assert np.allclose(k.wrist_extension(Q), ref, atol=1e-12)
    assert 0.08 < k.wrist_extension(Q).min() and k.wrist_extension(Q).max() < 0.54


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


def test_reach_margin_moves_the_base_instead_of_stretching_the_arm():
    """A stretched arm at the goal leaves the stem servo and the final approach
    no room; with the reach term the base covers the rest (R3: the arm ends at
    0.54 m, fully stretched, without it)."""
    m = WholeBodyModel()
    cost = WholeBodyCost(m, Goal(p=np.array([1.5, 0.3, 0.40])), w=CostWeights(reach=1e4))
    log = run_closed_loop(X0, MPPI(m, MPPIConfig(seed=0)), cost, SafetyFilter(m), steps=150)
    assert np.linalg.norm(m.tcp_world(log.X[-1])[:3, 3] - cost.goal.p) < 0.02
    assert m.kin.wrist_extension(log.X[-1][3:]) < cost.w.reach_max_m + 0.02


def test_wrist_extension_after_advance_needs_no_inverse_kinematics():
    """The shoulder stays put and the wrist centre translates with the TCP."""
    k = PiperKinematics()
    q = np.array([0.15, 1.72, -1.15, -0.06, -0.23, 0.23])
    assert k.wrist_extension_after_advance(q, 0.0) == pytest.approx(k.wrist_extension(q))
    F = k.link_frames(q)
    w = F[4, :3, 3] - F[2, :3, 3]
    assert k.wrist_extension_after_advance(q, 0.12) == pytest.approx(np.linalg.norm(w + 0.12 * F[-1, :3, 2]))


def test_advance_term_leaves_the_wrist_room_for_the_final_approach():
    """A pose goal followed by a 0.12 m advance (stem_grasp): without the term the
    arm reaches the pre-grasp pose with the wrist 0.46 m out after the advance,
    where the Piper's wrist singularity is close; with it, under 0.40 m."""
    stem = np.array([1.1, -0.1])
    axis = np.r_[stem, 0.0] / np.linalg.norm(stem)
    pre = np.r_[stem, 0.396] - 0.12 * axis
    m = WholeBodyModel()
    cost = WholeBodyCost(m, Goal(p=pre, approach_axis=axis, advance_m=0.12),
                         w=CostWeights(reach=1e4, reach_advanced=1e3, orient=2.0))
    log = run_closed_loop(X0, MPPI(m, MPPIConfig(seed=0)), cost, SafetyFilter(m), steps=120)
    x = np.asarray(log.X)[-1]
    assert np.linalg.norm(m.tcp_world(x)[:3, 3] - pre) < 0.02
    assert m.kin.wrist_extension_after_advance(x[3:], 0.12) < cost.w.reach_max_advanced_m + 0.02


OBSTACLE_GOAL = np.array([0.62, -0.35, 0.40])
OBSTACLE = ([0.58, -0.145, 0.434], 0.03)


@pytest.mark.parametrize("seed", [0, 2])
def test_obstacle_on_the_direct_path_is_avoided_and_the_goal_reached(seed):
    """The obstacle blocks the straight TCP path (the unobstructed run passes
    6 cm *through* it) while the goal configuration itself is 7 cm clear.

    Before P3A.6 the plain MPPI detoured safely but ended 3–5 cm short (seed 2
    also froze for 120 steps against the safety filter). Elitism, gradient
    refinement and a planner margin above the filter's d_safe fix both: 8/8
    seeds ≤ 1 cm in the offline sweep.
    """
    m, cost, mppi, safety = _controller(OBSTACLE_GOAL, obstacles=[OBSTACLE], seed=seed)
    log = run_closed_loop(X0, mppi, cost, safety, steps=150)
    dist = spheres_distance_fn([OBSTACLE[0]], [OBSTACLE[1]])
    clear = [(dist(m.collision_spheres(x[None])[0][0])[0] - m.collision_spheres(x[None])[1]).min()
             for x in log.X]
    assert min(clear) > 0.02, min(clear)                      # never inside d_safe
    err = np.linalg.norm(m.tcp_world(log.X[-1])[:3, 3] - OBSTACLE_GOAL)
    assert err < 0.015, err
    assert log.reasons.count("clearance") == 0                # no deadlock against the filter


def test_solve_never_returns_worse_than_stopping():
    """Elitism: the zero ("stop") sequence is always scored, so the returned
    plan cannot cost more than holding still (the old weighted average often
    did, and the arm wandered around the goal)."""
    m, cost, mppi, safety = _controller([0.58, 0.10, 0.40])
    x = X0.copy()
    x[3:] = [0.1, 1.4, -1.2, 0.0, 0.6, 0.0]
    zero = np.zeros((1, mppi.cfg.horizon, 8))
    for _ in range(5):
        mppi.solve(x, cost, np.zeros(8))
        assert mppi.last_cost <= cost(m.rollout(x, zero), zero, np.zeros(8))[0] + 1e-9


def test_gradient_refinement_lowers_the_cost():
    m, cost, _, _ = _controller(OBSTACLE_GOAL, obstacles=[OBSTACLE])
    x = X0.copy()
    plain = MPPI(m, MPPIConfig(seed=3, refine_iters=0))
    refined = MPPI(m, MPPIConfig(seed=3, refine_iters=3))
    plain.solve(x, cost, np.zeros(8))
    refined.solve(x, cost, np.zeros(8))
    assert refined.last_cost < plain.last_cost


# ----------------------------------------------------- W1 sequential baseline
def test_sequential_baseline_skips_the_base_when_the_goal_is_reachable():
    m = WholeBodyModel()
    plan = choose_base_pose(m, np.array([0.58, 0.10, 0.40]), X0)
    assert plan is not None and np.allclose(plan.pose, X0[:3]) and plan.travel == 0.0


def test_sequential_baseline_reaches_r3_base_first_then_arm_only():
    goal = np.array([1.5, 0.3, 0.40])
    m = WholeBodyModel()
    cost = WholeBodyCost(m, Goal(p=goal))
    res = run_sequential(X0, cost, SafetyFilter(m), steps=150, seed=0)
    assert res.plan is not None and res.plan.ik_error_m < 0.005 and res.switch_step > 0
    P = np.array([x[:3] for x in res.log.X])
    assert np.allclose(P[res.switch_step:], P[res.switch_step])           # base frozen in the arm phase
    assert np.linalg.norm(P[res.switch_step, :2] - res.plan.pose[:2]) < 0.04
    q = np.array([x[3:] for x in res.log.X[:res.switch_step + 1]])
    assert np.allclose(q, X0[3:])                                          # arm frozen in the base phase
    assert np.linalg.norm(m.tcp_world(res.log.X[-1])[:3, 3] - goal) < 0.02


def test_sequential_baseline_avoids_base_poses_in_collision():
    goal = np.array([1.5, 0.3, 0.40])
    m = WholeBodyModel()
    free = choose_base_pose(m, goal, X0)
    blocker = spheres_distance_fn([np.r_[free.pose[:2], 0.15]], [0.1])     # obstacle at that pose
    plan = choose_base_pose(m, goal, X0, distance_fn=blocker)
    assert plan is not None and np.linalg.norm(plan.pose[:2] - free.pose[:2]) > 0.1
    C, r = m.collision_spheres(np.r_[plan.pose, plan.q][None])
    assert (blocker(C[0])[0] - r).min() > 0.03


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


def test_no_entry_policy_moves_in_unknown_space_but_never_into_it():
    """Eye-in-hand camera: most of the arm sits in never-observed space. "no_entry"
    lets those spheres move but refuses a step that takes an observed sphere
    into unknown space; "stop" (the strict default) stops outright."""
    m = WholeBodyModel()
    x_limit = 0.45                                            # observed for x < 0.45, unknown beyond

    def field(p):
        return np.full(len(p), 1.0), p[:, 0] < x_limit

    C0, _ = m.collision_spheres(X0[None])
    assert (C0[0, :, 0] >= x_limit).any() and (C0[0, :, 0] < x_limit).any()
    forward = np.r_[0.2, 0.0, np.zeros(6)]
    strict = SafetyFilter(m, distance_fn=field)
    assert strict.project(X0, forward, forward).reason == "watchdog"
    lenient = SafetyFilter(m, distance_fn=field, unknown_policy="no_entry")
    back = np.r_[-0.2, 0.0, np.zeros(6)]
    rep = lenient.project(X0, back, back)                      # backing away keeps known spheres known
    assert rep.reason in ("ok", "rate") and rep.u[0] < 0
    x_edge = X0.copy()
    x_edge[0] = x_limit - 0.01 - C0[0, :, 0].max() + 0.0        # front-most sphere 1 cm short of unknown
    rep = lenient.project(x_edge, forward, forward)
    assert rep.reason == "clearance" and rep.u[0] < forward[0]
    Cn, _ = m.collision_spheres(m.rollout(x_edge, rep.u[None, None])[0, 1][None])
    assert (Cn[0, :, 0] < x_limit).sum() >= (m.collision_spheres(x_edge[None])[0][0, :, 0] < x_limit).sum()


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
    # outside the grid (base, arm behind the camera): valid and free, not unknown
    d_out, v_out = fn(np.array([[-0.5, 0.0, 0.2]]))
    assert v_out[0] and np.isinf(d_out[0])
    strict = semantic_distance_fn(SemanticDistanceQuery(vmap), now=0.0, outside_free=False)
    assert not strict(np.array([[-0.5, 0.0, 0.2]]))[1][0]
