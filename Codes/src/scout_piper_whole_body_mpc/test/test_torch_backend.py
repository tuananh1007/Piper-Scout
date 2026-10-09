"""Torch backend against the numpy reference (skipped without torch).

    cd Codes/src/scout_piper_whole_body_mpc && PYTHONPATH=../scout_piper_scene_repr/python python -m pytest test -q
"""

import numpy as np
import pytest

from scout_piper_whole_body_mpc.costs.terms import CostWeights, Goal, WholeBodyCost
from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel
from scout_piper_whole_body_mpc.safety.projection import SafetyFilter
from scout_piper_whole_body_mpc.scene_adapter import gridded, spheres_distance_fn
from scout_piper_whole_body_mpc.sim import run_closed_loop
from scout_piper_whole_body_mpc.solvers.mppi import MPPIConfig

# A module-level importorskip stops ROS's launch_testing pytest plugin from
# collecting the package's other test files, so skip per test instead.
try:
    import torch
except ImportError:
    torch = None
else:
    from scout_piper_whole_body_mpc.torch_backend import (TorchCost, TorchGridField, TorchModel,
                                                          TorchMPPI)

pytestmark = pytest.mark.skipif(torch is None, reason="torch not installed")

Q0 = np.array([0.0, 1.2, -1.0, 0.0, 0.5, 0.0])
X0 = np.r_[0.0, 0.0, 0.0, Q0]
F64 = torch.float64 if torch is not None else None


def _random_controls(m, n=16, H=20, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0, 0.4, (n, H, 8)), m.u_low, m.u_high)


def test_rollout_kinematics_and_spheres_match_numpy():
    m = WholeBodyModel()
    tm = TorchModel(m, dtype=F64)
    U = _random_controls(m)
    X = m.rollout(X0, U)
    Xt = tm.rollout(tm.tensor(X0), tm.tensor(U)).numpy()
    assert np.allclose(X, Xt, atol=1e-12)
    F = m.kin.link_frames(X[..., 3:])
    Ft = tm.link_frames(tm.tensor(X[..., 3:])).numpy()
    assert np.allclose(F, Ft, atol=1e-12)
    assert np.allclose(m.kin.manipulability(X[..., 3:]), tm.manipulability(tm.tensor(F)).numpy(), atol=1e-6)
    assert np.allclose(m.kin.wrist_extension(X[..., 3:]), tm.wrist_extension(tm.tensor(X[..., 3:])).numpy())


def test_cost_matches_numpy_term_by_term():
    m = WholeBodyModel()
    dist = spheres_distance_fn([[0.58, -0.145, 0.434]], [0.03])
    leaf = lambda p: np.where(np.linalg.norm(p - [0.6, 0.1, 0.5], axis=1) < 0.1, 1.0, 0.0)  # noqa: E731
    axis = np.array([1.0, 0.0, 0.0])
    w = CostWeights(reach=1e4, reach_advanced=1e3, goal_terminal_linear=20.0)
    cost = WholeBodyCost(m, Goal(p=np.array([0.62, -0.2, 0.4]), approach_axis=axis, advance_m=0.12),
                         distance_fn=dist, leaf_fn=leaf, w=w)
    U = _random_controls(m, seed=3)
    X = m.rollout(X0, U)
    u_prev = np.full(8, 0.05)
    tm = TorchModel(m, dtype=F64)
    J = cost(X, U, u_prev)
    Jt = TorchCost(tm, cost)(tm.tensor(X), tm.tensor(U), tm.tensor(u_prev)).numpy()
    assert np.allclose(J, Jt, rtol=1e-6, atol=1e-6)
    R = np.eye(3)
    cost_R = WholeBodyCost(m, Goal(p=np.array([0.62, -0.2, 0.4]), R=R))
    assert np.allclose(cost_R(X, U), TorchCost(tm, cost_R)(tm.tensor(X), tm.tensor(U)).numpy(), rtol=1e-6)


def test_grid_field_matches_the_numpy_grid():
    dist = spheres_distance_fn([[0.6, 0.0, 0.4], [0.9, 0.3, 0.5]], [0.05, 0.08])
    g = gridded(dist, [0.2, -0.4, 0.0], [1.2, 0.6, 0.9], 0.02)
    f = TorchGridField.from_grid_fn(g, dtype=F64)
    P = np.random.default_rng(0).uniform([0.25, -0.35, 0.05], [1.15, 0.55, 0.85], (2000, 3))
    d, valid = f.distance(torch.tensor(P, dtype=F64))
    assert valid.all()
    assert np.allclose(d.numpy(), g(P)[0], atol=1e-9)
    assert np.abs(d.numpy() - dist(P)[0]).max() < 0.02          # trilinear on a 2 cm grid


def test_snapshot_field_matches_the_field_sampler():
    pytest.importorskip("scipy")
    try:
        from scout_piper_scene_repr_py.distance_query import SemanticDistanceQuery
        from scout_piper_scene_repr_py.field import FieldSampler, export_field
        from scout_piper_scene_repr_py.voxel_map import SemanticVoxelMap, grid_around
    except ImportError:
        pytest.skip("scout_piper_scene_repr python package not on PYTHONPATH")
    vmap = SemanticVoxelMap(grid_around([0.6, 0.0, 0.4], 0.2, 0.01))
    spec = vmap.spec
    idx = np.argwhere(np.ones(spec.shape, bool))
    c = spec.index_to_world(idx)
    stem = np.linalg.norm(c[:, :2] - [0.6, 0.0], axis=1) < 0.012
    leaf = np.linalg.norm(c - [0.55, 0.08, 0.45], axis=1) < 0.04
    vmap.observed[:] = True
    vmap.last_seen[:] = 10.0
    vmap.hits["stem"][tuple(idx[stem].T)] = 10
    vmap.hits["leaf"][tuple(idx[leaf].T)] = 10
    vmap.stamp, vmap.version = 10.0, 1
    snap = export_field(SemanticDistanceQuery(vmap))
    ref = FieldSampler(snap)
    f = TorchGridField.from_snapshot(snap, now=10.0, max_age_s=5.0, dtype=F64)
    P = np.random.default_rng(1).uniform(spec.origin + 0.01, spec.origin + 0.39, (1000, 3))
    q = ref.query(P, now=10.0, max_voxel_age_s=5.0)
    d, valid = f.distance(torch.tensor(P, dtype=F64))
    assert valid.numpy().all() and q["fresh"].all()
    assert np.allclose(d.numpy(), q["hard"], atol=1e-9)
    outside = torch.tensor([[5.0, 5.0, 5.0]], dtype=F64)
    d_out, v_out = f.distance(outside)
    assert not bool(v_out[0]) and float(d_out[0]) <= 0.0           # unknown is not free


@pytest.mark.parametrize("goal, needs_base", [([0.58, 0.10, 0.40], False), ([1.50, 0.30, 0.40], True)])
def test_torch_mppi_reaches_r1_and_r3(goal, needs_base):
    m = WholeBodyModel()
    cost = WholeBodyCost(m, Goal(p=np.array(goal)))
    mppi = TorchMPPI(m, MPPIConfig(seed=0), device="cpu")
    log = run_closed_loop(X0, mppi, cost, SafetyFilter(m), steps=150)
    err = np.linalg.norm(m.tcp_world(log.X[-1])[:3, 3] - goal)
    assert err < 0.02
    assert (log.base_distance > 0.3) == needs_base


def test_torch_mppi_avoids_an_obstacle_through_the_device_field():
    m = WholeBodyModel()
    dist = spheres_distance_fn([[0.58, -0.145, 0.434]], [0.03])
    g = gridded(dist, [0.0, -0.8, 0.0], [1.2, 0.6, 1.0], 0.01)
    goal = np.array([0.62, -0.35, 0.40])
    cost = WholeBodyCost(m, Goal(p=goal), distance_fn=g)
    mppi = TorchMPPI(m, MPPIConfig(seed=1), device="cpu", field=TorchGridField.from_grid_fn(g))
    log = run_closed_loop(X0, mppi, cost, SafetyFilter(m, distance_fn=dist), steps=150)
    assert np.linalg.norm(m.tcp_world(log.X[-1])[:3, 3] - goal) < 0.02
    C, r = m.collision_spheres(np.array(log.X))
    assert (dist(C.reshape(-1, 3))[0].reshape(C.shape[:-1]) - r).min() > 0.0
