"""Semantic voxel map + planner distance query on a ray-cast synthetic scene.

    cd Codes/src/scout_piper_scene_repr && python -m pytest test -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from scout_piper_scene_repr_py.distance_query import (  # noqa: E402
    SemanticDistanceQuery, signed_distance_field)
from scout_piper_scene_repr_py.policy import load_policies  # noqa: E402
from scout_piper_scene_repr_py.voxel_map import SemanticVoxelMap, grid_around  # noqa: E402

H, W = 240, 320
K = np.array([[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1.0]])
STEM_X, STEM_Z, STEM_R = 0.03, 0.40, 0.005     # vertical stem (axis ∥ y)
LEAF_Z = 0.30                                  # leaf patch: x∈[-0.10,-0.02], y∈[-0.04,0.04]
WALL_Z = 0.60


def render(T_world_cam=np.eye(4), leaf_dx=0.0):
    """Ray-cast depth + class masks for a camera at T_world_cam (looking +z)."""
    vv, uu = np.mgrid[0:H, 0:W]
    d = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones((H, W))], -1)
    R, o = T_world_cam[:3, :3], T_world_cam[:3, 3]
    dw = d @ R.T
    depth = np.full((H, W), np.inf)
    lab = np.zeros((H, W), int)
    # wall
    t = (WALL_Z - o[2]) / dw[..., 2]
    depth = np.where(t > 0, t, depth)
    # leaf
    t = (LEAF_Z - o[2]) / dw[..., 2]
    p = o + dw * t[..., None]
    hit = (t > 0) & (p[..., 0] > -0.10 + leaf_dx) & (p[..., 0] < -0.02 + leaf_dx) & (np.abs(p[..., 1]) < 0.04) & (t < depth)
    depth, lab = np.where(hit, t, depth), np.where(hit, 3, lab)
    # stem: solve |(o + t d)_xz − c| = r
    ox, oz = o[0] - STEM_X, o[2] - STEM_Z
    a = dw[..., 0] ** 2 + dw[..., 2] ** 2
    b = 2 * (ox * dw[..., 0] + oz * dw[..., 2])
    c = ox ** 2 + oz ** 2 - STEM_R ** 2
    disc = b * b - 4 * a * c
    t = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
    hit = (disc > 0) & (t > 0) & (t < depth)
    depth, lab = np.where(hit, t, depth), np.where(hit, 1, lab)
    z = depth  # camera z equals ray parameter because d_z = 1 in camera frame
    return z.astype(np.float32), {"stem": lab == 1, "leaf": lab == 3}


def build(frames=3, voxel=0.005, **kw):
    vmap = SemanticVoxelMap(grid_around([0.0, 0.0, 0.40], 0.25, voxel), max_range_m=1.0)
    for k in range(frames):
        depth, masks = render(**kw)
        vmap.integrate(depth, masks, K, np.eye(4), stamp=float(k), stride=2)
    return vmap


def test_sdf_sign_and_scale():
    occ = np.zeros((9, 9, 9), bool); occ[4, 4, 4] = True
    f = signed_distance_field(occ, 0.01)
    assert f[4, 4, 4] < 0 and np.isclose(f[4, 4, 7], 0.025)


def test_thin_stem_is_preserved():
    vmap = build()
    pts = vmap.occupied_points("stem")
    assert len(pts) > 0
    # coverage of the visible stem centreline segment (|y| < 0.15 m in view)
    ys = np.arange(-0.15, 0.15, 0.005)
    covered = [np.any((np.abs(pts[:, 1] - y) < 0.005) & (np.abs(pts[:, 0] - STEM_X) < 0.01)) for y in ys]
    assert np.mean(covered) > 0.9
    # no stem voxel far from the true stem
    assert np.all(np.abs(pts[:, 0] - STEM_X) < STEM_R + 0.01)


def test_distance_to_stem_matches_geometry():
    q = SemanticDistanceQuery(build())
    pad = q.policies["stem"].padding_m
    offs = np.array([0.02, 0.04, 0.06])
    pts = np.column_stack([STEM_X + np.zeros(3), np.zeros(3), STEM_Z - STEM_R - offs])  # camera side
    r = q.query(pts)
    assert r.valid.all()
    assert np.all(r.hard_class == "stem")
    true = offs - pad                               # distance to surface minus padding
    assert np.all(np.abs(r.hard_distance - true) < 0.006), (r.hard_distance, true)
    assert np.all(r.hard_gradient[:, 2] < 0)        # moving toward the camera increases clearance


def test_unknown_is_not_free_and_stale_is_invalid():
    vmap = build()
    q = SemanticDistanceQuery(vmap, max_age_s=1.0)
    behind_wall = np.array([[0.0, 0.0, 0.64]])
    r = q.query(behind_wall)
    assert not r.valid[0] and r.hard_distance[0] <= 0.0
    r = q.query(np.array([[STEM_X, 0, 0.30]]), now=vmap.stamp + 5.0)
    assert not r.valid[0]


def test_wall_is_hard_via_other_class():
    q = SemanticDistanceQuery(build())
    r = q.query(np.array([[0.15, 0.10, WALL_Z - 0.05]]))
    assert r.valid[0] and r.hard_class[0] == "other"
    assert abs(r.hard_distance[0] - (0.05 - q.policies["other"].padding_m)) < 0.006


def test_leaf_is_soft_not_hard():
    q = SemanticDistanceQuery(build())
    near_leaf = np.array([[-0.06, 0.0, LEAF_Z - 0.005]])
    cost, hard = q.leaf_cost(near_leaf)
    assert cost[0] > 0 and not hard[0]
    r = q.query(near_leaf)
    assert r.hard_class[0] != "leaf"
    cost_far, _ = q.leaf_cost(np.array([[-0.06, 0.0, LEAF_Z - 0.08]]))
    assert cost_far[0] == 0.0


def test_grasp_mode_excludes_only_the_target_region():
    q = SemanticDistanceQuery(build())
    target = np.array([STEM_X, 0.0, STEM_Z - STEM_R])
    probe = target - [0, 0, 0.005]
    farther_on_stem = np.array([STEM_X, 0.08, STEM_Z - STEM_R - 0.005])
    a = q.query(np.vstack([probe, farther_on_stem]))
    g = q.query(np.vstack([probe, farther_on_stem]), mode="grasp", target_point=target,
                exclusion_radius_m=0.03)
    assert g.hard_distance[0] > a.hard_distance[0] + 0.01      # target peduncle released
    assert abs(g.hard_distance[1] - a.hard_distance[1]) < 1e-6  # rest of the stem still hard


def test_moved_leaf_is_cleared_by_free_space():
    vmap = SemanticVoxelMap(grid_around([0.0, 0.0, 0.40], 0.25, 0.005))
    for k in range(3):
        d, m = render()
        vmap.integrate(d, m, K, np.eye(4), stamp=float(k), stride=2)
    old = vmap.occupied("leaf").sum()
    for k in range(3, 15):
        d, m = render(leaf_dx=0.15)            # leaf moved out of the old region
        vmap.integrate(d, m, K, np.eye(4), stamp=float(k), stride=2)
    pts = vmap.occupied_points("leaf")
    assert old > 0 and np.all(pts[:, 0] > -0.10 + 0.15 - 0.01)


def test_repo_policy_yaml_loads():
    here = os.path.dirname(__file__)
    pol = load_policies(os.path.join(here, "..", "config", "semantic_classes.yaml"))
    assert pol["stem"].behavior == "hard" and pol["leaf"].max_penetration_m == 0.02
    assert pol["other"].behavior == "hard"
