"""Snapshot round trip, nvblox ESDF merge, target attractor, robot self-filter.

    cd Codes/src/scout_piper_scene_repr && python -m pytest test -q
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from scout_piper_scene_repr_py.attractor import axis_quaternion, largest_cluster, target_goal  # noqa: E402
from scout_piper_scene_repr_py.distance_query import SemanticDistanceQuery  # noqa: E402
from scout_piper_scene_repr_py.field import (AGE_UNKNOWN, FieldSampler, export_field, fill_msg,  # noqa: E402
                                             snapshot_distance_fn, snapshot_from_msg)
from scout_piper_scene_repr_py.nvblox_field import (grid_from_response, merge_class_grids,  # noqa: E402
                                                    sample_grid, target_points)
from scout_piper_scene_repr_py.policy import DEFAULT_POLICIES  # noqa: E402
from scout_piper_scene_repr_py.self_filter import (DEFAULT_PIPER_FINGER_BOXES, parse_boxes,  # noqa: E402
                                                   robot_mask)
from scout_piper_scene_repr_py.voxel_map import SemanticVoxelMap, grid_around  # noqa: E402


def _msg():
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace()), origin=SimpleNamespace())


def _stem_map():
    vmap = SemanticVoxelMap(grid_around([0.6, 0.0, 0.4], 0.15, 0.01))
    idx = np.argwhere(np.ones(vmap.spec.shape, bool))
    c = vmap.spec.index_to_world(idx)
    stem = np.linalg.norm(c[:, :2] - [0.6, 0.0], axis=1) < 0.012
    leaf = np.linalg.norm(c - [0.55, 0.08, 0.45], axis=1) < 0.04
    seen = c[:, 0] < 0.70                               # the far side was never observed
    vmap.observed[tuple(idx[seen].T)] = True
    vmap.last_seen[tuple(idx[seen].T)] = 10.0
    vmap.hits["stem"][tuple(idx[stem & seen].T)] = 10
    vmap.hits["leaf"][tuple(idx[leaf & seen].T)] = 10
    vmap.stamp, vmap.version = 10.0, 1
    return vmap


# ------------------------------------------------------------- snapshot
def test_snapshot_round_trips_through_the_message():
    snap = export_field(SemanticDistanceQuery(_stem_map()))
    back = snapshot_from_msg(fill_msg(_msg(), snap, "odom"))
    assert back.shape == snap.shape and back.hard_classes == snap.hard_classes
    assert np.allclose(back.origin, snap.origin) and abs(back.stamp - snap.stamp) < 1e-6
    assert np.array_equal(back.hard_distance, snap.hard_distance)
    assert np.array_equal(back.age_ds, snap.age_ds) and np.array_equal(back.soft_distance, snap.soft_distance)


def test_snapshot_distance_fn_unknown_inside_free_outside():
    snap = export_field(SemanticDistanceQuery(_stem_map()))
    fn = snapshot_distance_fn(snap, now=10.0, max_voxel_age_s=5.0)
    d, valid = fn(np.array([[0.62, 0.0, 0.4],          # 2 cm from the stem axis, observed
                            [0.72, 0.0, 0.4],          # inside the grid, never observed
                            [2.0, 2.0, 2.0]]))         # outside the grid
    assert valid[0] and 0.0 < d[0] < 0.02
    assert not valid[1] and d[1] <= 0.0                # unknown is not free
    assert valid[2] and np.isinf(d[2])                 # the grid covers the plant only
    stale = snapshot_distance_fn(snap, now=20.0, max_voxel_age_s=5.0)(np.array([[0.62, 0.0, 0.4]]))
    assert not stale[1][0]


# ---------------------------------------------------------- nvblox merge
def _esdf(fn, origin, vs, n, unobserved=None):
    idx = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1).reshape(-1, 3)
    c = np.asarray(origin) + (idx + 0.5) * vs
    d = fn(c).astype(np.float32)
    if unobserved is not None:
        d[unobserved(c)] = -1000.0
    return d.ravel(), n


def test_esdf_response_layout_and_unobserved_values():
    data, dims = _esdf(lambda c: c[:, 0] * 100 + c[:, 1] * 10 + c[:, 2], [0, 0, 0], 1.0, (3, 4, 5),
                       unobserved=lambda c: c[:, 2] > 4)
    g = grid_from_response(data, dims, [0, 0, 0], 1.0)
    assert g.distance.shape == (3, 4, 5)
    assert g.distance[2, 1, 3] == pytest.approx(250 + 15 + 3.5)        # x major, z fastest
    assert np.isnan(g.distance[0, 0, 4])
    assert np.isnan(sample_grid(g, np.array([[1.5, 1.5, 4.2]]))[0])      # touches an unobserved voxel


def test_merge_matches_the_policies_of_the_cpu_snapshot():
    stem_fn = lambda c: np.linalg.norm(c[:, :2] - [0.6, 0.0], axis=1) - 0.01        # noqa: E731
    leaf_fn = lambda c: np.linalg.norm(c - [0.55, 0.08, 0.45], axis=1) - 0.04       # noqa: E731
    stem = grid_from_response(*_esdf(stem_fn, [0.45, -0.15, 0.25], 0.003, (100, 100, 100)),
                              [0.45, -0.15, 0.25], 0.003)
    leaf = grid_from_response(*_esdf(leaf_fn, [0.45, -0.15, 0.25], 0.01, (30, 30, 30),
                                     unobserved=lambda c: c[:, 0] > 0.70),
                              [0.45, -0.15, 0.25], 0.01)
    snap = merge_class_grids({"stem": stem, "leaf": leaf, "branch": None}, DEFAULT_POLICIES,
                             origin=[0.5, -0.1, 0.3], shape=(20, 20, 20), voxel_size=0.01, stamp=5.0,
                             conservative=False)
    assert snap.hard_classes == ["stem"] and snap.soft_class == "leaf"
    s = FieldSampler(snap)
    q = s.query(np.array([[0.63, 0.0, 0.4]]), now=5.0, max_voxel_age_s=1.0)
    assert q["fresh"][0]
    assert q["hard"][0] == pytest.approx(0.03 - 0.01 - DEFAULT_POLICIES["stem"].padding_m, abs=2e-3)
    cons = merge_class_grids({"stem": stem}, DEFAULT_POLICIES, origin=[0.5, -0.1, 0.3],
                             shape=(20, 20, 20), voxel_size=0.01, stamp=5.0, conservative=True)
    assert np.nanmax(snap.hard_distance - cons.hard_distance) == pytest.approx(0.5 * np.sqrt(3) * 0.01, abs=1e-6)
    none = merge_class_grids({"stem": None}, DEFAULT_POLICIES, [0, 0, 0], (2, 2, 2), 0.01, 0.0)
    assert (none.age_ds == AGE_UNKNOWN).all()


# ------------------------------------------------------------- attractor
def test_target_goal_from_the_largest_cluster():
    rng = np.random.default_rng(0)
    flower = np.array([0.9, 0.2, 0.5]) + rng.normal(0, 0.006, (40, 3))
    speck = np.array([[0.7, -0.3, 0.6]])                                   # isolated false positive
    pts = np.concatenate([flower, speck])
    assert len(largest_cluster(pts, 0.01)) == 40
    g = target_goal(pts, 0.01, robot_xy=[0.0, 0.0], offset_m=0.12)
    axis = np.r_[0.9, 0.2, 0.0] / np.hypot(0.9, 0.2)
    assert np.allclose(g.axis, axis, atol=0.02) and np.allclose(g.grasp_point, flower.mean(0), atol=1e-9)
    assert np.allclose(g.pre_grasp, g.grasp_point - 0.12 * g.axis)
    q = axis_quaternion(g.axis)
    x, y, z, w = q
    z_axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    assert np.allclose(z_axis, g.axis, atol=1e-9)                          # what the MPC reads back
    assert target_goal(np.zeros((0, 3)), 0.01, [0, 0]) is None


def test_target_points_from_the_target_esdf():
    fn = lambda c: np.linalg.norm(c - [0.5, 0.0, 0.5], axis=1) - 0.02       # noqa: E731
    g = grid_from_response(*_esdf(fn, [0.4, -0.1, 0.4], 0.005, (40, 40, 40)), [0.4, -0.1, 0.4], 0.005)
    p = target_points(g)
    assert len(p) > 50 and np.allclose(p.mean(0), [0.5, 0.0, 0.5], atol=0.003)


# ------------------------------------------------------------ self-filter
def test_self_filter_removes_finger_pixels_but_keeps_the_scene_behind():
    H, W = 120, 160
    K = np.array([[100.0, 0, 80], [0, 100.0, 60], [0, 0, 1]])
    boxes = parse_boxes(["finger:0,0,0,0.02,0.02,0.04"])
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.03, 0.10]                     # finger 10 cm in front, below the optical axis
    depth = np.full((H, W), 0.6)                     # plant 60 cm away
    u, v = 80, int(60 + 100 * 0.03 / 0.10)
    depth[v - 8:v + 8, u - 8:u + 8] = 0.10           # what the finger shows
    m = robot_mask(depth, K, boxes, [T])
    assert m[v, u] and m.sum() >= 150
    assert not m[5, 5] and not (m & (depth > 0.5)).any()     # the scene behind stays
    assert not robot_mask(depth, K, boxes, [None]).any()
    assert len(parse_boxes(DEFAULT_PIPER_FINGER_BOXES)) == 2
    with pytest.raises(ValueError):
        parse_boxes(["finger:1,2,3"])


# --------------------------------------------------------- offline from episodes
def _synthetic_scene_episode(n=3):
    """Camera at (0.3, 0, 0.45) looking along +x at a wall 0.6 m away: a stem band,
    a leaf blob and a target blob labelled on it."""
    H, W = 60, 80
    K = np.array([[60.0, 0, 40], [0, 60.0, 30], [0, 0, 1]])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = [0, -1, 0], [0, 0, -1], [1, 0, 0]       # optical x right, y down, z forward
    T[:3, 3] = [0.3, 0.0, 0.45]
    lab = np.zeros((H, W), np.uint8)
    lab[:, 38:42] = 1                                                     # stem
    lab[10:20, 50:62] = 3                                                 # leaf
    lab[24:28, 38:42] = 4                                                 # target on the stem
    return {"depth": np.full((n, H, W), 0.6, np.float16), "K": K, "T_world_cam": np.repeat(T[None], n, 0),
            "labels": np.repeat(lab[None], n, 0), "stamps": np.arange(n, dtype=float)}


def test_map_from_episode_places_classes_where_the_camera_saw_them():
    from scout_piper_scene_repr_py.offline import map_from_episode
    vmap = map_from_episode(_synthetic_scene_episode(), half_extent_m=0.3, voxel_size_m=0.01, stride=1)
    stem = vmap.occupied_points("stem")
    tgt = vmap.occupied_points("target")
    assert len(stem) > 10 and np.allclose(stem[:, 0].mean(), 0.9, atol=0.015)
    assert len(tgt) >= 3 and abs(tgt[:, 1].mean()) < 0.02 and len(vmap.occupied_points("leaf")) > 5
    g = target_goal(tgt, vmap.spec.voxel_size, robot_xy=[0.0, 0.0])
    assert g.pre_grasp[0] == pytest.approx(g.grasp_point[0] - 0.12, abs=0.01)
    with pytest.raises(KeyError):
        map_from_episode({"depth": np.zeros((1, 2, 2))})
