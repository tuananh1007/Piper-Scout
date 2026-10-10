"""End pose chosen for the view (view_pose.py; synthetic world, no ROS)."""

import os
import sys

import numpy as np
import pytest

from scout_piper_jepa.synthetic import TARGET, eye_in_hand_pose, make_flower_scene
from scout_piper_jepa.view_pose import (choose_view_end_pose, goal_poses, line_of_sight_visible_fn,
                                        render_visible_fn, sphere_occluder, view_angle)
from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel

M = WholeBodyModel()
GOAL = np.array([0.70, 0.0, 0.45])


def test_goal_poses_put_the_tcp_on_the_goal():
    X = goal_poses(M, GOAL, n=20000)
    assert len(X) > 50
    err = np.linalg.norm(M.tcp_world(X)[:, :3, 3] - GOAL, axis=1)
    assert err.max() < 0.011                                    # z within 1 cm, x and y solved exactly


def test_sphere_occluder_blocks_only_rays_through_a_sphere():
    occ = sphere_occluder(np.array([[0.5, 0.0, 0.0]]), np.array([0.05]))
    origins = np.array([[0.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.8, 0.0, 0.0]])
    assert occ(origins, np.array([1.0, 0.0, 0.0])).tolist() == [True, False, False]   # behind / beside / past it
    assert not occ(origins[:1], np.array([0.4, 0.0, 0.0]))[0]                         # target before the sphere


def test_line_of_sight_agrees_with_rendering_when_it_says_visible():
    world = make_flower_scene()
    cam = eye_in_hand_pose(M)
    X = goal_poses(M, GOAL, n=40000)[:600]
    T = cam(X)
    other = world.labels != TARGET
    geo = line_of_sight_visible_fn(world.target_position(), sphere_occluder(world.positions[other],
                                                                            world.radii[other]),
                                   half_fov_rad=np.radians(30.0))(X, T)
    ren = render_visible_fn(world)(X, T)
    assert geo.sum() > 30
    assert (geo & ren).sum() / geo.sum() > 0.85                 # conservative: "visible" is nearly always true


@pytest.mark.parametrize("oracle", [False, True])
def test_chosen_end_pose_sees_the_flower(oracle):
    world = make_flower_scene()
    cam = eye_in_hand_pose(M)
    tgt = world.target_position()
    other = world.labels != TARGET
    vis = render_visible_fn(world) if oracle else line_of_sight_visible_fn(
        tgt, sphere_occluder(world.positions[other], world.radii[other]), np.radians(30.0))
    x_now = np.r_[0.0, 0.0, 0.0, 0.0, 1.2, -1.0, 0.0, 0.5, 0.0]
    ch = choose_view_end_pose(M, cam, GOAL, tgt, x_now, vis, n=40000)
    assert ch is not None and ch.visible > 0
    T = cam(ch.state[None])
    assert render_visible_fn(world)(ch.state[None], T)[0]       # the rendered end view has the flower
    assert np.allclose(M.tcp_world(ch.state)[:3, 3], GOAL, atol=0.011)
    assert ch.angle_deg == pytest.approx(np.degrees(view_angle(T, tgt)[0][0]), abs=1e-6)
    assert np.isclose(np.linalg.norm(ch.axis), 1.0)


def test_field_occluder_sees_a_leaf_between_camera_and_target():
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "scout_piper_scene_repr", "python"))
    field = pytest.importorskip("scout_piper_scene_repr_py.field")
    from scout_piper_jepa.view_pose import field_occluder  # noqa: PLC0415
    vs, n = 0.01, 40
    origin = np.array([0.4, -0.2, 0.25])
    g = origin + (np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1) + 0.5) * vs
    leaf = np.maximum(np.abs(g[..., 0] - 0.62) - 0.004, np.linalg.norm(g[..., 1:] - [-0.05, 0.47], axis=-1) - 0.06)
    snap = field.DistanceFieldSnapshot(origin=origin, voxel_size=vs, shape=(n,) * 3, stamp=0.0, hard_classes=["stem"],
                                       hard_distance=np.full((n,) * 3, 1.0, np.float32),
                                       hard_class=np.zeros((n,) * 3, np.uint8), soft_class="leaf",
                                       soft_distance=leaf.astype(np.float32), age_ds=np.zeros((n,) * 3, np.uint16))
    occ = field_occluder(snap)
    tgt = np.array([0.75, -0.05, 0.47])
    out = occ(np.array([[0.45, -0.05, 0.47], [0.45, 0.12, 0.47]]), tgt)
    assert out.tolist() == [True, False]                        # through the leaf / past its edge
