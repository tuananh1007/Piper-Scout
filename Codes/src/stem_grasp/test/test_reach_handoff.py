"""Reach-then-servo handoff (P0.4.11): goal transform and the reach monitor."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stem_grasp.core import select_grasp_candidates
from stem_grasp.reach_handoff import ReachMonitor, candidate_goal_in_world


def test_goal_in_world_moves_position_and_approach_axis():
    stem = np.stack([np.full(20, 0.6), np.zeros(20), np.linspace(0.2, 0.5, 20)], 1)
    cand = select_grasp_candidates(stem, target_offset=0.12)[0]
    T = np.eye(4)                                     # planning frame 1 m ahead, turned 90 deg
    T[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    T[:3, 3] = [1.0, 0.0, 0.18]
    p, quat = candidate_goal_in_world(cand["pre_pos"], cand["quat"], T)
    assert p == pytest.approx(T[:3, :3] @ cand["pre_pos"] + T[:3, 3], abs=1e-9)
    z_planning = Rotation.from_quat(cand["quat"]).as_matrix()[:, 2]
    z_world = Rotation.from_quat(quat).as_matrix()[:, 2]
    assert z_world == pytest.approx(T[:3, :3] @ z_planning, abs=1e-6)
    # the gripper z axis points from the pre-grasp position to the stem point
    to_stem = cand["pos"] - cand["pre_pos"]
    assert z_planning == pytest.approx(to_stem / np.linalg.norm(to_stem), abs=1e-6)


def test_handoff_after_reached_is_held():
    m = ReachMonitor(start_t=0.0, settle_s=1.0)
    assert m.update(None, None, 0.5) == "wait"                 # no status yet
    assert m.update("whole_body", 1.0, 1.0) == "wait"
    assert m.update("reached", 2.0, 2.0) == "wait"
    assert m.update("whole_body", 2.5, 2.5) == "wait"          # left the goal: restart settling
    assert m.update("reached", 3.0, 3.0) == "wait"
    assert m.update("reached", 3.6, 3.6) == "wait"
    assert m.update("reached", 4.1, 4.1) == "handoff"


@pytest.mark.parametrize("mode, status_t, now, kw", [
    (None, None, 11.0, {}),                          # MPC never answered
    ("reached", -1.0, 11.0, {}),                     # only a status from before the goal
    ("whole_body", 5.0, 7.5, {}),                    # status stopped arriving
    ("whole_body", 60.5, 60.5, {"timeout_s": 60.0}),  # never reached
])
def test_reach_times_out(mode, status_t, now, kw):
    assert ReachMonitor(start_t=0.0, **kw).update(mode, status_t, now) == "timeout"


def _stem_surface(x, y, z0, z1, r=0.004, n=3000, seed=0):
    """Camera-facing half of a vertical cylinder in a z-up frame, as the depth camera sees it."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(z0, z1, n)
    a = rng.uniform(-np.pi / 2, np.pi / 2, n)
    return (np.stack([x - r * np.cos(a), y + r * np.sin(a), z], 1)
            + rng.normal(0, 0.0005, (n, 3))).astype(np.float32)


def test_main_stem_in_a_z_up_frame_needs_vertical_axis_2():
    from stem_grasp.core import extract_main_stem, skeletonize_plant_points
    G, sk = skeletonize_plant_points(_stem_surface(0.8, 0.15, 0.07, 0.42), voxel_size=0.003)
    stem = extract_main_stem(G, sk, vertical_axis=2)
    assert np.ptp(stem[:, 2]) > 0.25                     # follows the 0.35 m stem
    assert len(extract_main_stem(G, sk, vertical_axis=1)) < 6   # the ROS 1 default does not


def test_approach_hint_puts_the_pre_grasp_on_the_robot_side():
    stem = np.stack([np.full(40, 0.8), np.full(40, 0.15), np.linspace(0.07, 0.42, 40)], 1)
    for c in select_grasp_candidates(stem, target_offset=0.12, approach_hint=-stem.mean(axis=0)):
        approach = (c["pre_pos"] - c["pos"]) / 0.12
        assert abs(approach @ np.array([0.0, 0.0, 1.0])) < 1e-6            # perpendicular to the stem
        assert np.linalg.norm(c["pre_pos"][:2]) < np.linalg.norm(c["pos"][:2])   # closer to the arm
    side = select_grasp_candidates(stem, target_offset=0.12)[0]             # no hint: ROS 1 rule
    assert (side["pre_pos"] - side["pos"]) / 0.12 == pytest.approx([0.0, 1.0, 0.0])
