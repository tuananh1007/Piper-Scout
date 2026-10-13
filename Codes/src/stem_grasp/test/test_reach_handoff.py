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


def test_handoff_within_the_pipeline_tolerance_without_mpc_reached():
    """The MPC can creep at 1.1-1.3 cm from its goal for seconds; 2 cm and 15 deg
    are enough for the servo and the final approach."""
    mon = ReachMonitor(start_t=0.0, settle_s=1.0)
    assert mon.update("handoff_ready", 1.0, 1.0, tcp_error_m=0.012, angle_deg=13.0) == "wait"
    assert mon.update("handoff_ready", 2.1, 2.1, tcp_error_m=0.011, angle_deg=12.0) == "handoff"
    mon = ReachMonitor(start_t=0.0, settle_s=1.0)
    for t in (1.0, 2.5, 4.0):        # too far, or the approach axis still off
        assert mon.update("handoff_ready", t, t, tcp_error_m=0.03, angle_deg=5.0) == "wait"
        assert mon.update("handoff_ready", t, t, tcp_error_m=0.005, angle_deg=20.0) == "wait"
    assert mon.update("idle", 5.0, 5.0, tcp_error_m=0.0, angle_deg=0.0) == "wait"   # cancelled


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


def _half_shell(center, radius, rng, n=800, facing=(-1.0, 0.0)):
    """Single-view stem cloud: the half of a vertical cylinder facing ``facing``."""
    base = np.arctan2(facing[1], facing[0])
    a = base + rng.uniform(-np.pi / 2, np.pi / 2, n)
    z = rng.uniform(0.25, 0.60, n)
    pts = np.column_stack([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a), z])
    return pts + rng.normal(0, 0.0005, pts.shape)


@pytest.mark.parametrize("radius", [0.002, 0.004, 0.008])
def test_stem_axis_point_recovers_the_axis_from_the_visible_half(radius):
    from stem_grasp.core import stem_axis_point
    rng = np.random.default_rng(1)
    cloud = _half_shell((1.25, 0.0), radius, rng)
    surface = np.array([1.25 - radius, 0.0, 0.40])            # where the skeleton sits
    centre, r = stem_axis_point(cloud, surface, [0.0, 0.0, 1.0])
    assert np.linalg.norm(centre[:2] - [1.25, 0.0]) < 0.0015
    assert centre[2] == pytest.approx(0.40)                   # height along the stem kept
    assert r == pytest.approx(radius, abs=0.0015)


def test_stem_axis_point_refuses_a_poor_fit():
    from stem_grasp.core import stem_axis_point
    rng = np.random.default_rng(2)
    flat = np.column_stack([np.full(400, 1.25), rng.uniform(-0.03, 0.03, 400), rng.uniform(0.3, 0.5, 400)])
    assert stem_axis_point(flat, [1.25, 0.0, 0.4], [0.0, 0.0, 1.0]) is None   # a wall, not a stem
    assert stem_axis_point(flat[:5], [1.25, 0.0, 0.4], [0.0, 0.0, 1.0]) is None


def test_view_gate_requests_views_then_proceeds():
    from stem_grasp.reach_handoff import ViewGate
    g = ViewGate(max_views=2, settle_s=1.0, request_timeout_s=5.0, startup_wait_s=3.0)
    assert g.decide(0.0, None, None) == "wait"                             # planner may still be starting
    assert g.decide(2.9, None, None) == "wait"
    assert g.decide(3.1, None, None) == "proceed"                          # active perception not running
    assert g.decide(4.0, {"field": True}, 3.9) == "wait"                   # running, no estimate yet
    assert g.decide(7.1, {"field": True}, 7.0) == "proceed"                # still none after startup_wait_s
    assert g.decide(10.0, {"needs_view": True}, 6.0) == "proceed"         # status too old
    assert g.decide(10.0, {"needs_view": False}, 9.5) == "proceed"
    assert g.decide(10.0, {"needs_view": True}, 9.5) == "request"
    g.requested(10.0)
    assert g.decide(11.0, {"needs_view": True}, 10.9) == "wait"            # view pending
    g.view_started()
    g.view_done(20.0)
    assert g.decide(20.5, {"needs_view": True}, 20.4) == "wait"            # settling
    assert g.decide(21.2, {"needs_view": True}, 20.9) == "wait"            # status from before the settle
    assert g.decide(21.5, {"needs_view": True}, 21.3) == "request"
    g.requested(21.5)
    g.view_started()
    g.view_done(30.0)
    assert g.views == 2 and g.decide(40.0, {"needs_view": True}, 39.9) == "proceed"   # at most 2 views
    g.reset()
    assert g.decide(50.0, {"needs_view": True}, 49.9) == "request"
    g.requested(50.0)
    assert g.decide(56.0, {"needs_view": True}, 55.9) == "proceed" and g.gave_up == "no view published"
    g.reset()
    g.failed("plan_view service not available")
    assert g.decide(60.0, {"needs_view": True}, 59.9) == "proceed"


def test_view_gate_returns_to_the_scan_pose_without_a_candidate():
    from stem_grasp.reach_handoff import ViewGate
    g = ViewGate(max_views=1, settle_s=1.0, return_after_s=3.0, startup_wait_s=0.0)
    assert not g.should_return(10.0)                                       # no view yet
    assert g.decide(0.0, {"needs_view": True}, 0.0) == "request"
    g.requested(0.0)
    g.view_started()
    g.view_done(10.0)
    assert g.decide(11.5, {"needs_view": True}, 11.2) == "proceed"         # at most 1 view
    assert not g.should_return(12.0)                                       # still within return_after_s
    assert g.should_return(13.5)                                           # still scanning: go back
    g.returned = True
    assert not g.should_return(20.0)                                       # once only
    g2 = ViewGate(max_views=4, startup_wait_s=0.0)
    g2.requested(0.0)
    g2.view_started()
    g2.view_done(1.0)
    g2.returned = True
    assert g2.decide(10.0, {"needs_view": True}, 9.9) == "proceed"         # no views after the return
