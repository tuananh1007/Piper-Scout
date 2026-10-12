"""Candidates, uncertainty, policies, VGGT helpers and a short episode (no ROS)."""

import numpy as np
import pytest

from scout_piper_active_perception.candidates import camera_pose_fn, move_time, sample_views
from scout_piper_active_perception.uncertainty import (Weights, assess, ratio_map, u_jepa_from_pixels,
                                                       u_jepa_from_state, u_twin_from_state)
from scout_piper_active_perception.vggt import (SparseViewBatcher, View, VGGTReconstructor, fuse_depth,
                                                is_novel_view, metric_scale, preprocess, reconstruct)
from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel

M = WholeBodyModel()
CAM_FN = camera_pose_fn(M)


def test_candidates_look_at_the_target_within_range():
    target = np.array([0.9, 0.1, 0.45])
    x = np.r_[0.2, 0.0, 0.0, 0.0, 1.2, -1.0, 0.0, 0.5, 0.0]
    c = sample_views(M, CAM_FN, target, x, n=20000, rng=np.random.default_rng(0))
    assert len(c.X) > 20 and c.arm_only.any() and (~c.arm_only).any()
    rel = target - c.T[:, :3, 3]
    depth = (rel * c.T[:, :3, 2]).sum(1)
    ang = np.degrees(np.arccos(np.clip(depth / np.linalg.norm(rel, axis=1), -1, 1)))
    assert (depth >= 0.2 - 1e-6).all() and (depth <= 0.45 + 1e-6).all()
    assert (ang[~c.arm_only] < 1e-3).all() and (ang[c.arm_only] < 10.0 + 1e-6).all()
    assert np.allclose(c.X[c.arm_only][:, :3], x[:3])
    assert (np.linalg.norm(c.X[:, :2] - x[:2], axis=1) <= 1.0 + 1e-9).all()


def test_move_time():
    x = np.zeros(9)
    t = move_time(M, x, np.array([x, np.r_[0.3, 0, 0, np.zeros(6)], np.r_[0, 0, 0, 0.6, np.zeros(5)]]))
    assert t[0] == pytest.approx(0.5)
    assert t[1] == pytest.approx(0.5 + 0.3 / M.scout.v_max)
    assert t[2] == pytest.approx(0.5 + 0.6 / M.arm.qd_max)


def test_uncertainty_terms():
    assert ratio_map(0.001, 0.001) == pytest.approx(0.5)
    u = assess(0.0005, 0.10, u_jepa=0.2)
    assert u.joint < 0.5 < u.corridor and u.needs_view and u.unmet() == ["corridor"]
    assert not assess(0.0005, 0.01, 0.9).needs_view            # JEPA not required by default
    assert assess(0.0005, 0.01, 0.9, required=("joint", "corridor", "jepa")).needs_view
    assert 0.0 <= u.total <= 1.0 and set(u.as_dict()) >= {"u_total", "needs_view", "unmet"}
    assert u_jepa_from_state({"status": "lost"}) == 1.0
    assert u_jepa_from_state({"status": "tracking", "confidence": 0.8}) == pytest.approx(0.2)
    assert u_jepa_from_state({"status": "tracking", "confidence": 0.8, "stamp": 0.0}, now=5.0) == 1.0
    assert u_jepa_from_pixels(300) == 0.0 and u_jepa_from_pixels(0) == 1.0
    assert u_twin_from_state({"touching": False, "confidence": 0.1}) == 0.0
    assert u_twin_from_state({"touching": True, "confidence": 0.7}) == pytest.approx(0.3)
    w = Weights(1.0, 0.0, 0.0)
    assert assess(0.0, 0.0, 0.6, weights=w).total == pytest.approx(0.6)


def test_view_batcher_and_fusion_with_a_fake_model():
    H, W = 48, 64
    K = np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])
    b = SparseViewBatcher(max_views=3, min_baseline_m=0.03, min_angle_deg=8.0)
    poses = []
    for i in range(5):
        T = np.eye(4)
        T[0, 3] = 0.05 * i
        poses.append(T)
        depth = np.full((H, W), 0.4)
        depth[:, 30:34] = np.nan                                # the thin stem the sensor misses
        assert b.offer(View(np.zeros((H, W, 3), np.uint8), depth, T, K, float(i)))
    assert not b.offer(View(np.zeros((H, W, 3), np.uint8), None, poses[-1], K, 9.0))
    assert len(b.views) == 3 and b.views[0].stamp == 0.0        # anchor kept, oldest others dropped
    assert not is_novel_view(poses[0], [poses[0]]) and is_novel_view(poses[0], [poses[2]])

    scale = 2.5                                                 # VGGT units -> metres

    def fake(x):
        S, _, h, w = x.shape
        d = np.full((S, h, w), 0.4 / scale, np.float32)
        d[:, :, int(30 * w / W):int(34 * w / W)] = 0.38 / scale    # the stem, slightly in front
        E = np.stack([np.c_[np.eye(3), -v.T_world_cam[:3, 3] / scale] for v in b.views])
        return {"depth": d, "depth_conf": np.full((S, h, w), 5.0, np.float32), "extrinsic": E,
                "intrinsic": np.repeat(K[None], S, 0)}
    x = preprocess(np.zeros((3, H, W, 3), np.uint8))
    assert x.shape[1] == 3 and x.shape[3] == 518 and x.shape[2] % 14 == 0
    model = VGGTReconstructor(model_fn=fake)
    pred = model(np.stack([v.rgb for v in b.views]))
    assert metric_scale(pred, b.views) == pytest.approx(scale, rel=1e-3)
    no_depth = [View(v.rgb, None, v.T_world_cam, v.K, v.stamp) for v in b.views]
    assert metric_scale(pred, no_depth) == pytest.approx(scale, rel=1e-3)   # from the camera baselines
    out = reconstruct(b, model)
    fused, src, agree = out["views"][0]
    assert (src[:, 31] == 2).all() and np.allclose(fused[:, 31], 0.38, atol=0.01)
    assert (src[:, 5] == 1).all() and agree == pytest.approx(1.0)
    f2, s2, a2 = fuse_depth(np.full((2, 2), np.nan), np.ones((2, 2)), np.zeros((2, 2)))
    assert (s2 == 0).all() and np.isnan(a2)


def test_short_episode_runs_and_scores():
    from scout_piper_active_perception.scene import Camera
    from scout_piper_active_perception.sim import run_episode
    r = run_episode(1, "fused", max_views=2, camera=Camera(image_hw=(120, 212)), n_candidates=60,
                    record_trace=True)
    assert r.views <= 2 and r.first_view_target_px >= 0
    assert r.min_hard_clearance_m > 0.0
    assert r.stop in ("requirements met", "max views", "no reachable view")
    assert len(r.trace) == r.views
