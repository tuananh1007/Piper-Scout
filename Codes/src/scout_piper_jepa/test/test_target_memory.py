"""Stage A target memory on a synthetic two-flower scene (no ROS, no torch).

    cd Codes && python -m pytest src/scout_piper_jepa/test -q
"""

import numpy as np

from scout_piper_jepa.action import action_embedding
from scout_piper_jepa.encoder import ColorPatchEncoder
from scout_piper_jepa.metrics import score_sequence
from scout_piper_jepa.target_memory import (TargetMemory, TargetMemoryConfig,
                                            cell_centers_px, mask_to_grid)

H, W = 240, 320


def _disk(cx, cy, r=18):
    yy, xx = np.mgrid[:H, :W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _scene(n=40, occlude=range(15, 21), seed=0):
    """Target moves right 4 px/frame; an identical distractor sits still at
    x=285, so the two disks stay at least one 16 px patch apart (touching
    instances are a separate, harder case). A grey bar hides the target for
    the frames in ``occlude``."""
    rng = np.random.default_rng(seed)
    frames, tmasks, dmasks = [], [], []
    for k in range(n):
        img = np.zeros((H, W, 3), np.uint8)
        img[...] = (40, 120, 40)
        img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
        t = _disk(70 + 4 * k, 120)
        d = _disk(285, 125)
        img[t] = (220, 40, 60)
        img[d] = (220, 40, 60)
        if k in occlude:
            bar = np.zeros((H, W), bool)
            bar[:, 120 + 4 * (k - 15) - 40: 120 + 4 * (k - 15) + 60] = True
            img[bar] = (128, 128, 128)
            t = t & ~bar
        frames.append(img)
        tmasks.append(t if t.sum() > 50 else None)
        dmasks.append([d])
    return frames, tmasks, dmasks


def _run(cfg, frames, first_mask, enc):
    mem = TargetMemory(cfg)
    clip = [frames[0]]
    mem.initialize(enc.encode(clip), first_mask)
    preds, states = [], []
    for k, f in enumerate(frames):
        clip = (clip + [f])[-2:]
        s = mem.update(enc.encode(clip), stamp=k * 0.1)
        states.append(s)
        preds.append(s.u_mean if s.visible else None)
    return preds, states


def test_grid_helpers():
    m = np.zeros((32, 48), bool); m[:16, :24] = True
    g = mask_to_grid(m, (2, 3))
    assert np.allclose(g, [[1, 0.5, 0], [0, 0, 0]])
    c = cell_centers_px((2, 3), (32, 48))
    assert np.allclose(c[0, 0], [8, 8]) and np.allclose(c[1, 2], [40, 24])


def test_keeps_identity_through_occlusion_with_twin_distractor():
    frames, tm, dm = _scene()
    preds, states = _run(TargetMemoryConfig(), frames, tm[0], ColorPatchEncoder(16))
    sc = score_sequence(preds, tm, dm)
    assert sc.id_retention > 0.9, sc
    assert sc.false_switch_rate == 0.0, sc
    assert sc.occlusion_recovery_frames <= 2, sc
    assert any(s.status == "occluded" for s in states[15:21])
    assert states[-1].status == "tracking"


def test_without_motion_prior_twin_is_ambiguous():
    """Ablation: with no spatial prior, two identical instances split the
    probability mass, so the distribution is far more uncertain."""
    frames, tm, _ = _scene(occlude=())
    _, with_prior = _run(TargetMemoryConfig(), frames, tm[0], ColorPatchEncoder(16))
    _, no_prior = _run(TargetMemoryConfig(prior_sigma_px=None), frames, tm[0], ColorPatchEncoder(16))
    h_with = np.mean([s.entropy_norm for s in with_prior])
    h_without = np.mean([s.entropy_norm for s in no_prior])
    assert h_without > h_with + 0.05, (h_with, h_without)


def test_lost_after_long_absence_and_reground():
    frames, tm, _ = _scene(n=30, occlude=())
    cfg = TargetMemoryConfig(lost_after_frames=5)
    enc = ColorPatchEncoder(16)
    mem = TargetMemory(cfg)
    mem.initialize(enc.encode([frames[0]]), tm[0])
    blank = np.full_like(frames[0], (40, 120, 40))
    for _ in range(6):
        s = mem.update(enc.encode([blank]))
    assert s.status == "lost" and not s.visible and s.confidence == 0.0
    mem.reinitialize(enc.encode([frames[10]]), tm[10])
    s = mem.update(enc.encode([frames[11]]))
    assert s.status == "tracking" and s.visible


def test_target_3d_from_depth():
    frames, tm, _ = _scene(n=3, occlude=())
    enc = ColorPatchEncoder(16)
    mem = TargetMemory()
    mem.initialize(enc.encode([frames[0]]), tm[0])
    K = np.array([[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1]])
    depth = np.full((H, W), 0.5, np.float32)
    T = np.eye(4); T[:3, 3] = [1.0, 2.0, 3.0]
    s = mem.update(enc.encode(frames[:2]), depth=depth, K=K, T_world_cam=T)
    assert s.p_world is not None
    assert abs(s.p_world[2] - 3.5) < 1e-6                 # z = depth + offset
    u = s.u_mean
    assert abs(s.p_world[0] - (1.0 + (u[0] - W / 2) * 0.5 / 300)) < 0.02
    # no valid depth -> no metric estimate, not a hallucinated one
    s2 = mem.update(enc.encode(frames[1:3]), depth=np.full((H, W), np.nan), K=K, T_world_cam=T)
    assert s2.p_world is None
    # no camera pose (TF gap) -> no world position, not a camera-frame one
    assert mem.update(enc.encode(frames[1:3]), depth=depth, K=K).p_world is None


def test_action_embedding_base_and_arm():
    def fk(q):                       # toy arm: EE 0.5 m ahead, q[0] = yaw joint
        T = np.eye(4); c, s = np.cos(q[0]), np.sin(q[0])
        T[:2, :2] = [[c, -s], [s, c]]
        T[:3, 3] = [0.5 * c, 0.5 * s, 0.3 + q[1]]
        return T
    q = np.zeros(6)
    a = action_embedding(q, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]), 0.1, fk)
    assert np.allclose(a[:2], [0.1, 0.0]) and np.allclose(a[2:5], [0.1, 0, 0])
    a = action_embedding(q, np.array([0, 0, 0, 0.2, 0, 0, 0, 0]), 0.5, fk)
    assert np.allclose(a[:2], 0) and np.allclose(a[2:5], [0, 0, 0.1])
    a = action_embedding(q, np.array([0, 1.0, 0, 0, 0, 0, 0, 0]), 0.1, fk)
    assert np.isclose(a[1], 0.1) and np.isclose(a[7], 0.1)    # yaw shows in Δr_z
