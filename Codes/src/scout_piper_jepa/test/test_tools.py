"""Episode export assembly, annotation, E1 runner and training helpers (no ROS, no torch)."""

from types import SimpleNamespace

import numpy as np
import pytest

from scout_piper_jepa.annotate import apply_polygons, import_masks, info, interpolate_polygons, rasterize
from scout_piper_jepa.e1 import h1_decision, run_method
from scout_piper_jepa.episode import EpisodeAssembler
from scout_piper_jepa.predictor import PersistencePredictor
from scout_piper_jepa.train import _segments, evaluate, training_episodes
from test_target_memory import _scene


def _stamp(t):
    return SimpleNamespace(stamp=SimpleNamespace(sec=int(t), nanosec=int(round((t % 1) * 1e9))),
                           frame_id="camera_color_optical_frame")


def _img(arr, enc, t):
    a = np.ascontiguousarray(arr)
    return SimpleNamespace(header=_stamp(t), height=a.shape[0], width=a.shape[1], encoding=enc,
                           step=a.reshape(a.shape[0], -1).view(np.uint8).shape[1], is_bigendian=0,
                           data=a.tobytes())


def _q(yaw):
    return SimpleNamespace(x=0.0, y=0.0, z=np.sin(yaw / 2), w=np.cos(yaw / 2))


# ------------------------------------------------------------ export assembly
def test_assembler_synchronises_depth_state_pose_and_segmentation():
    calls = []

    def tf(target, source, ts):
        calls.append((target, source, round(ts, 3)))
        T = np.eye(4)
        T[0, 3] = ts
        return T

    asm = EpisodeAssembler("/rgb", "/depth", "/info", seg_topic="/seg", tf_lookup=tf, stride=2)
    asm.add("/info", SimpleNamespace(k=[500.0, 0, 32, 0, 500.0, 24, 0, 0, 1]))
    names = [f"piper_joint{i}" for i in range(1, 9)]
    for k in range(6):
        t = 1.0 + 0.1 * k
        asm.add("/odom", SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=0.1 * k, y=0.0, z=0.0), orientation=_q(0.2)))))
        asm.add("/joint_states", SimpleNamespace(name=names, position=[0.01 * k] * 6 + [0.005 * k, -0.005 * k]))
        asm.add("/depth", _img(np.full((48, 64), 500 + k, np.uint16), "16UC1", t))
        asm.add("/seg", _img(np.full((48, 64), 255, np.uint8), "mono8", t + (0.01 if k == 2 else 0.2)))
        asm.add("/rgb", _img(np.full((48, 64, 3), k, np.uint8), "bgr8", t))
    asm.add("/seg", _img(np.full((48, 64), 255, np.uint8), "mono8", 1.4))   # recorded after its frame
    ep = asm.result()
    assert ep["frames"].shape == (3, 48, 64, 3) and list(ep["frames"][:, 0, 0, 0]) == [0, 2, 4]
    assert np.allclose(ep["states"][1], [0.2, 0.0, 0.2] + [0.02] * 6)
    assert np.allclose(ep["depth"][2], 0.504, atol=1e-3)                    # mm -> m, same frame
    assert list(ep["seg_masks"].reshape(3, -1).all(1)) == [False, True, True]    # within 50 ms, also late
    assert np.allclose(ep["T_world_cam"][:, 0, 3], [1.0, 1.2, 1.4])
    assert np.allclose(ep["gripper"], [0.0, 0.02, 0.04])                    # opening = joint7 − joint8
    assert ep["K"][0, 0] == 500.0 and calls[0][:2] == ("odom", "camera_color_optical_frame")
    with pytest.raises(ValueError):
        EpisodeAssembler("/rgb").result()


# ---------------------------------------------------------------- annotation
def test_polygons_interpolate_between_keyframes_and_hide():
    sq = lambda x: np.array([[x, 0], [x + 4, 0], [x + 4, 4], [x, 4]], float)  # noqa: E731
    per = interpolate_polygons({0: sq(0), 4: sq(8), 6: None, 9: sq(2)}, 10)
    assert np.allclose(per[2], sq(4)) and np.allclose(per[4], sq(8))
    assert per[6] is None and per[7] is None and per[8] is None and np.allclose(per[9], sq(2))
    tri = interpolate_polygons({0: sq(0), 4: np.array([[0, 0], [4, 0], [0, 4]], float)}, 5)
    assert len(tri[1]) == 4 and len(tri[3]) == 3                          # nearer keyframe copied
    m = rasterize(sq(2), (8, 10))
    assert m.sum() == 16 and m[0, 2] and m[3, 5] and not m[4, 2] and not m[0, 1]
    assert not rasterize(None, (8, 10)).any()
    first = apply_polygons(None, {0: sq(0), 2: sq(0)}, 6, (8, 10))           # first session: frames 0-2
    both = apply_polygons(first, {4: sq(2), 5: None}, 6, (8, 10))            # second: frames 4-5
    assert list(both.reshape(6, -1).any(1)) == [True, True, True, False, True, False]


def test_masks_import_from_png_files(tmp_path):
    cv2 = pytest.importorskip("cv2")
    ep_path = str(tmp_path / "ep.npz")
    np.savez_compressed(ep_path, frames=np.zeros((5, 8, 10, 3), np.uint8), target_masks=np.zeros((5, 8, 10), bool))
    d = tmp_path / "masks"
    d.mkdir()
    m = np.zeros((8, 10), np.uint8)
    m[2:5, 3:7] = 255
    cv2.imwrite(str(d / "000003.png"), m)
    cv2.imwrite(str(d / "frame_000001.png"), m)
    assert import_masks(ep_path, str(d)) == 2
    assert import_masks(ep_path, str(d), key="distractors") == 2
    i = info(ep_path)
    assert i["target_annotated_frames"] == 2 and i["distractor_instances"] == 1
    assert np.load(ep_path)["target_masks"][3, 3, 4]


# ------------------------------------------------------------------------ E1
def _episode(tmp_path, seg_ok=True):
    frames, tmasks, dmasks = _scene()
    H, W = frames[0].shape[:2]
    tm = np.stack([m if m is not None else np.zeros((H, W), bool) for m in tmasks])
    seg = tm.copy() if seg_ok else np.stack([d[0] for d in dmasks])            # framewise on the twin
    p = str(tmp_path / ("ok.npz" if seg_ok else "twin.npz"))
    np.savez_compressed(p, frames=np.stack(frames), target_masks=tm, seg_masks=seg,
                        distractors=np.stack([np.stack(d) for d in dmasks]))
    return p


def test_e1_methods_run_and_the_decision_rule(tmp_path):
    good, twin = _episode(tmp_path), _episode(tmp_path, seg_ok=False)
    t0 = run_method({"kind": "segmentation"}, [good], "cpu")
    assert t0["id_retention"] == 1.0 and t0["false_switch_rate"] == 0.0
    assert run_method({"kind": "segmentation"}, [twin], "cpu")["false_switch_rate"] > 0.5
    ref = run_method({"kind": "memory", "encoder": "color_patch"}, [good], "cpu")
    assert ref["id_retention"] > 0.8 and ref["episodes"] == 1
    assert "skipped" in run_method({"kind": "memory", "encoder": "vjepa", "hub_entry": ""}, [good], "cpu")
    pytest.importorskip("cv2")
    t1 = run_method({"kind": "optical_flow"}, [good], "cpu")
    assert 0.0 <= t1["id_retention"] <= 1.0
    res = {"T0": {"id_retention": 0.70, "occlusion_recovery_frames": 3.0},
           "T2": {"id_retention": 0.80, "occlusion_recovery_frames": 2.0},
           "T4": {"id_retention": 0.86, "occlusion_recovery_frames": 2.0}}
    assert h1_decision(res)["decision"].startswith("H1 supported") and h1_decision(res)["best_baseline"] == "T2"
    res["T4"]["id_retention"] = 0.82
    assert h1_decision(res)["decision"].startswith("H1 not supported")
    res["T4"]["occlusion_recovery_frames"] = 0.5
    assert h1_decision(res)["decision"].startswith("H1 supported")
    assert h1_decision({"T4": {"skipped": "x"}})["decision"] == "incomplete"


# --------------------------------------------------------------------- train
def test_training_episodes_split_on_missing_states_and_score_persistence():
    assert _segments(np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1], bool), 3) == [slice(0, 3), slice(4, 8)]
    T, H, W, hf, wf, C = 12, 48, 64, 6, 8, 4
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(T, hf, wf, C)).astype(np.float32)
    tm = np.zeros((T, H, W), bool)
    tm[:, 16:24, 24:32] = True
    states = np.tile(np.r_[0.0, 0.0, 0.0, 0.0, 1.2, -1.0, 0.0, 0.5, 0.0], (T, 1))
    states[:, 0] = np.linspace(0, 0.2, T)
    states[5] = np.nan
    grip = np.linspace(0.06, 0.0, T)                                       # gripper closing
    ep = {"frames": np.zeros((T, H, W, 3), np.uint8), "target_masks": tm, "states": states, "gripper": grip}
    fk = lambda q: np.broadcast_to(np.eye(4), np.shape(q)[:-1] + (4, 4))       # noqa: E731
    eps = training_episodes(ep, Z, fk, min_len=4)
    assert [len(e["Z"]) for e in eps] == [5, 6]
    e = eps[1]
    assert e["A"].shape == (5, 9) and e["target"].shape == (6, hf, wf) and e["visible"].all()
    assert np.allclose(e["u"][0], [28.0, 20.0])                              # target centre in pixels
    assert np.allclose(e["A"][:, 8], np.diff(grip)[6:11])                    # Δg from the recorded opening
    assert np.allclose(e["S"], states[6:12, 3:9])
    s = evaluate(PersistencePredictor(), eps, K=1, H=2, every=1)
    assert s["n_windows"] > 0 and s["E_target@1"] >= 0.0
