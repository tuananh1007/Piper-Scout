"""P3B.9: train the Stage B predictors (P0 / P2 / P3) and score them on E3.

    # recorded episodes (episode.py export + annotate, with states): features from V-JEPA on the GPU
    python3 -m scout_piper_jepa.train EP*.npz --encoder vjepa --hub-entry vjepa2_vit_large \\
        --image-size 256 --device cuda --steps 3000 --out-dir runs/e3_vjepa

    # pipeline check without data or GPU (synthetic world, numpy features)
    python3 -m scout_piper_jepa.train --synthetic 300 --steps 300 --device cpu --out-dir runs/e3_smoke

Per episode the dense features are computed once and cached next to it
(``<episode>.feat_<encoder>_<entry>.npz``). Actions Γ come from consecutive
whole-body states (``action.action_from_states`` with the Piper FK of
scout_piper_whole_body_mpc); target / plant masks are reduced to the feature
grid. Episodes are split (not windows) into training and test sets
(``--test-frac``). Each method is saved as ``<out-dir>/<method>.pt``
(``TorchACPredictor.load``) and scored against persistence with the E3
metrics; ``results.json`` holds everything.

The checkpoint records the time between training frames (``step_s``, the
median of the episode stamps; synthetic: stride × the MPC step). A predictor
step of the MPC must span the same time: the predictive MPC node derives
``jepa_stride`` from it (``jepa_stride / rate_hz`` = ``step_s``), so export
the episodes with ``--stride`` giving a multiple of the MPC period (camera
at 30 fps, MPC at 10 Hz: ``--stride 6`` → 0.2 s, ``jepa_stride`` 2).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .action import action_from_states
from .predictor import PersistencePredictor
from .prediction_metrics import score_predictions
from .synthetic import BACKGROUND, STEM, TARGET
from .target_memory import cell_centers_px, mask_to_grid


def _fk():
    from scout_piper_whole_body_mpc.dynamics.piper import PiperKinematics  # noqa: PLC0415
    return PiperKinematics().tcp


# ------------------------------------------------------------ features
def episode_features(path: str, encoder, tag: str, clip_length: int = 2) -> np.ndarray:
    cache = path[:-4] + f".feat_{tag}.npz" if path.endswith(".npz") else path + f".feat_{tag}.npz"
    if os.path.exists(cache):
        return np.load(cache)["Z"]
    frames = np.load(path)["frames"]
    Z, clip = [], []
    for f in frames:
        clip = (clip + [f])[-clip_length:]
        Z.append(encoder.encode(clip))
    Z = np.stack(Z).astype(np.float16)
    np.savez_compressed(cache, Z=Z)
    return Z


def _segments(valid: np.ndarray, min_len: int) -> List[slice]:
    """Contiguous runs of valid frames at least min_len long."""
    out, start = [], None
    for i, v in enumerate(list(valid) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append(slice(start, i))
            start = None
    return out


def training_episodes(ep: dict, Z: np.ndarray, fk, min_len: int) -> List[dict]:
    """Episode file + features -> training dicts (Z, A, target, plant, label, u, visible),
    one per contiguous run of frames with a valid state."""
    states = np.asarray(ep["states"], float)
    T, H, W = ep["frames"].shape[:3]
    grid = Z.shape[1:3]
    tgt = np.stack([mask_to_grid(m, grid) >= 0.3 for m in ep["target_masks"]])
    pm = ep.get("plant_masks", ep.get("seg_masks"))
    plant = np.stack([mask_to_grid(m, grid) >= 0.3 for m in pm]) if pm is not None else np.zeros_like(tgt)
    centers = cell_centers_px(grid, (H, W)).reshape(-1, 2)
    out = []
    for sl in _segments(np.isfinite(states).all(1), min_len):
        t = tgt[sl]
        n = t.reshape(len(t), -1).sum(1)
        u = (t.reshape(len(t), -1, 1) * centers[None]).sum(1) / np.maximum(n, 1)[:, None]
        X = states[sl]
        out.append({"Z": Z[sl].astype(np.float32), "A": action_from_states(X[:-1], X[1:], fk),
                    "target": t, "plant": plant[sl] | t,
                    "label": np.where(t, TARGET, np.where(plant[sl], STEM, BACKGROUND)),
                    "u": u, "visible": n > 0, "image_hw": (H, W)})
    return out


def target_descriptor(ep: dict) -> Optional[np.ndarray]:
    """Mean feature over the target cells of the first frame the target is visible."""
    for Z, t in zip(ep["Z"], ep["target"]):
        if t.any():
            r = Z[t].mean(0)
            return r / max(np.linalg.norm(r), 1e-8)
    return None


# ----------------------------------------------------------- evaluation
def evaluate(pred, episodes: Sequence[dict], K: int, H: int, every: int = 3,
             r_fixed: Optional[np.ndarray] = None) -> Dict[str, float]:
    """E3 scores, averaged over episodes weighted by their number of windows."""
    acc, total = {}, 0
    for ep in episodes:
        r = r_fixed if r_fixed is not None else target_descriptor(ep)
        if r is None:
            continue
        Zp, Zt, Lt, Ut, Vt, U0 = [], [], [], [], [], []
        for t0 in range(K - 1, len(ep["Z"]) - H, every):
            if not ep["visible"][t0]:
                continue
            Zp.append(pred.rollout(ep["Z"][None, t0 - K + 1:t0 + 1], ep["A"][None, t0:t0 + H])[0])
            sl = slice(t0 + 1, t0 + H + 1)
            Zt.append(ep["Z"][sl]); Lt.append(ep["label"][sl]); Ut.append(ep["u"][sl])
            Vt.append(ep["visible"][sl]); U0.append(ep["u"][t0])
        if not Zp:
            continue
        s = score_predictions(np.array(Zp), np.array(Zt), np.array(Lt), np.array(Ut), np.array(Vt), r,
                              np.array(U0), ep["image_hw"]).summary()
        n = len(Zp)
        for k, v in s.items():
            if np.isfinite(v):
                a = acc.setdefault(k, [0.0, 0])
                a[0] += v * n
                a[1] += n
        total += n
    out = {k: round(a[0] / a[1], 3) for k, a in acc.items() if a[1]}
    out["n_windows"] = total
    return out


# --------------------------------------------------------------- inputs
def synthetic_split(n_train: int, n_test: int, frames: int = 14, stride: int = 2, seed: int = 0):
    from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel  # noqa: PLC0415
    from .synthetic import eye_in_hand_pose, make_flower_scene, synthetic_episodes, target_truth  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    m = WholeBodyModel()
    world = make_flower_scene(seed=seed)
    cam = eye_in_hand_pose(m)
    cand = np.concatenate([rng.normal(0, [0.05, 0.05, 0.1], (20000, 3)),
                           rng.uniform(m.kin.lower, m.kin.upper, (20000, 6))], 1)
    _, vis = target_truth(world.render(cam(cand))["label"], world.camera)
    starts = cand[vis]
    while len(starts) < n_train + n_test:
        pert = np.repeat(starts, 4, 0) + rng.normal(0, [0.03, 0.03, 0.05] + [0.12] * 6, (len(starts) * 4, 9))
        pert[:, 3:] = np.clip(pert[:, 3:], m.kin.lower, m.kin.upper)
        _, v2 = target_truth(world.render(cam(pert))["label"], world.camera)
        starts = np.concatenate([starts, pert[v2]])
    rng.shuffle(starts)
    kw = dict(frames=frames, stride=stride, fk=m.kin.tcp, rng=rng, scale=0.3)
    eps = synthetic_episodes(world, m, cam, starts[:n_train + n_test], **kw)
    for e in eps:
        e["image_hw"] = world.camera.image_hw
    r = world.features[world.labels == TARGET][0]
    return eps[:n_train], eps[n_train:], r, stride * m.dt


def recorded_split(paths: Sequence[str], encoder, tag: str, test_frac: float, min_len: int, seed: int):
    fk = _fk()
    eps_per_file, dts = [], []
    for p in paths:
        ep = dict(np.load(p))
        if "states" not in ep:
            raise SystemExit(f"{p}: no states (export the bag with the robot running; E3 needs them)")
        if "stamps" in ep and len(ep["stamps"]) > 1:
            dts.append(np.diff(np.asarray(ep["stamps"], float)))
        Z = episode_features(p, encoder, tag)
        eps_per_file.append(training_episodes(ep, Z, fk, min_len))
    order = np.random.default_rng(seed).permutation(len(paths))
    n_test = max(1, int(round(test_frac * len(paths)))) if len(paths) > 1 else 0
    test = [e for i in order[:n_test] for e in eps_per_file[i]]
    train = [e for i in order[n_test:] for e in eps_per_file[i]]
    step_s = float(np.median(np.concatenate(dts))) if dts else 0.0
    return train, test or train, None, step_s


def main(argv=None) -> None:
    from .encoder import make_encoder  # noqa: PLC0415
    from .torch_predictor import ACPredictorConfig, train_predictor  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", nargs="*")
    ap.add_argument("--synthetic", type=int, default=0, help="N synthetic training episodes instead of files")
    ap.add_argument("--encoder", default="color_patch", choices=["color_patch", "vjepa", "dinov2"])
    ap.add_argument("--hub-entry", default="")
    ap.add_argument("--image-size", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--methods", default="P0,P2,P3")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--history", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=8, help="evaluation horizon (predictor steps)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="runs/e3")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.perf_counter()
    if a.synthetic:
        train, test, r, step_s = synthetic_split(a.synthetic, max(a.synthetic // 5, 10), seed=a.seed)
    else:
        if not a.episodes:
            ap.error("give episode files or --synthetic N")
        kw = {}
        if a.encoder != "color_patch":
            kw = {"device": a.device, **({"hub_entry": a.hub_entry} if a.hub_entry else {}),
                  **({"image_size": a.image_size} if a.image_size else {})}
        enc = make_encoder(a.encoder, **kw)
        tag = a.encoder + (f"_{a.hub_entry}" if a.hub_entry else "")
        train, test, r, step_s = recorded_split(a.episodes, enc, tag, a.test_frac, a.history + a.horizon + 1,
                                                a.seed)
    if not train:
        raise SystemExit("no usable training episodes")
    grid, C = train[0]["Z"].shape[1:3], train[0]["Z"].shape[-1]
    res = {"setup": {"train_episodes": len(train), "test_episodes": len(test), "grid_hw": list(grid),
                     "feat_dim": int(C), "encoder": "synthetic" if a.synthetic else a.encoder,
                     "hub_entry": a.hub_entry, "steps": a.steps, "device": a.device,
                     "step_s": round(step_s, 4),
                     "data_s": round(time.perf_counter() - t0, 1)}}
    K, H = a.history, a.horizon
    res["persistence"] = evaluate(PersistencePredictor(), test, K, H, r_fixed=r)
    print(json.dumps({"persistence": res["persistence"]}), flush=True)
    makers = {"P0": ACPredictorConfig.p0, "P2": ACPredictorConfig.p2, "P3": ACPredictorConfig.p3}
    for name in a.methods.split(","):
        cfg = makers[name](grid_hw=tuple(grid), feat_dim=int(C), history=K, steps=a.steps, seed=a.seed,
                           device=a.device, step_s=step_s)
        t = time.perf_counter()
        model = train_predictor(train, cfg)
        model.save(os.path.join(a.out_dir, f"{name}.pt"))
        res[name] = {**evaluate(model, test, K, H, r_fixed=r), "train_s": round(time.perf_counter() - t, 1),
                     "final_loss": round(float(np.mean(model.loss_history[-50:])), 4)}
        print(json.dumps({name: res[name]}), flush=True)
    with open(os.path.join(a.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    keys = ["E_target@4", "vis_AUROC@4", "id_acc@4", "target_L1@4", "global_L1@4"]
    print("\n| Method | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys))
    for name in ["persistence"] + a.methods.split(","):
        print(f"| {name} | " + " | ".join(str(res[name].get(k, "–")) for k in keys) + " |")


if __name__ == "__main__":
    main()
