#!/usr/bin/env python3
"""E3 on the synthetic world: persistence vs P0 / P2 / P3 (needs torch).

Starts are whole-body states whose eye-in-hand camera sees the flower; each
episode is smooth random base + arm motion rendered every ``stride`` control
steps (0.1 s each). Scores per horizon follow PIPER_JEPA_EXPERIMENTS.md §10.

    cd Codes/src
    PYTHONPATH=scout_piper_jepa:scout_piper_whole_body_mpc \\
        python3 scout_piper_jepa/benchmarks/e3_synthetic.py --steps 1500 --out e3.json

Synthetic only: it checks the pipeline and the direction of the P2 vs P3
effect; the E3 claims need recorded robot episodes and V-JEPA features.
"""

import argparse
import json
import time

import numpy as np

from scout_piper_jepa.prediction_metrics import score_predictions
from scout_piper_jepa.predictor import PersistencePredictor
from scout_piper_jepa.synthetic import (TARGET, eye_in_hand_pose, make_flower_scene,
                                        synthetic_episodes, target_truth)
from scout_piper_jepa.torch_predictor import ACPredictorConfig, train_predictor
from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel


def visible_starts(world, model, cam, rng, n_seed=20000, n_max=4000):
    cand = np.concatenate([rng.normal(0, [0.05, 0.05, 0.1], (n_seed, 3)),
                           rng.uniform(model.kin.lower, model.kin.upper, (n_seed, 6))], 1)
    _, vis = target_truth(world.render(cam(cand))["label"], world.camera)
    starts = cand[vis]
    for _ in range(3):                      # grow by perturbing poses that see the flower
        pert = np.repeat(starts, 6, 0) + rng.normal(0, [0.03, 0.03, 0.05] + [0.12] * 6, (len(starts) * 6, 9))
        pert[:, 3:] = np.clip(pert[:, 3:], model.kin.lower, model.kin.upper)
        _, v2 = target_truth(world.render(cam(pert))["label"], world.camera)
        starts = np.concatenate([starts, pert[v2]])[:n_max]
    rng.shuffle(starts)
    return starts


def evaluate(pred, episodes, r, image_hw, K, H, every=3):
    Zp, Zt, Lt, Ut, Vt, U0 = [], [], [], [], [], []
    for ep in episodes:
        for t0 in range(K - 1, len(ep["Z"]) - H, every):
            if not ep["visible"][t0]:
                continue
            Zp.append(pred.rollout(ep["Z"][None, t0 - K + 1:t0 + 1], ep["A"][None, t0:t0 + H])[0])
            sl = slice(t0 + 1, t0 + H + 1)
            Zt.append(ep["Z"][sl]); Lt.append(ep["label"][sl]); Ut.append(ep["u"][sl])
            Vt.append(ep["visible"][sl]); U0.append(ep["u"][t0])
    s = score_predictions(np.array(Zp), np.array(Zt), np.array(Lt), np.array(Ut), np.array(Vt), r,
                          np.array(U0), image_hw)
    return {**s.summary(), "n_windows": len(Zp)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--frames", type=int, default=14)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    m = WholeBodyModel()
    world = make_flower_scene(seed=a.seed)
    cam = eye_in_hand_pose(m)
    starts = visible_starts(world, m, cam, rng)
    kw = dict(frames=a.frames, stride=a.stride, fk=m.kin.tcp, rng=rng, scale=0.3)
    train = synthetic_episodes(world, m, cam, starts[:a.train], **kw)
    test = synthetic_episodes(world, m, cam, starts[a.train:a.train + a.test], **kw)
    r = world.features[world.labels == TARGET][0]
    K, H = 2, 8
    res = {"setup": {"train_episodes": len(train), "test_episodes": len(test), "frames": a.frames,
                     "predictor_step_s": 0.1 * a.stride, "train_steps": a.steps,
                     "target_visible_fraction": round(float(np.mean([e["visible"].mean() for e in train])), 3)}}
    res["persistence"] = evaluate(PersistencePredictor(), test, r, world.camera.image_hw, K, H)
    print(json.dumps({"persistence": res["persistence"]}), flush=True)
    for name, cfg in (("P0", ACPredictorConfig.p0(steps=a.steps, history=K, seed=a.seed)),
                      ("P2", ACPredictorConfig.p2(steps=a.steps, history=K, seed=a.seed)),
                      ("P3", ACPredictorConfig.p3(steps=a.steps, history=K, seed=a.seed))):
        t = time.perf_counter()
        model = train_predictor(train, cfg)
        res[name] = {**evaluate(model, test, r, world.camera.image_hw, K, H),
                     "train_s": round(time.perf_counter() - t)}
        print(json.dumps({name: res[name]}), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
