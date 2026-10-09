"""P3B.10: latency of the learned predictor and of the predictive MPC cost.

    python3 -m scout_piper_jepa.latency --device cuda --samples 64,128,256 --horizon 4
    python3 -m scout_piper_jepa.latency --model runs/e3/P3.pt --device cuda --fp16

Times (median / p95 over ``--repeats`` after warm-up):
  rollout  TorchACPredictor.rollout for samples × horizon predictor steps
  cost     JepaVisibilityCost on MPC rollouts (Γ from states, prediction,
           read-out), what one MPPI iteration adds per control step

Without ``--model`` an untrained network of the default size is used (same
cost as a trained one). The control budget is 100 ms per step for everything;
the predictive cost runs once per MPPI iteration (2 by default).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def _time(fn, repeats: int, warmup: int, sync) -> np.ndarray:
    out = []
    for i in range(repeats + warmup):
        t = time.perf_counter()
        fn()
        sync()
        if i >= warmup:
            out.append(1e3 * (time.perf_counter() - t))
    return np.array(out)


def main(argv=None) -> None:
    import torch  # noqa: PLC0415

    from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel  # noqa: PLC0415

    from .predictive_cost import JepaVisibilityCost  # noqa: PLC0415
    from .predictor import StateConditionedPredictor  # noqa: PLC0415
    from .torch_predictor import ACPredictorConfig, TorchACPredictor  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true", help="autocast to half precision on CUDA")
    ap.add_argument("--samples", default="32,64,128,256")
    ap.add_argument("--horizon", type=int, default=4, help="predictor steps")
    ap.add_argument("--stride", type=int, default=4, help="MPC steps per predictor step")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    model = TorchACPredictor.load(a.model, device=a.device) if a.model else \
        TorchACPredictor(ACPredictorConfig(device=a.device))
    cfg = model.cfg
    hf, wf = cfg.grid_hw
    sync = torch.cuda.synchronize if a.device.startswith("cuda") else (lambda: None)
    m = WholeBodyModel()
    rng = np.random.default_rng(0)
    Zh = rng.normal(size=(cfg.history, hf, wf, cfg.feat_dim)).astype(np.float32)
    Zh /= np.linalg.norm(Zh, axis=-1, keepdims=True)
    r = Zh[-1, hf // 2, wf // 2]
    rows = []
    ctx = torch.autocast("cuda", dtype=torch.float16) if a.fp16 and a.device.startswith("cuda") else None
    for S in [int(v) for v in a.samples.split(",")]:
        A = rng.normal(0, 0.05, (S, a.horizon, 9)).astype(np.float32)
        Zb = np.broadcast_to(Zh[None], (S,) + Zh.shape)

        def roll():
            if ctx:
                with ctx:
                    model.rollout(Zb, A)
            else:
                model.rollout(Zb, A)
        t_roll = _time(roll, a.repeats, a.warmup, sync)
        H_mpc = a.horizon * a.stride
        X0 = np.r_[0.0, 0.0, 0.0, 0.0, 1.2, -1.0, 0.0, 0.5, 0.0]
        U = np.clip(rng.normal(0, 0.2, (S, H_mpc, 8)), m.u_low, m.u_high)
        X = m.rollout(X0, U)
        cost = JepaVisibilityCost(StateConditionedPredictor(model, m.kin.tcp, a.stride),
                                  image_hw=(hf * 8, wf * 8), stride=a.stride)
        cost.set_context(Zh, r, np.array([wf * 4.0, hf * 4.0]))
        t_cost = _time(lambda: cost(X, U), a.repeats, a.warmup, sync)
        row = {"device": a.device, "fp16": bool(ctx), "samples": S, "horizon": a.horizon,
               "grid": [hf, wf], "feat_dim": cfg.feat_dim,
               "rollout_median_ms": round(float(np.median(t_roll)), 1),
               "rollout_p95_ms": round(float(np.percentile(t_roll, 95)), 1),
               "cost_median_ms": round(float(np.median(t_cost)), 1),
               "cost_p95_ms": round(float(np.percentile(t_cost, 95)), 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    print("\n| Samples | Rollout median / p95 ms | Cost median / p95 ms |\n|---|---|---|")
    for r_ in rows:
        print(f"| {r_['samples']} | {r_['rollout_median_ms']} / {r_['rollout_p95_ms']} | "
              f"{r_['cost_median_ms']} / {r_['cost_p95_ms']} |")
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            for r_ in rows:
                f.write(json.dumps(r_) + "\n")


if __name__ == "__main__":
    main()
