#!/usr/bin/env python3
"""P2.2.2: cycle time of the Phase 2B MPPI visual servo (budget 10 ms on the Orin).

One ``MppiVisualServo.step`` per row: cost of all samples over the horizon,
the gradient refinement and the safety filter, on numpy, torch-CPU and (when
available) torch-CUDA.

    cd Codes/src/scout_piper_whole_body_mpc
    PYTHONPATH=. python3 benchmarks/visual_servo_timing.py --samples 256,512 --repeats 30
"""

import argparse
import json
import time

import numpy as np

from scout_piper_whole_body_mpc.visual_servo import (MppiVisualServo, ServoTarget, VisualServoConfig,
                                                      camera_geometry)

K = np.array([[380.0, 0.0, 320.0], [0.0, 380.0, 240.0], [0.0, 0.0, 1.0]])
Q0 = np.array([0.0, 1.2, -1.0, 0.0, 0.5, 0.0])


def flange_cam():
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = [0.05, 0.0, 0.05]
    return T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="256,512")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--backends", default="numpy,torch-cpu,torch-cuda")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = []
    for backend in a.backends.split(","):
        if backend.startswith("torch"):
            try:
                import torch  # noqa: PLC0415
            except ImportError:
                continue
            if backend == "torch-cuda" and not torch.cuda.is_available():
                continue
        for S in [int(v) for v in a.samples.split(",")]:
            cfg = VisualServoConfig(samples=S, horizon=a.horizon, refine_iters=a.refine, seed=0,
                                    backend="numpy" if backend == "numpy" else "torch",
                                    device=backend.split("-")[-1] if backend != "numpy" else "auto")
            servo = MppiVisualServo(cfg)
            F = servo.model.kin.link_frames(Q0)
            p = F[-1, :3, 3] + 0.12 * F[-1, :3, 2] + 0.02 * F[-1, :3, 0]
            q, ts = Q0.copy(), []
            for k in range(a.repeats + 3):
                uv = camera_geometry(servo.model, np.r_[0, 0, 0, q], flange_cam(), K, p)[0]
                tg = ServoTarget(uv_meas=uv, K=K, image_hw=(480, 640), T_flange_cam=flange_cam(), p_base=p)
                t = time.perf_counter()
                qd, _ = servo.step(q, tg)
                if k >= 3:
                    ts.append(1e3 * (time.perf_counter() - t))
                q = q + cfg.dt * qd
            row = {"backend": backend, "samples": S, "horizon": a.horizon, "refine": a.refine,
                   "median_ms": round(float(np.median(ts)), 1), "p95_ms": round(float(np.percentile(ts, 95)), 1)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    print("\n| Backend | Samples | Median ms | p95 ms |\n|---|---|---|---|")
    for r in rows:
        print(f"| {r['backend']} | {r['samples']} | {r['median_ms']} | {r['p95_ms']} |")
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
