#!/usr/bin/env python3
"""WE2/WE3 offline screening on synthetic goals: arm-only (W0, frozen base),
sequential base-then-arm (W1) and unified MPC (W3). R1–R3 have no obstacles;
O1 puts a 3 cm sphere on the straight TCP path (the P3A.6 case). Prints one
JSON line per run.

    PYTHONPATH=. python3 benchmarks/reachability.py --seeds 3
"""

import argparse
import json
import time

import numpy as np

from scout_piper_whole_body_mpc.baselines.sequential import run_sequential
from scout_piper_whole_body_mpc.costs.terms import Goal, WholeBodyCost
from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel
from scout_piper_whole_body_mpc.safety.projection import SafetyFilter
from scout_piper_whole_body_mpc.scene_adapter import spheres_distance_fn
from scout_piper_whole_body_mpc.sim import run_closed_loop
from scout_piper_whole_body_mpc.solvers.mppi import MPPI, MPPIConfig

X0 = np.r_[0.0, 0.0, 0.0, 0.0, 1.2, -1.0, 0.0, 0.5, 0.0]
SCENES = {                                   # goal, obstacles (centre, radius)
    "R1": ([0.58, 0.10, 0.40], []),
    "R2": ([0.80, -0.20, 0.35], []),
    "R3": ([1.50, 0.30, 0.40], []),
    "O1": ([0.62, -0.35, 0.40], [([0.58, -0.145, 0.434], 0.03)]),
}
METHODS = ("W0_arm_only", "W1_sequential", "W3_unified")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args()
    for scene, (goal, obstacles) in SCENES.items():
        for method in METHODS:
            for seed in range(a.seeds):
                m = WholeBodyModel()
                dist = spheres_distance_fn(*zip(*obstacles)) if obstacles else None
                cost = WholeBodyCost(m, Goal(p=np.array(goal)), distance_fn=dist)
                safety = SafetyFilter(m, distance_fn=dist)
                t = time.perf_counter()
                if method == "W1_sequential":
                    log = run_sequential(X0, cost, safety, steps=a.steps, seed=seed).log
                else:
                    freeze = method == "W0_arm_only"
                    log = run_closed_loop(X0, MPPI(m, MPPIConfig(seed=seed, arm_only=freeze)), cost, safety,
                                          steps=a.steps, freeze_base=freeze)
                err = float(np.linalg.norm(m.tcp_world(log.X[-1])[:3, 3] - goal))
                row = {"scene": scene, "method": method, "seed": seed,
                       "final_error_m": round(err, 4), "success": err < 0.02,
                       "steps": len(log.U), "base_distance_m": round(log.base_distance, 3),
                       "ms_per_step": round(1e3 * (time.perf_counter() - t) / max(len(log.U), 1), 1)}
                if dist is not None:
                    row["min_clearance_m"] = round(min(
                        float((dist(m.collision_spheres(x[None])[0][0])[0] - m.collision_spheres(x[None])[1]).min())
                        for x in log.X), 4)
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
