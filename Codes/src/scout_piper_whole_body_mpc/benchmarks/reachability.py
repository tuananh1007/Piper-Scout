#!/usr/bin/env python3
"""WE2/WE3 offline screening: arm-only (W0, frozen base) vs unified MPC on
R1/R3 synthetic goals, no obstacles. Prints one JSON line per run.

    python3 benchmarks/reachability.py --seeds 3
"""

import argparse
import json
import time

import numpy as np

from scout_piper_whole_body_mpc.costs.terms import Goal, WholeBodyCost
from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel
from scout_piper_whole_body_mpc.safety.projection import SafetyFilter
from scout_piper_whole_body_mpc.sim import run_closed_loop
from scout_piper_whole_body_mpc.solvers.mppi import MPPI, MPPIConfig

X0 = np.r_[0.0, 0.0, 0.0, 0.0, 1.2, -1.0, 0.0, 0.5, 0.0]
SCENES = {"R1": [0.58, 0.10, 0.40], "R2": [0.80, -0.20, 0.35], "R3": [1.50, 0.30, 0.40]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args()
    for scene, goal in SCENES.items():
        for method, freeze in (("W0_arm_only", True), ("W3_unified", False)):
            for seed in range(a.seeds):
                m = WholeBodyModel()
                cost = WholeBodyCost(m, Goal(p=np.array(goal)))
                t = time.perf_counter()
                log = run_closed_loop(X0, MPPI(m, MPPIConfig(seed=seed, arm_only=freeze)), cost, SafetyFilter(m),
                                      steps=a.steps, freeze_base=freeze)
                err = float(np.linalg.norm(m.tcp_world(log.X[-1])[:3, 3] - goal))
                print(json.dumps({"scene": scene, "method": method, "seed": seed,
                                  "final_error_m": round(err, 4), "success": err < 0.02,
                                  "steps": len(log.U), "base_distance_m": round(log.base_distance, 3),
                                  "ms_per_step": round(1e3 * (time.perf_counter() - t) / len(log.U), 1)}))


if __name__ == "__main__":
    main()
