#!/usr/bin/env python3
"""P3.3 / WE3: whole-body reachability in synthetic cluttered-plant scenes.

Methods on the same scenes, start state and safety filter:
  W0  arm-only MPPI (base frozen)
  W1  sequential: base pose from IK, base phase, then arm-only MPPI
  W2  holistic reactive QP (one-step, lazy base)
  W3  unified whole-body MPPI (this package's controller)

The planners and the safety filter use the scene on a 1 cm voxel grid
(trilinear, ≤ 3.4 mm from the exact geometry); reported clearances use the
exact capsules. A scene's target is "arm-unreachable" when position IK from the start base
pose (16 random arm seeds) finds no collision-free solution within 5 mm.
Success: final TCP error < 2 cm with no hard collision on the way. The
exit gate of P3.3.2 is that W3 reaches ≥ 50 % of the arm-unreachable targets.

    cd Codes/src/scout_piper_whole_body_mpc
    PYTHONPATH=. python3 benchmarks/plant_scenes.py --scenes 30 --jobs 4 --out plant_scenes.jsonl
    PYTHONPATH=. python3 benchmarks/plant_scenes.py --summary plant_scenes.jsonl     # table only

One JSON line per (scene, method), then a summary table.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from scout_piper_whole_body_mpc.baselines.reactive_qp import ReactiveQPConfig, run_reactive_qp
from scout_piper_whole_body_mpc.baselines.sequential import SequentialConfig, run_sequential, solve_position_ik
from scout_piper_whole_body_mpc.costs.terms import CostWeights, Goal, WholeBodyCost
from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel
from scout_piper_whole_body_mpc.safety.projection import SafetyFilter
from scout_piper_whole_body_mpc.scenes import make_plant_scene
from scout_piper_whole_body_mpc.sim import run_closed_loop
from scout_piper_whole_body_mpc.solvers.mppi import MPPI, MPPIConfig

X0 = np.r_[0.0, 0.0, 0.0, 0.0, 1.2, -1.0, 0.0, 0.5, 0.0]
METHODS = ("W0", "W1", "W2", "W3")


def arm_reachable(model, scene, x0, seeds: int = 16, tol_m: float = 0.005, clearance_m: float = 0.03) -> bool:
    rng = np.random.default_rng(0)
    cfg = SequentialConfig()
    dist = scene.distance_fn()
    for i in range(seeds):
        q0 = x0[3:] if i == 0 else rng.uniform(model.kin.lower, model.kin.upper)
        q, err = solve_position_ik(model, x0[None, :3], scene.goal, q0, cfg)
        if err[0] > tol_m:
            continue
        C, r = model.collision_spheres(np.r_[x0[:3], q[0]][None])
        d, _ = dist(C[0])
        if (d - r).min() > clearance_m:
            return True
    return False


def run_scene(args):
    """All methods on one scene (the voxel fields are built once per scene)."""
    seed, methods, steps, reach_margin, samples = args
    scene = make_plant_scene(seed)
    m = WholeBodyModel()
    exact = scene.distance_fn()                 # reported clearances, QP linearisation
    leaf_exact = scene.leaf_fn()
    dist, leaf = scene.grid_fields()            # MPPI and the safety filter: 1 cm voxel field
    reachable = arm_reachable(m, scene, X0)
    rows = []
    for method in methods:
        w = CostWeights(reach=1e4 if reach_margin and method != "W0" else 0.0)
        cost = WholeBodyCost(m, Goal(p=scene.goal), distance_fn=exact if method == "W2" else dist,
                             leaf_fn=leaf, w=w)
        safety = SafetyFilter(m, distance_fn=dist)
        t = time.perf_counter()
        mcfg = MPPIConfig(seed=seed, samples=samples, arm_only=method == "W0")
        if method == "W0":
            log = run_closed_loop(X0, MPPI(m, mcfg), cost, safety, steps=steps, freeze_base=True)
        elif method == "W1":
            log = run_sequential(X0, cost, safety, steps=steps, seed=seed,
                                 mppi_cfg=MPPIConfig(seed=seed, samples=samples)).log
        elif method == "W2":
            log = run_reactive_qp(X0, cost, safety, steps=steps,
                                  cfg=ReactiveQPConfig(reach_max_m=0.36 if reach_margin else None))
        else:
            log = run_closed_loop(X0, MPPI(m, mcfg), cost, safety, steps=steps)
        secs = time.perf_counter() - t
        X = np.array(log.X)
        C, r = m.collision_spheres(X)
        d, _ = exact(C.reshape(-1, 3))
        clear = (d.reshape(C.shape[:-1]) - r).min()
        leaf_contact = int((leaf_exact(C.reshape(-1, 3)).reshape(C.shape[:-1]) > 0).any(1).sum())
        err = float(np.linalg.norm(m.tcp_world(X[-1])[:3, 3] - scene.goal))
        rows.append({"scene": scene.name, "method": method, "arm_reachable": reachable,
                     "goal": [round(float(v), 3) for v in scene.goal],
                     "final_error_m": round(err, 4), "min_clearance_m": round(float(clear), 4),
                     "success": bool(err < 0.02 and clear >= 0.0), "steps": len(log.U),
                     "base_distance_m": round(log.base_distance, 3), "leaf_contact_steps": leaf_contact,
                     "ms_per_step": round(1e3 * secs / max(len(log.U), 1), 1)})
    return rows


def summary(rows) -> str:
    out = ["| Method | All scenes | Arm-reachable | Arm-unreachable | Median steps (success) | Median base travel |",
           "|---|---|---|---|---|---|"]
    for meth in METHODS:
        rs = [r for r in rows if r["method"] == meth]
        if not rs:
            continue
        def frac(sel):
            sel = list(sel)
            return f"{sum(r['success'] for r in sel)}/{len(sel)}" if sel else "–"
        ok = [r for r in rs if r["success"]]
        steps = f"{int(np.median([r['steps'] for r in ok]))}" if ok else "–"
        travel = f"{np.median([r['base_distance_m'] for r in ok]):.2f} m" if ok else "–"
        out.append(f"| {meth} | {frac(rs)} | {frac(r for r in rs if r['arm_reachable'])} | "
                   f"{frac(r for r in rs if not r['arm_reachable'])} | {steps} | {travel} |")
    unreach = [r for r in rows if r["method"] == "W3" and not r["arm_reachable"]]
    if unreach:
        rate = sum(r["success"] for r in unreach) / len(unreach)
        out.append(f"\nP3.3.2 exit gate (W3 reaches ≥ 50 % of arm-unreachable targets): "
                   f"{100 * rate:.0f} % → {'PASS' if rate >= 0.5 else 'FAIL'}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--first", type=int, default=0, help="first scene seed")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--samples", type=int, default=256, help="MPPI samples (W0, W1 arm phase, W3)")
    ap.add_argument("--reach-margin", action="store_true",
                    help="deployed reach margin (0.36 m) for W1-W3 instead of pure reachability")
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes")
    ap.add_argument("--out", default="", help="also append the JSON lines to this file")
    ap.add_argument("--summary", default="", help="only print the table of an existing JSONL file")
    a = ap.parse_args()
    if a.summary:
        with open(a.summary, encoding="utf-8") as f:
            print(summary([json.loads(line) for line in f if line.strip()]))
        return
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    jobs = [(s, a.methods.split(","), a.steps, a.reach_margin, a.samples)
            for s in range(a.first, a.first + a.scenes)]
    rows = []
    out = open(a.out, "a", encoding="utf-8") if a.out else None
    with Pool(a.jobs) if a.jobs > 1 else _Serial() as pool:
        for scene_rows in pool.imap_unordered(run_scene, jobs):
            for row in scene_rows:
                rows.append(row)
                line = json.dumps(row)
                print(line, flush=True)
                if out:
                    out.write(line + "\n"); out.flush()
    print(summary(rows))
    sys.stdout.flush()


class _Serial:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @staticmethod
    def imap_unordered(fn, it):
        return map(fn, it)


if __name__ == "__main__":
    main()
