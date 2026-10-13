#!/usr/bin/env python3
"""P4.3 / P4.4 (simulated): views and time to a grasp-ready scene per policy.

Scenes: ``scene.make_ap_scene`` (plant rows of scout_piper_whole_body_mpc with
the target in the middle of a branch and 1-3 leaves in front of it on the
approach line). Each policy starts from the same scan pose and the same first
view and takes views until its stem joint estimate (σ ≤ 1 mm) and approach
corridor (≤ 5 % unknown) requirements are met, ``--max-views`` is reached or
no reachable view is left (the fixed ring is exhausted after its 6 points).
See ``sim.run_episode`` for the loop and the metrics.

Exit criterion of Phase 4 (ROADMAP.md): fewer views or lower acquisition time
at equal or better grasp readiness than the baselines.

    cd Codes/src/scout_piper_active_perception
    PYTHONPATH=.:../scout_piper_whole_body_mpc:../scout_piper_scene_repr/python \\
        python3 benchmarks/nbv_benchmark.py --scenes 20 --jobs 4 --out nbv.jsonl
    # closed loop through the whole-body MPC (slower):  --executor mpc --scenes 5
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scout_piper_active_perception.nbv import POLICIES  # noqa: E402
from scout_piper_active_perception.sim import run_episode  # noqa: E402


def job(args):
    seed, policy, kw = args
    t0 = time.perf_counter()
    try:
        r = run_episode(seed, policy, **kw).row()
    except Exception as exc:  # noqa: BLE001 — one failed scene must not stop the table
        r = {"seed": seed, "policy": policy, "error": repr(exc)}
    r["wall_s"] = round(time.perf_counter() - t0, 1)
    return r


def summarize(rows):
    print(f"{'policy':<11} {'n':>3} {'success':>8} {'views':>6} {'time s':>7} {'views→ok':>9} "
          f"{'axis mm':>8} {'corr unk':>9} {'clear cm':>9} {'leaf mm':>8}")
    out = {}
    for pol in POLICIES:
        rs = [r for r in rows if r.get("policy") == pol and "error" not in r]
        if not rs:
            continue
        ok = [r for r in rs if r["success"]]
        ov = [r["oracle_views"] for r in rs if r["oracle_views"] is not None]
        s = {"n": len(rs), "success": len(ok), "views_median": float(np.median([r["views"] for r in rs])),
             "time_median_s": float(np.median([r["time_s"] for r in rs])),
             "views_to_ok_median": float(np.median(ov)) if ov else None,
             "axis_mm_median": float(np.median([r["axis_error_mm"] for r in rs])),
             "corridor_unknown_median": float(np.median([r["corridor_unknown"] for r in rs])),
             "min_clearance_cm": 100 * min(r["min_hard_clearance_m"] for r in rs),
             "max_leaf_pen_mm": 1000 * max(r["max_leaf_penetration_m"] for r in rs)}
        out[pol] = s
        print(f"{pol:<11} {s['n']:>3} {s['success']:>4}/{s['n']:<3} {s['views_median']:>6.1f} {s['time_median_s']:>7.1f} "
              f"{(s['views_to_ok_median'] if s['views_to_ok_median'] is not None else float('nan')):>9.1f} "
              f"{s['axis_mm_median']:>8.2f} {s['corridor_unknown_median']:>9.3f} {s['min_clearance_cm']:>9.1f} "
              f"{s['max_leaf_pen_mm']:>8.1f}")
    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"{len(errs)} runs failed, e.g. {errs[0]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--max-views", type=int, default=8)
    ap.add_argument("--executor", choices=("teleport", "mpc"), default="teleport")
    ap.add_argument("--lambda-t", type=float, default=0.02, help="score per second of motion")
    ap.add_argument("--lambda-turn", type=float, default=0.15, help="score per radian of approach-axis turn")
    ap.add_argument("--pose-sigma-mm", type=float, default=0.5)
    ap.add_argument("--pose-sigma-deg", type=float, default=0.1)
    ap.add_argument("--no-reach-margin", action="store_true",
                    help="views at any valid arm pose (upper bound; the MPC does not reach stretched arms)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    kw = {"max_views": a.max_views, "executor": a.executor, "lambda_t": a.lambda_t, "lambda_turn": a.lambda_turn,
          "pose_sigma_m": a.pose_sigma_mm / 1000.0, "pose_sigma_deg": a.pose_sigma_deg,
          "reach_max_m": None if a.no_reach_margin else 0.36}
    jobs = [(s, p, kw) for s in range(a.first_seed, a.first_seed + a.scenes) for p in a.policies.split(",")]
    rows = []
    f = open(a.out, "w", encoding="utf-8") if a.out else None
    with Pool(a.jobs) as pool:
        for r in pool.imap_unordered(job, jobs):
            rows.append(r)
            line = json.dumps(r)
            print(line, flush=True)
            if f:
                f.write(line + "\n")
                f.flush()
    if f:
        f.close()
    summarize(rows)


if __name__ == "__main__":
    main()
