#!/usr/bin/env python3
"""P3A.8 / WE7: MPPI solve time per control step, numpy vs torch (CPU / CUDA).

Times ``solve`` (sampling iterations, elitism and refinement, as configured)
from a fixed state toward a goal 0.6 m out, with no geometry and with the
voxel field of a synthetic plant scene. Warm-up solves are discarded (CUDA
kernel compilation, allocator). The budget is the 100 ms control period;
the node warns when solves take over 90 ms.

    cd Codes/src/scout_piper_whole_body_mpc
    PYTHONPATH=. python3 benchmarks/timing.py                                  # numpy only
    PYTHONPATH=. python3 benchmarks/timing.py --backends numpy,torch-cpu,torch-cuda \\
        --samples 128,256,512,1024,2048 --out timing.jsonl

Prints one JSON line per configuration and a table (median / p95 ms).
"""

import argparse
import json
import time

import numpy as np

from scout_piper_whole_body_mpc.costs.terms import CostWeights, Goal, WholeBodyCost
from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel
from scout_piper_whole_body_mpc.scenes import make_plant_scene
from scout_piper_whole_body_mpc.solvers.mppi import MPPI, MPPIConfig

X0 = np.r_[0.0, 0.0, 0.0, 0.0, 1.2, -1.0, 0.0, 0.5, 0.0]


def make_solver(backend: str, m, cfg, grid):
    if backend == "numpy":
        return MPPI(m, cfg)
    from scout_piper_whole_body_mpc.torch_backend import TorchGridField, TorchMPPI  # noqa: PLC0415
    device = backend.split("-", 1)[1]
    field = None
    if grid is not None:
        field = TorchGridField.from_grid_fn(grid[0], grid[1], device=device)
    return TorchMPPI(m, cfg, device=device, field=field)


def time_solves(solver, cost, n: int, warmup: int, sync=None):
    x, u_prev = X0.copy(), np.zeros(8)
    out = []
    for i in range(n + warmup):
        t = time.perf_counter()
        u = solver.solve(x, cost, u_prev)
        if sync:
            sync()
        if i >= warmup:
            out.append(1e3 * (time.perf_counter() - t))
        solver.shift()
        u_prev = u
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", default="numpy")
    ap.add_argument("--samples", default="128,256,512")
    ap.add_argument("--refine-iters", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--solves", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--geometry", default="none,plant", help="none and/or plant (scene P03 voxel field)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    m = WholeBodyModel()
    goal = Goal(p=np.array([0.62, -0.17, 0.35]))
    w = CostWeights(reach=1e4)
    rows = []
    grids = {}
    for geom in a.geometry.split(","):
        if geom == "plant":
            grids[geom] = make_plant_scene(3).grid_fields()
        else:
            grids[geom] = None
    for backend in a.backends.split(","):
        sync = None
        if backend == "torch-cuda":
            import torch  # noqa: PLC0415
            if not torch.cuda.is_available():
                print(json.dumps({"backend": backend, "skipped": "CUDA not available"}))
                continue
            sync = torch.cuda.synchronize
        for geom, grid in grids.items():
            cost = WholeBodyCost(m, goal, w=w, distance_fn=None if grid is None else grid[0],
                                 leaf_fn=None if grid is None else grid[1])
            for s in [int(v) for v in a.samples.split(",")]:
                cfg = MPPIConfig(samples=s, iterations=a.iterations, refine_iters=a.refine_iters, seed=0)
                solver = make_solver(backend, m, cfg, grid)
                ms = time_solves(solver, cost, a.solves, a.warmup, sync)
                row = {"backend": backend, "geometry": geom, "samples": s, "iterations": a.iterations,
                       "refine_iters": a.refine_iters, "median_ms": round(float(np.median(ms)), 1),
                       "p95_ms": round(float(np.percentile(ms, 95)), 1),
                       "within_90ms": bool(np.percentile(ms, 95) < 90.0)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print("\n| Backend | Geometry | Samples | Median ms | p95 ms | p95 < 90 ms |\n|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['backend']} | {r['geometry']} | {r['samples']} | {r['median_ms']} | {r['p95_ms']} | "
              f"{'yes' if r['within_90ms'] else 'no'} |")


if __name__ == "__main__":
    main()
