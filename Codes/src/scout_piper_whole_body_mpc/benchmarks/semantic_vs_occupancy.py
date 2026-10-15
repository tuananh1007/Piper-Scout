#!/usr/bin/env python3
"""P1.4.2 (synthetic): planning success with a semantic scene vs occupancy only.

Same planner (the unified whole-body MPPI, W3), same scenes and start state;
only the scene representation differs:

  occupancy  every surface is a hard obstacle (an Octomap-style map): stems,
             branches, pots *and leaves* (discs thickened by ``leaf_thickness_m``)
  semantic   stems, branches and pots hard; leaves soft (cost w·(d_soft − d)²,
             pushing through up to max_penetration_m allowed, as in
             semantic_classes.yaml)

Success (both): final TCP error < 2 cm, no hard collision (stems, branches,
pots) and no leaf penetration deeper than ``max_penetration_m`` on the way.
Goal of P1.4.2: ≥ 30 % fewer failures with the semantic representation.

Scenes are ``scenes.make_plant_scene`` (targets at branch tips, leaves around
them), or recorded scenes (P1.4.1): ``--episode`` takes episodes exported from
``record_bag.sh scene`` bags (scout_piper_jepa ``episode export``, with
labels); the semantic voxel map is built offline from their depth, poses and
labels (scout_piper_scene_repr ``offline.py``), and the two representations
differ only in the leaf policy (soft vs hard). The goal is the pre-grasp
point of the target class (attractor) unless ``--goal`` is given; the robot
starts at the episode's first state.

    cd Codes/src/scout_piper_whole_body_mpc
    PYTHONPATH=. python3 benchmarks/semantic_vs_occupancy.py --scenes 20 --jobs 4
    PYTHONPATH=.:../scout_piper_scene_repr/python python3 benchmarks/semantic_vs_occupancy.py \
        --episode scene01.npz scene02.npz ...
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from scout_piper_whole_body_mpc.costs.terms import CostWeights, Goal, WholeBodyCost
from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel
from scout_piper_whole_body_mpc.safety.projection import SafetyFilter
from scout_piper_whole_body_mpc.scene_adapter import disc_distance, gridded
from scout_piper_whole_body_mpc.scenes import make_plant_scene
from scout_piper_whole_body_mpc.sim import run_closed_loop
from scout_piper_whole_body_mpc.solvers.mppi import MPPI, MPPIConfig

X0 = np.r_[0.0, 0.0, 0.0, 0.0, 1.2, -1.0, 0.0, 0.5, 0.0]


def run_scene(args):
    seed, steps, samples, max_pen, thick = args
    scene = make_plant_scene(seed)
    m = WholeBodyModel()
    hard_exact = scene.distance_fn()

    def leaf_d(p):
        p = np.asarray(p, float).reshape(-1, 3)
        if not len(scene.leaf_c):
            return np.full(len(p), np.inf)
        return disc_distance(p, scene.leaf_c, scene.leaf_n, scene.leaf_r).min(1)

    def occ_exact(p):
        d, v = hard_exact(p)
        return np.minimum(d, leaf_d(p) - thick / 2), v

    lo, hi = scene.bounds()
    reps = {"semantic": (gridded(hard_exact, lo, hi), gridded(scene.leaf_fn(), lo, hi, values=True)),
            "occupancy": (gridded(occ_exact, lo, hi), None)}
    rows = []
    for name, (dist, leaf) in reps.items():
        cost = WholeBodyCost(m, Goal(p=scene.goal), distance_fn=dist, leaf_fn=leaf, w=CostWeights())
        t = time.perf_counter()
        log = run_closed_loop(X0, MPPI(m, MPPIConfig(seed=seed, samples=samples)), cost,
                              SafetyFilter(m, distance_fn=dist), steps=steps)
        X = np.array(log.X)
        C, r = m.collision_spheres(X)
        P = C.reshape(-1, 3)
        hard_clear = float((hard_exact(P)[0].reshape(C.shape[:-1]) - r).min())
        leaf_pen = float(np.clip(-(leaf_d(P).reshape(C.shape[:-1]) - r), 0, None).max())
        err = float(np.linalg.norm(m.tcp_world(X[-1])[:3, 3] - scene.goal))
        goal_in_leaf = bool(leaf_d(scene.goal[None])[0] < 0.03 + thick / 2)
        rows.append({"scene": scene.name, "representation": name, "goal_near_leaf": goal_in_leaf,
                     "final_error_m": round(err, 4), "min_hard_clearance_m": round(hard_clear, 4),
                     "max_leaf_penetration_m": round(leaf_pen, 4),
                     "success": bool(err < 0.02 and hard_clear >= 0.0 and leaf_pen <= max_pen),
                     "steps": len(log.U), "s": round(time.perf_counter() - t, 1)})
    return rows


def run_episode(args):
    """One recorded scene: map from the episode, both leaf policies, same planner."""
    path, goal_arg, steps, samples, max_pen, map_stride = args
    from scout_piper_scene_repr_py.attractor import target_goal  # noqa: PLC0415
    from scout_piper_scene_repr_py.distance_query import SemanticDistanceQuery  # noqa: PLC0415
    from scout_piper_scene_repr_py.offline import map_from_episode  # noqa: PLC0415
    from scout_piper_scene_repr_py.policy import DEFAULT_POLICIES, ClassPolicy  # noqa: PLC0415

    from scout_piper_whole_body_mpc.scene_adapter import semantic_distance_fn, semantic_leaf_fn  # noqa: PLC0415

    ep = dict(np.load(path))
    vmap = map_from_episode(ep, stride=map_stride)
    st = ep.get("states")
    x0 = X0.copy()
    if st is not None and np.isfinite(st).all(1).any():
        x0 = st[np.isfinite(st).all(1)][0].astype(float)
    if goal_arg is not None:
        goal = np.asarray(goal_arg, float)
    else:
        g = target_goal(vmap.occupied_points("target"), vmap.spec.voxel_size, x0[:2])
        if g is None:
            return [{"scene": os.path.basename(path), "skipped": "no target class in the map; pass --goal"}]
        goal = g.pre_grasp
    m = WholeBodyModel()
    occ = dict(DEFAULT_POLICIES)
    occ["leaf"] = ClassPolicy(name="leaf", label_id=3, behavior="hard", padding_m=0.0)
    sem_q = SemanticDistanceQuery(vmap, DEFAULT_POLICIES, max_age_s=1e9)
    hard_exact = semantic_distance_fn(sem_q, now=vmap.stamp)
    leaf_q = sem_q
    rows = []
    for name, pol in (("occupancy", occ), ("semantic", DEFAULT_POLICIES)):
        q = SemanticDistanceQuery(vmap, pol, max_age_s=1e9)
        dist = semantic_distance_fn(q, now=vmap.stamp)
        leaf = semantic_leaf_fn(q) if name == "semantic" else None
        cost = WholeBodyCost(m, Goal(p=goal), distance_fn=dist, leaf_fn=leaf, w=CostWeights())
        t = time.perf_counter()
        log = run_closed_loop(x0, MPPI(m, MPPIConfig(seed=0, samples=samples)), cost,
                              SafetyFilter(m, distance_fn=dist, unknown_policy="no_entry"), steps=steps)
        X = np.array(log.X)
        C, r = m.collision_spheres(X)
        P = C.reshape(-1, 3)
        d_hard, v_hard = hard_exact(P)
        known = v_hard.reshape(C.shape[:-1])
        clear = np.where(known, d_hard.reshape(C.shape[:-1]) - r, np.inf)
        leaf_d = leaf_q._sample(leaf_q.class_field("leaf"), P).reshape(C.shape[:-1]) if "leaf" in vmap.hits else None
        pen = float(np.clip(-(leaf_d - r), 0, None).max()) if leaf_d is not None else 0.0
        err = float(np.linalg.norm(m.tcp_world(X[-1])[:3, 3] - goal))
        rows.append({"scene": os.path.basename(path), "representation": name, "goal_near_leaf": False,
                     "goal": [round(float(v), 3) for v in goal], "final_error_m": round(err, 4),
                     "min_hard_clearance_m": round(float(clear.min()), 4), "max_leaf_penetration_m": round(pen, 4),
                     "success": bool(err < 0.02 and clear.min() >= 0.0 and pen <= max_pen),
                     "steps": len(log.U), "s": round(time.perf_counter() - t, 1)})
    return rows


def summary(rows) -> str:
    out = ["| Representation | Success | Failures | Goal near a leaf: success |", "|---|---|---|---|"]
    fails = {}
    for rep in ("occupancy", "semantic"):
        rs = [r for r in rows if r["representation"] == rep]
        if not rs:
            continue
        ok = sum(r["success"] for r in rs)
        near = [r for r in rs if r["goal_near_leaf"]]
        fails[rep] = len(rs) - ok
        out.append(f"| {rep} | {ok}/{len(rs)} | {len(rs) - ok} | "
                   f"{sum(r['success'] for r in near)}/{len(near)} |")
    if "occupancy" in fails and "semantic" in fails and fails["occupancy"]:
        red = 1 - fails["semantic"] / fails["occupancy"]
        out.append(f"\nP1.4.2 goal (≥ 30 % fewer failures): {100 * red:.0f} % fewer → "
                   f"{'PASS' if red >= 0.3 else 'FAIL'}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--max-penetration", type=float, default=0.02, help="m, semantic_classes.yaml leaf")
    ap.add_argument("--leaf-thickness", type=float, default=0.01, help="m, occupancy inflation of leaves")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--summary", default="", help="only print the table of an existing JSONL file")
    ap.add_argument("--episode", nargs="*", default=[], help="recorded scene episodes instead of synthetic")
    ap.add_argument("--goal", type=float, nargs=3, default=None, help="world goal for --episode (one scene)")
    ap.add_argument("--map-stride", type=int, default=2, help="--episode: depth pixel stride for the map")
    a = ap.parse_args()
    if a.summary:
        with open(a.summary, encoding="utf-8") as f:
            print(summary([json.loads(line) for line in f if line.strip()]))
        return
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if a.episode:
        fn, jobs = run_episode, [(p, a.goal, a.steps, a.samples, a.max_penetration, a.map_stride)
                                 for p in a.episode]
    else:
        fn = run_scene
        jobs = [(s, a.steps, a.samples, a.max_penetration, a.leaf_thickness)
                for s in range(a.first, a.first + a.scenes)]
    rows = []
    out = open(a.out, "a", encoding="utf-8") if a.out else None
    pool = Pool(a.jobs) if a.jobs > 1 else None
    it = pool.imap_unordered(fn, jobs) if pool else map(fn, jobs)
    for scene_rows in it:
        for row in scene_rows:
            if "skipped" in row:
                print(json.dumps(row), flush=True)
                continue
            rows.append(row)
            print(json.dumps(row), flush=True)
            if out:
                out.write(json.dumps(row) + "\n"); out.flush()
    if pool:
        pool.close()
    print(summary(rows))


if __name__ == "__main__":
    main()
