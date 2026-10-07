#!/usr/bin/env python3
"""Stage C on the synthetic world: geometry-only MPC (C2) vs JEPA-aware MPC (C3).

The whole-body MPC (scout_piper_whole_body_mpc) drives the TCP to a pre-grasp
point; C3 adds ``JepaVisibilityCost`` through ``WholeBodyCost.extra``. The
predictor is either the oracle (renders the true future: the upper bound on
what prediction can buy) or a trained checkpoint (``--model``, from
torch_predictor; needs torch).

The default goal sits between the flower and its identical twin: a camera
looking straight at it sees the twin while the leaf hides the flower, so
geometry alone can end with the wrong flower in view. Reported per run:
  visible_frac   fraction of control steps with the flower truly in view
  centre_px      mean distance of the flower from the image centre when visible
  final_visible  flower in view at the end
  memory_on      the Stage A memory, run on the executed views, ends on the
                 flower ("target"), on the twin ("twin") or reports it lost
  err_cm         final TCP error

    cd Codes/src
    PYTHONPATH=scout_piper_jepa:scout_piper_whole_body_mpc \\
        python3 scout_piper_jepa/benchmarks/visibility_mpc.py --seeds 3
"""

import argparse
import json
import time

import numpy as np

from scout_piper_jepa.predictive_cost import JepaVisibilityCost
from scout_piper_jepa.predictor import OracleStatePredictor, StateConditionedPredictor
from scout_piper_jepa.synthetic import (DISTRACTOR, TARGET, eye_in_hand_pose, make_flower_scene,
                                        target_truth)
from scout_piper_jepa.target_memory import TargetMemory, TargetMemoryConfig
from scout_piper_whole_body_mpc.costs.terms import Goal, WholeBodyCost
from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel
from scout_piper_whole_body_mpc.safety.projection import SafetyFilter
from scout_piper_whole_body_mpc.solvers.mppi import MPPI, MPPIConfig


def start_state(world, model, cam, rng, n=40000, min_dist=0.25):
    """Base at the origin, arm pose whose TCP is at least ``min_dist`` from the
    flower and whose camera sees it near the image centre."""
    cand = np.concatenate([np.zeros((n, 3)), rng.uniform(model.kin.lower, model.kin.upper, (n, 6))], 1)
    far = np.linalg.norm(model.tcp_world(cand)[:, :3, 3] - world.target_position(), axis=1) > min_dist
    cand = cand[far]
    lab = world.render(cam(cand))["label"]
    u, vis = target_truth(lab, world.camera)
    nt = (lab == TARGET).reshape(len(cand), -1).sum(1)
    H, W = world.camera.image_hw
    score = np.where(vis, np.linalg.norm(u - [W / 2, H / 2], axis=1) - 3 * nt, np.inf)
    return cand[int(np.argmin(score))]


def upsample(mask_grid, image_hw):
    hf, wf = mask_grid.shape
    H, W = image_hw
    return np.kron(mask_grid, np.ones((H // hf, W // wf), bool))


def upsample_depth(depth_grid, image_hw):
    hf, wf = depth_grid.shape
    H, W = image_hw
    return np.kron(np.where(np.isfinite(depth_grid), depth_grid, 0.0), np.ones((H // hf, W // wf)))


def run(method, world, model, cam, x0, goal, seed, steps, predictor=None, samples=64, anchor=True,
        anchor_tol_m=0.03):
    r_t = world.features[world.labels == TARGET][0]
    H, W = world.camera.image_hw
    cost = WholeBodyCost(model, Goal(p=goal))
    vc = None
    if method != "C2":
        vc = JepaVisibilityCost(predictor, world.camera.image_hw, camera_pose=cam, K=world.camera.K,
                                stride=predictor.stride)
        cost.extra.append(vc)
    mppi = MPPI(model, MPPIConfig(samples=samples, horizon=16, refine_iters=1, seed=seed))
    sf = SafetyFilter(model)
    rng = np.random.default_rng(seed)
    x, u_prev = np.asarray(x0, float).copy(), np.zeros(8)
    r0 = world.render(cam(x), rng=rng)
    mem = TargetMemory(TargetMemoryConfig())
    mem.initialize(r0["Z"], upsample(r0["label"] == TARGET, (H, W)))
    u_now = target_truth(r0["label"], world.camera)[0]
    hist = [r0["Z"]]
    # 3-D anchor from the grounding frame; later depth estimates refresh it only
    # when they agree (a tracker that slipped to the twin must not drag it along)
    p_world = mem.update(r0["Z"], image_hw=(H, W), depth=upsample_depth(r0["depth"], (H, W)),
                         K=world.camera.K, T_world_cam=cam(x)).p_world
    vis, centre = [], []
    t = time.perf_counter()
    for k in range(steps):
        rr = world.render(cam(x), rng=rng)
        un, vn = target_truth(rr["label"], world.camera)
        vis.append(bool(vn))
        if vn:
            centre.append(float(np.linalg.norm(un - [W / 2, H / 2])))
        st = mem.update(rr["Z"], image_hw=(H, W), depth=upsample_depth(rr["depth"], (H, W)),
                        K=world.camera.K, T_world_cam=cam(x))
        if st.visible:
            u_now = st.u_mean
            if st.p_world is not None and (p_world is None or np.linalg.norm(st.p_world - p_world) < anchor_tol_m):
                p_world = st.p_world
        hist = (hist + [rr["Z"]])[-2:]
        if vc is not None:
            vc.set_context(np.stack(hist), r_t, u_now, p_world=p_world if anchor else None)
        uc = mppi.solve(x, cost, u_prev)
        rep = sf.project(x, uc, u_prev)
        x = model.rollout(x, rep.u[None, None])[0, 1]
        u_prev = rep.u
        mppi.shift()
        if np.linalg.norm(model.tcp_world(x)[:3, 3] - goal) < 0.01 and np.abs(rep.u).max() < 0.05:
            break
    rr = world.render(cam(x), rng=rng)
    st = mem.update(rr["Z"], image_hw=(H, W))
    lab_up = np.kron(rr["label"], np.ones((H // rr["label"].shape[0], W // rr["label"].shape[1]), int))
    on = "lost"
    if st.status != "lost":
        ui, vi = int(np.clip(st.u_mean[0], 0, W - 1)), int(np.clip(st.u_mean[1], 0, H - 1))
        on = {TARGET: "target", DISTRACTOR: "twin"}.get(int(lab_up[vi, ui]), "elsewhere")
    _, vfin = target_truth(rr["label"], world.camera)
    return {"method": method, "seed": seed, "steps": k + 1,
            "err_cm": round(100 * float(np.linalg.norm(model.tcp_world(x)[:3, 3] - goal)), 1),
            "visible_frac": round(float(np.mean(vis)), 2),
            "centre_px": round(float(np.mean(centre)), 1) if centre else None,
            "final_visible": bool(vfin), "memory_on": on,
            "s_per_step": round((time.perf_counter() - t) / (k + 1), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--goal", type=float, nargs=3, default=[0.70, 0.0, 0.45])
    ap.add_argument("--model", default="", help="TorchACPredictor checkpoint for C3-learned")
    ap.add_argument("--stride", type=int, default=4)
    a = ap.parse_args()
    m = WholeBodyModel()
    world = make_flower_scene()
    cam = eye_in_hand_pose(m)
    x0 = start_state(world, m, cam, np.random.default_rng(0))
    goal = np.array(a.goal)
    methods = {"C2": None, "C3-oracle": OracleStatePredictor(world, cam, stride=a.stride)}
    if a.model:
        from scout_piper_jepa.torch_predictor import TorchACPredictor  # noqa: PLC0415
        methods["C3-learned"] = StateConditionedPredictor(TorchACPredictor.load(a.model), m.kin.tcp, a.stride)
    for seed in range(a.seeds):
        for name, pred in methods.items():
            print(json.dumps(run(name, world, m, cam, x0, goal, seed, a.steps, pred)), flush=True)


if __name__ == "__main__":
    main()
