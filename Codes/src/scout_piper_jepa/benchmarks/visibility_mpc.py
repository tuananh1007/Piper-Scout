#!/usr/bin/env python3
"""Stage C on the synthetic world: geometry-only MPC (C2) vs JEPA-aware MPC (C3).

The whole-body MPC (scout_piper_whole_body_mpc) drives the TCP to a pre-grasp
point; C3 adds ``JepaVisibilityCost`` through ``WholeBodyCost.extra``. The
predictor is either the oracle (renders the true future: the upper bound on
what prediction can buy) or a trained checkpoint (``--model``, from
torch_predictor; needs torch).

The default goal sits between the flower and its identical twin: a camera
looking straight at it sees the twin while the leaf hides the flower, so
geometry alone can end with the wrong flower in view. ``view_feasibility``
first checks that the scene allows a view-preserving end pose: of the sampled
whole-body poses that put the TCP on the goal, the fraction that see the
flower. C3 runs with the visibility cost additive (``C3-*``, the P3B.7 setup)
and secondary (``C3c-*``: it may cost at most ``--tol`` of goal error,
``WholeBodyCost.secondary``). ``C2v`` / ``C3cv`` add the end pose chosen for
the view (``view_pose.choose_view_end_pose``: the approach axis of a goal pose
whose camera has a free line of sight to the flower, the leaf and the twin
as occluders; ``--view-oracle`` scores by rendering instead) as a pose goal
with orientation weight ``--view-w-orient``.
``--hard N`` first runs C2 (cheap) from N
random start poses that see the flower and keeps the ``--hard-k`` where C2
loses the view most: only there can a visibility cost show a gain. Reported
per run:
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
from scout_piper_jepa.view_pose import (choose_view_end_pose, line_of_sight_visible_fn, render_visible_fn,
                                        sphere_occluder)
from scout_piper_whole_body_mpc.costs.terms import CostWeights, Goal, WholeBodyCost
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


def view_feasibility(world, model, cam, goal, n=200_000, z_tol=0.01, seed=0):
    """Whole-body poses with the TCP within ``z_tol`` of the goal (arm joints
    sampled, base yaw sampled, base x, y solved so the TCP is on the goal):
    (number of such poses, fraction that see the flower, fraction that see it
    within a quarter image width of the centre)."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(model.kin.lower, model.kin.upper, (n, 6))
    X = np.concatenate([np.zeros((n, 3)), q], 1)
    p = model.tcp_world(X)[:, :3, 3]                        # TCP in the base frame (base at the origin)
    keep = np.abs(p[:, 2] - goal[2]) < z_tol
    X, p = X[keep], p[keep]
    th = rng.uniform(-np.pi, np.pi, len(X))
    c, s = np.cos(th), np.sin(th)
    X[:, 2] = th
    X[:, 0] = goal[0] - (c * p[:, 0] - s * p[:, 1])
    X[:, 1] = goal[1] - (s * p[:, 0] + c * p[:, 1])
    u, vis = target_truth(world.render(cam(X))["label"], world.camera)
    H, W = world.camera.image_hw
    centred = vis & (np.linalg.norm(u - [W / 2, H / 2], axis=1) < W / 4)
    return len(X), float(vis.mean()) if len(X) else 0.0, float(centred.mean()) if len(X) else 0.0


def candidate_starts(world, model, cam, n, rng, min_dist=0.25, pool=40000):
    """n start states (base at the origin) whose camera sees the flower, TCP
    farther than ``min_dist`` from it, otherwise random."""
    cand = np.concatenate([np.zeros((pool, 3)), rng.uniform(model.kin.lower, model.kin.upper, (pool, 6))], 1)
    far = np.linalg.norm(model.tcp_world(cand)[:, :3, 3] - world.target_position(), axis=1) > min_dist
    cand = cand[far]
    _, vis = target_truth(world.render(cam(cand))["label"], world.camera)
    cand = cand[vis]
    return cand[rng.choice(len(cand), min(n, len(cand)), replace=False)]


def upsample(mask_grid, image_hw):
    hf, wf = mask_grid.shape
    H, W = image_hw
    return np.kron(mask_grid, np.ones((H // hf, W // wf), bool))


def upsample_depth(depth_grid, image_hw):
    hf, wf = depth_grid.shape
    H, W = image_hw
    return np.kron(np.where(np.isfinite(depth_grid), depth_grid, 0.0), np.ones((H // hf, W // wf)))


def view_axis(world, model, cam, x0, goal, oracle=False):
    """Approach axis of the end pose chosen for the view (None if none sees the flower)."""
    target = world.target_position()
    if oracle:
        vis = render_visible_fn(world)
    else:
        other = world.labels != TARGET
        vis = line_of_sight_visible_fn(target, sphere_occluder(world.positions[other], world.radii[other]),
                                       half_fov_rad=np.radians(30.0))
    ch = choose_view_end_pose(model, cam, goal, target, x0, vis, rng=np.random.default_rng(0))
    return None if ch is None else ch.axis


def run(method, world, model, cam, x0, goal, seed, steps, predictor=None, samples=64, anchor=True,
        anchor_tol_m=0.03, tol_m=0.01, axis=None, w_orient=2.0):
    r_t = world.features[world.labels == TARGET][0]
    H, W = world.camera.image_hw
    vc = None
    if method not in ("C2", "C2v"):
        vc = JepaVisibilityCost(predictor, world.camera.image_hw, camera_pose=cam, K=world.camera.K,
                                stride=predictor.stride)
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
        # a fresh cost per control step (as the node builds it): the secondary
        # constraint's best goal error is per step
        cost = WholeBodyCost(model, Goal(p=goal, approach_axis=axis), w=CostWeights(orient=w_orient))
        if vc is not None:
            (cost.secondary if method.startswith("C3c") else cost.extra).append(vc)   # C3c, C3cv
            cost.secondary_tol_m = tol_m
        uc = mppi.solve(x, cost, u_prev)
        rep = sf.project(x, uc, u_prev)
        x = model.rollout(x, rep.u[None, None])[0, 1]
        u_prev = rep.u
        mppi.shift()
        T_now = model.tcp_world(x)
        if (np.linalg.norm(T_now[:3, 3] - goal) < 0.01 and np.abs(rep.u).max() < 0.05
                and (axis is None or T_now[:3, 2] @ axis > np.cos(np.radians(5.0)))):
            break
    rr = world.render(cam(x), rng=rng)
    st = mem.update(rr["Z"], image_hw=(H, W))
    lab_up = np.kron(rr["label"], np.ones((H // rr["label"].shape[0], W // rr["label"].shape[1]), int))
    on = "lost"
    if st.status != "lost":
        ui, vi = int(np.clip(st.u_mean[0], 0, W - 1)), int(np.clip(st.u_mean[1], 0, H - 1))
        on = {TARGET: "target", DISTRACTOR: "twin"}.get(int(lab_up[vi, ui]), "elsewhere")
    _, vfin = target_truth(rr["label"], world.camera)
    z = model.tcp_world(x)[:3, 2]
    return {"method": method, "seed": seed, "steps": k + 1,
            "axis_err_deg": None if axis is None else round(float(np.degrees(np.arccos(np.clip(z @ axis, -1, 1)))), 1),
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
    ap.add_argument("--samples", type=int, default=64, help="MPPI samples")
    ap.add_argument("--tol", type=float, default=0.01, help="m of goal error the C3c visibility cost may cost")
    ap.add_argument("--methods", default="C2,C3-oracle,C3c-oracle",
                    help="of C2, C2v, C3-oracle, C3c-oracle, C3cv-oracle, C3-learned, C3c-learned, C3cv-learned")
    ap.add_argument("--view-oracle", action="store_true", help="choose the C2v / C3cv end pose by rendering")
    ap.add_argument("--view-w-orient", type=float, default=30.0,
                    help="orientation weight for the C2v / C3cv pose goal (the node's w_orient is 2)")
    ap.add_argument("--hard", type=int, default=0, help="screen N random starts with C2, keep the worst")
    ap.add_argument("--hard-k", type=int, default=3)
    ap.add_argument("--stride", type=int, default=0,
                    help="controller steps per predictor step (0: from --model's training interval, "
                         "2 if it has none; without --model 4)")
    a = ap.parse_args()
    m = WholeBodyModel()
    world = make_flower_scene()
    cam = eye_in_hand_pose(m)
    x0 = start_state(world, m, cam, np.random.default_rng(0))
    goal = np.array(a.goal)
    net = None
    if a.model:
        from scout_piper_jepa.torch_predictor import TorchACPredictor, predictor_stride  # noqa: PLC0415
        net = TorchACPredictor.load(a.model)
        fit, warning = predictor_stride(net.cfg.step_s, m.dt)
        if warning:
            print(warning)
        a.stride = a.stride or fit
    a.stride = a.stride or 4
    oracle = OracleStatePredictor(world, cam, stride=a.stride)
    preds = {"C2": None, "C2v": None, "C3-oracle": oracle, "C3c-oracle": oracle, "C3cv-oracle": oracle}
    if net is not None:
        preds["C3-learned"] = preds["C3c-learned"] = preds["C3cv-learned"] = \
            StateConditionedPredictor(net, m.kin.tcp, a.stride)

    def axis_for(name, xs):
        return view_axis(world, m, cam, xs, goal, a.view_oracle) if name == "C2v" or "cv-" in name else None
    names = [n for n in a.methods.split(",") if n in preds]
    n_goal, f_vis, f_centre = view_feasibility(world, m, cam, goal)
    print(json.dumps({"feasibility": {"goal": list(goal), "goal_poses": n_goal, "flower_visible": round(f_vis, 3),
                                      "flower_near_centre": round(f_centre, 3)}}), flush=True)
    if a.hard:
        starts = candidate_starts(world, m, cam, a.hard, np.random.default_rng(1))
        screen = []
        for i, xs in enumerate(starts):
            r = run("C2", world, m, cam, xs, goal, 0, a.steps, None, samples=a.samples)
            screen.append((r["visible_frac"], i))
            print(json.dumps({"screen": i, **r}), flush=True)
        worst = [i for _, i in sorted(screen)[:a.hard_k]]
        for i in worst:
            for name in names:
                ax = axis_for(name, starts[i])
                r = run(name, world, m, cam, starts[i], goal, 0, a.steps, preds[name], samples=a.samples, tol_m=a.tol,
                        axis=ax, w_orient=a.view_w_orient if ax is not None else 2.0)
                print(json.dumps({"start": i, **r}), flush=True)
        return
    for seed in range(a.seeds):
        for name in names:
            ax = axis_for(name, x0)
            print(json.dumps(run(name, world, m, cam, x0, goal, seed, a.steps, preds[name], samples=a.samples,
                                 tol_m=a.tol, axis=ax, w_orient=a.view_w_orient if ax is not None else 2.0)),
                  flush=True)


if __name__ == "__main__":
    main()
