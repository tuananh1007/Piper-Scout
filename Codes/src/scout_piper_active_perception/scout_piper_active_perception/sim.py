"""Simulated active-perception episodes (P4.3 closed loop, P4.4 ablations).

One episode: the robot starts at the scan pose the reach leaves it in (camera
about 0.4 m in front of the target on the approach line, looking at it), the
first view is taken, then until the required uncertainty terms are met or
``max_views`` is reached:

  1. assess U (stem joint σ, approach-corridor unknown fraction, target pixels)
  2. sample reachable view candidates around the current joint estimate,
     within the whole-body MPC's reach margin (``reach_max_m``; None = any
     kinematically valid arm pose, an upper bound the real MPC does not
     reach), checked against the *mapped* scene (occupied and unknown voxels
     block, as the MPC's no_entry policy would; outside the map the true
     obstacles stand in for the rest of the planning scene)
  3. the policy picks one; the robot moves there: ``executor="teleport"``
     (time from ``move_time``) or ``"mpc"`` (the whole-body MPPI of
     scout_piper_whole_body_mpc with its safety filter, to the view's TCP
     pose; time = MPC steps)
  4. render at the true camera pose (the reached state, with a per-view
     hand-eye/FK pose error), sense (noise, thin-structure dropout),
     integrate into the semantic voxel map at the *believed* pose, add the
     target points to the joint estimate and refit

Scored against the truth: axis error of the joint (max over ±3 cm), unknown
fraction of the approach corridor, success = error ≤ 2 mm and corridor
unknown ≤ 5 %, views, acquisition time, and the robot's closest approach to
the true stems (hard) and leaves along the visited states.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

from scout_piper_scene_repr_py.voxel_map import SemanticVoxelMap, grid_around
from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel

from .candidates import camera_pose_fn, move_time, sample_views
from .grid import OccupancyGrid, corridor_mask
from .nbv import ViewPlanner
from .scene import CLASSES, Camera, DepthSensor, backproject, make_ap_scene, render
from .stem_joint import NoiseModel, StemJointEstimator, axis_error
from .uncertainty import Weights, assess, u_jepa_from_pixels


def _small_rotation(rng, sigma_rad):
    w = rng.normal(0.0, sigma_rad, 3)
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def blocked_distance_fn(grid: OccupancyGrid, outside_fn=None):
    """Distance to the nearest occupied or unknown voxel (minus half a voxel)
    inside the grid, ``outside_fn`` (or free) outside it; (dist, valid)."""
    blocked = grid.occupied | grid.unknown
    D = distance_transform_edt(~blocked) * grid.voxel_size - 0.5 * grid.voxel_size if blocked.any() else None

    def fn(P):
        P = np.asarray(P, float).reshape(-1, 3)
        idx, inb = grid.index(P)
        out = np.full(len(P), np.inf)
        if outside_fn is not None and (~inb).any():
            out[~inb] = outside_fn(P[~inb])[0]
        if D is not None and inb.any():
            i = idx[inb]
            out[inb] = D[i[:, 0], i[:, 1], i[:, 2]]
        return out, np.ones(len(P), bool)
    return fn


@dataclass
class EpisodeResult:
    seed: int
    policy: str
    views: int
    time_s: float
    axis_error_mm: float
    axis_sigma_mm: float
    corridor_unknown: float
    success: bool
    stop: str
    oracle_views: Optional[int]
    min_hard_clearance_m: float
    max_leaf_penetration_m: float
    target_kind: str
    first_view_target_px: int
    trace: List[Dict] = field(default_factory=list)

    def row(self) -> Dict:
        d = dict(self.__dict__)
        d.pop("trace")
        return d


def initial_state(model, cam_fn, scene, guess, axis, standoff_m=0.4, rng=None):
    """The scan pose after the reach: camera on the approach line, ``standoff_m``
    from the target, looking at it; collision-free in the true scene."""
    rng = rng or np.random.default_rng(0)
    x_ref = np.r_[guess[:2] - (standoff_m + 0.45) * axis[:2], np.arctan2(axis[1], axis[0]), np.zeros(6)]
    cand = sample_views(model, cam_fn, guess, x_ref, n=40000, d_range=(standoff_m - 0.05, standoff_m + 0.05),
                        max_base_move_m=3.0, distance_fn=scene.plant.distance_fn(), clearance_m=0.03,
                        max_candidates=4000, rng=rng)
    if not len(cand.X):
        raise RuntimeError("no scan pose")
    view_dir = cand.T[:, :3, 2]
    off_axis = np.arccos(np.clip(view_dir @ axis, -1, 1))
    cam_off_line = np.linalg.norm(np.cross(cand.T[:, :3, 3] - guess, axis), axis=1)
    i = int(np.argmin(off_axis + 5.0 * cam_off_line))
    return cand.X[i]


def run_episode(seed: int, policy: str, max_views: int = 8, camera: Optional[Camera] = None,
                sensor: Optional[DepthSensor] = None, pose_sigma_m: float = 0.0005, pose_sigma_deg: float = 0.1,
                guess_sigma_m: float = 0.01, lambda_t: float = 0.02, weights: Optional[Weights] = None,
                executor: str = "teleport", n_candidates: int = 200, mpc_samples: int = 256,
                mpc_steps: int = 150, reach_max_m: Optional[float] = 0.36, record_trace: bool = False) -> EpisodeResult:
    camera = camera or Camera()
    sensor = sensor or DepthSensor()
    rng = np.random.default_rng(seed)
    scene = make_ap_scene(seed)
    model = WholeBodyModel()
    cam_fn = camera_pose_fn(model)
    p_true, d_true, r_true = scene.joint_axis()
    guess = scene.target + rng.normal(0.0, guess_sigma_m, 3)
    axis = np.r_[guess[:2], 0.0] / np.linalg.norm(guess[:2])

    def corridor():
        """Approach segment: 15 cm before the grasp point up to 1 cm before it
        (the stem's own surface is not corridor), on the current estimate."""
        p = est.cyl.p if est.fitted else guess
        return (p - 0.15 * axis, p - 0.01 * axis)
    vmap = SemanticVoxelMap(grid_around(guess, 0.25, 0.01), classes=CLASSES)
    noise = NoiseModel(sigma0=sensor.sigma0, k2=sensor.k2, f_px=camera.f, thin_lo_px=sensor.thin_lo_px,
                       thin_hi_px=sensor.thin_hi_px, pose_m=pose_sigma_m, pose_rad=float(np.radians(pose_sigma_deg)))
    est = StemJointEstimator(guess, noise=noise, rng=np.random.default_rng(seed + 7))
    planner = ViewPlanner(model, camera.K, camera.image_hw, camera.near_m, camera.far_m, lambda_t=lambda_t,
                          rng=np.random.default_rng(seed + 11))
    hard_true = scene.plant.distance_fn()
    from scout_piper_whole_body_mpc.scene_adapter import disc_distance  # noqa: PLC0415

    def leaf_d(P):
        pl = scene.plant
        return disc_distance(P, pl.leaf_c, pl.leaf_n, pl.leaf_r).min(1) if len(pl.leaf_c) else np.full(len(P), np.inf)

    def outside_true(P):
        """Stand-in for the rest of the planning scene outside the map:
        the true stems, pots and leaves."""
        d, v = hard_true(P)
        return np.minimum(d, leaf_d(P)), v

    x = initial_state(model, cam_fn, scene, guess, axis, rng=np.random.default_rng(seed + 3))
    visited = [x.copy()]
    stamp, t_total, views, trace = 0.0, 0.0, 0, []
    last_px = 0

    def take_view(x):
        nonlocal stamp, last_px
        T_bel = cam_fn(x)
        T_true = T_bel.copy()
        T_true[:3, :3] = T_bel[:3, :3] @ _small_rotation(rng, np.radians(pose_sigma_deg))
        T_true[:3, 3] = T_bel[:3, 3] + rng.normal(0.0, pose_sigma_m, 3)
        fr = sensor.sense(render(scene, T_true, camera), camera, rng)
        masks = {c: fr["label"] == i for i, c in enumerate(CLASSES)}
        stamp += 1.0
        vmap.integrate(np.nan_to_num(fr["depth"], nan=0.0), masks, camera.K, T_bel, stamp, stride=2)
        P = backproject(fr["depth"], camera.K, T_bel, masks["target"])
        last_px = int(masks["target"].sum())
        est.add_view(P, T_bel[:3, 3])
        est.fit()
        return last_px

    first_px = take_view(x)

    def state():
        grid = OccupancyGrid.from_voxel_map(vmap)
        cmask = corridor_mask(grid, *corridor(), radius_m=0.03, exclude=est.target_exclusion(0.01))
        u = assess(est.sigma() if est.fitted else 1.0, grid.unknown_fraction(cmask), u_jepa_from_pixels(last_px),
                   weights=weights)
        return grid, cmask, u

    def true_success(grid, cmask):
        err = axis_error(est.cyl, p_true, d_true, est.half_len) if est.fitted else np.inf
        return err, grid.unknown_fraction(cmask)

    grid, cmask, u = state()
    err, cu = true_success(grid, cmask)
    oracle = 0 if (err <= 0.002 and cu <= 0.05) else None
    stop = "requirements met"
    while u.needs_view:
        if views >= max_views:
            stop = "max views"
            break
        tgt = est.cyl.p if est.fitted else guess
        dist_fn = blocked_distance_fn(grid, outside_fn=outside_true)
        cand = sample_views(model, cam_fn, tgt, x, distance_fn=dist_fn, clearance_m=0.02, reach_max_m=reach_max_m,
                            max_candidates=n_candidates, rng=np.random.default_rng(seed * 100 + views))
        ch = planner.choose(policy, cand, x, est, grid, u, corridor(), axis)
        if ch is None:
            stop = "no reachable view"
            break
        x_goal = cand.X[ch["index"]]
        if executor == "mpc":
            x_new, dt = _drive_mpc(model, x, x_goal, dist_fn, mpc_samples, mpc_steps, seed + views)
        else:
            x_new, dt = x_goal, float(move_time(model, x, x_goal[None])[0])
        t_total += dt + 0.3                                   # + capture
        x = x_new
        visited.append(x.copy())
        px = take_view(x)
        views += 1
        grid, cmask, u = state()
        err, cu = true_success(grid, cmask)
        if oracle is None and err <= 0.002 and cu <= 0.05:
            oracle = views
        if record_trace:
            trace.append({"view": views, "x": x.round(4).tolist(), "target_px": px, "choice": ch,
                          "axis_error_mm": round(1000 * err, 3), "u": u.as_dict()})

    V = np.array(visited)
    C, r = model.collision_spheres(V)
    hard = float((hard_true(C.reshape(-1, 3))[0].reshape(C.shape[:-1]) - r).min())
    pen = float(np.clip(-(leaf_d(C.reshape(-1, 3)).reshape(C.shape[:-1]) - r), 0, None).max())
    err, cu = true_success(grid, cmask)
    return EpisodeResult(seed=seed, policy=policy, views=views, time_s=round(t_total, 2),
                         axis_error_mm=round(1000 * err, 3) if np.isfinite(err) else float("inf"),
                         axis_sigma_mm=round(1000 * est.sigma(), 3) if est.fitted else float("inf"),
                         corridor_unknown=round(cu, 4), success=bool(err <= 0.002 and cu <= 0.05), stop=stop,
                         oracle_views=oracle, min_hard_clearance_m=round(hard, 4), max_leaf_penetration_m=round(pen, 4),
                         target_kind=scene.plant.kinds[scene.target_capsule], first_view_target_px=first_px, trace=trace)


def _drive_mpc(model, x0, x_goal, distance_fn, samples, steps, seed):
    """Whole-body MPPI + safety filter to the TCP pose of ``x_goal``."""
    from scout_piper_whole_body_mpc.costs.terms import CostWeights, Goal, WholeBodyCost  # noqa: PLC0415
    from scout_piper_whole_body_mpc.safety.projection import SafetyFilter  # noqa: PLC0415
    from scout_piper_whole_body_mpc.sim import run_closed_loop  # noqa: PLC0415
    from scout_piper_whole_body_mpc.solvers.mppi import MPPI, MPPIConfig  # noqa: PLC0415
    T = model.tcp_world(x_goal)
    cost = WholeBodyCost(model, Goal(p=T[:3, 3], approach_axis=T[:3, 2]), distance_fn=distance_fn, w=CostWeights())
    log = run_closed_loop(np.asarray(x0, float), MPPI(model, MPPIConfig(seed=seed, samples=samples)), cost,
                          SafetyFilter(model, distance_fn=distance_fn), steps=steps)
    return np.array(log.X[-1]), (len(log.X) - 1) * model.dt + 0.5


if __name__ == "__main__":
    t0 = time.time()
    r = run_episode(0, "fused", record_trace=True)
    print(r.row(), f"{time.time() - t0:.1f} s")
