"""Grasp end pose chosen for the view (Stage C, open point of P3B.7).

A visibility cost on the trajectory keeps the flower in view during the
approach but does not decide what the camera sees at the end: near the goal
the goal cost takes over, and from some approach directions the flower is
hidden at every pose the MPC reaches (P3B.7). This module picks the end pose
itself. It samples whole-body poses that put the TCP on the goal
(``goal_poses``), keeps those whose eye-in-hand camera sees the target, and
returns the approach axis of the best one (``choose_view_end_pose``); the MPC
then gets a pose goal (``Goal(p, approach_axis)``). With the TCP fixed on the
goal the axis fixes the camera position and viewing direction, up to a roll
about the axis that only rotates the image.

Visibility of a candidate comes from a ``visible_fn(X, T_world_cam) -> bool``:
  * ``render_visible_fn``: renders the synthetic world (the oracle, as the
    oracle predictor; benchmarks only);
  * ``line_of_sight_visible_fn``: the target point inside the field of view and
    the ray to it free of obstacles; ``sphere_occluder`` (the synthetic world's
    points) or ``field_occluder`` (scout_piper_scene_repr distance-field
    snapshot: any class, leaves included, blocks the view) on the robot, with
    the target position from the Stage A memory (``p_world``).

Among the visible candidates the score prefers the target near the image
centre, a short move from the current state and a well-conditioned arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

VisibleFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
OccluderFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def goal_poses(model, goal: np.ndarray, n: int = 100_000, z_tol: float = 0.01,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Whole-body states (M, 9) with the TCP on ``goal``: arm joints sampled
    in their limits, kept when the TCP height is within ``z_tol``; base yaw
    sampled and the base position solved so the TCP lands on the goal."""
    rng = rng or np.random.default_rng(0)
    q = rng.uniform(model.kin.lower, model.kin.upper, (n, 6))
    X = np.concatenate([np.zeros((n, 3)), q], 1)
    p = model.tcp_world(X)[:, :3, 3]                     # TCP in the base frame (base at the origin)
    keep = np.abs(p[:, 2] - goal[2]) < z_tol
    X, p = X[keep], p[keep]
    th = rng.uniform(-np.pi, np.pi, len(X))
    c, s = np.cos(th), np.sin(th)
    X[:, 2] = th
    X[:, 0] = goal[0] - (c * p[:, 0] - s * p[:, 1])
    X[:, 1] = goal[1] - (s * p[:, 0] + c * p[:, 1])
    return X


def view_angle(T_world_cam: np.ndarray, target: np.ndarray):
    """(angle between the optical axis and the target direction, depth along it)."""
    d = np.asarray(target, float) - T_world_cam[..., :3, 3]
    z = T_world_cam[..., :3, 2]
    depth = (d * z).sum(-1)
    cosang = depth / np.maximum(np.linalg.norm(d, axis=-1), 1e-9)
    return np.arccos(np.clip(cosang, -1.0, 1.0)), depth


def render_visible_fn(world, min_cells: int = 2) -> VisibleFn:
    """Oracle visibility in the synthetic world (rendered labels)."""
    from .synthetic import target_truth  # noqa: PLC0415

    def fn(X, T):
        return target_truth(world.render(T)["label"], world.camera, min_cells=min_cells)[1]
    return fn


def sphere_occluder(centers: np.ndarray, radii: np.ndarray) -> OccluderFn:
    """Ray (camera -> target) blocked by any sphere it passes through, the
    target's own spheres excluded by the caller."""
    C, R = np.asarray(centers, float), np.asarray(radii, float)

    def fn(origins, target):
        o = np.atleast_2d(origins)
        d = np.asarray(target, float) - o                       # (M, 3)
        L = np.linalg.norm(d, axis=1)
        u = d / L[:, None]
        rel = C[None] - o[:, None]                              # (M, S, 3)
        t = np.clip((rel * u[:, None]).sum(-1), 0.0, None)
        closest = np.linalg.norm(rel - t[..., None] * u[:, None], axis=-1)
        return np.any((closest < R[None]) & (t < L[:, None] - R[None]), axis=1)
    return fn


def field_occluder(snapshot, step_m: float = 0.005, target_clear_m: float = 0.03,
                   block_m: Optional[float] = None) -> OccluderFn:
    """Ray blocked when any sample on it comes within ``block_m`` (default half
    a voxel: a leaf thinner than a voxel interpolates to a small positive
    distance) of an obstacle of the semantic field, hard or soft (a leaf hides
    the flower although the arm may brush it); the last ``target_clear_m``
    before the target is not tested (the target's own voxels). Unknown space
    does not block."""
    from scout_piper_scene_repr_py.field import FieldSampler  # noqa: PLC0415
    sampler = FieldSampler(snapshot)
    block = 0.5 * snapshot.voxel_size if block_m is None else block_m

    def fn(origins, target):
        o = np.atleast_2d(origins)
        d = np.asarray(target, float) - o
        L = np.linalg.norm(d, axis=1)
        k = int(np.ceil(L.max() / step_m))
        s = np.linspace(0.0, 1.0, k + 1)[None, :, None]
        pts = o[:, None] + s * d[:, None]                       # (M, k+1, 3)
        q = sampler.query(pts.reshape(-1, 3), snapshot.stamp, np.inf)
        inside = (q["hard"] < block) | (q["soft"] < block) if q["soft"] is not None else q["hard"] < block
        inside = (inside & q["in_bounds"]).reshape(len(o), -1)
        near_target = (1.0 - s[0, :, 0])[None] * L[:, None] < target_clear_m
        return np.any(inside & ~near_target, axis=1)
    return fn


def line_of_sight_visible_fn(target: np.ndarray, occluded: OccluderFn, half_fov_rad: float,
                             near_m: float = 0.07) -> VisibleFn:
    """Target inside a cone of ``half_fov_rad`` around the optical axis,
    farther than ``near_m`` (the D405's near limit) and not occluded."""
    target = np.asarray(target, float)

    def fn(X, T):
        ang, depth = view_angle(T, target)
        ok = (ang < half_fov_rad) & (depth > near_m)
        out = np.zeros(len(T), bool)
        if ok.any():
            out[ok] = ~occluded(T[ok, :3, 3], target)
        return out
    return fn


@dataclass
class ViewChoice:
    state: np.ndarray          # (9,) whole-body pose on the goal
    axis: np.ndarray           # (3,) its TCP approach axis (world): Goal.approach_axis
    angle_deg: float           # target off the optical axis at that pose
    candidates: int            # poses on the goal sampled
    visible: int               # of those checked, how many see the target
    checked: int


def choose_view_end_pose(model, cam, goal: np.ndarray, target: np.ndarray, x_now: np.ndarray,
                         visible_fn: VisibleFn, n: int = 100_000, max_check: int = 400,
                         half_fov_rad: float = np.radians(35.0), w_angle: float = 1.0, w_move: float = 0.5,
                         w_manip: float = 0.3, w_turn: float = 1.0, rng: Optional[np.random.Generator] = None) -> Optional[ViewChoice]:
    """Best end pose on ``goal`` whose camera sees ``target``, or None.

    Candidates with the target inside the cone are ranked by angle off the
    optical axis (rad), move from ``x_now`` (base metres + 0.3 × yaw and joint
    radians), turn of the approach axis from the current one (rad: the MPC
    reaches a pose goal far from its current axis slowly) and manipulability
    (relative to the best candidate's); ``visible_fn`` (possibly expensive) is
    evaluated on the best ``max_check`` of them and the best visible one wins."""
    X = goal_poses(model, np.asarray(goal, float), n=n, rng=rng)
    if len(X) == 0:
        return None
    T = cam(X)
    ang, depth = view_angle(T, target)
    cone = (ang < half_fov_rad) & (depth > 0.05)
    if not cone.any():
        return None
    Xc, Tc, ac = X[cone], T[cone], ang[cone]
    x_now = np.asarray(x_now, float)
    dyaw = np.angle(np.exp(1j * (Xc[:, 2] - x_now[2])))
    move = np.linalg.norm(Xc[:, :2] - x_now[:2], axis=1) + 0.3 * (np.abs(dyaw) + np.abs(Xc[:, 3:] - x_now[3:]).sum(1))
    manip = model.kin.manipulability(Xc[:, 3:])
    axes = model.tcp_world(Xc)[:, :3, 2]
    turn = np.arccos(np.clip(axes @ model.tcp_world(x_now)[:3, 2], -1.0, 1.0))
    score = (w_angle * ac + w_move * move + w_turn * turn
             + w_manip * (1.0 - manip / max(float(manip.max()), 1e-12)))
    order = np.argsort(score)[:max_check]
    vis = np.asarray(visible_fn(Xc[order], Tc[order]), bool)
    if not vis.any():
        return None
    best = order[np.flatnonzero(vis)[0]]
    axis = axes[best]
    return ViewChoice(state=Xc[best].copy(), axis=axis / np.linalg.norm(axis), angle_deg=float(np.degrees(ac[best])),
                      candidates=len(X), visible=int(vis.sum()), checked=len(order))
