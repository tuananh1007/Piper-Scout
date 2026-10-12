"""Reachable camera poses looking at the target (view candidates).

Two kinds of whole-body states x = (x_b, y_b, θ_b, q₁..q₆) are sampled:

  * base-solved: arm joints sampled within their limits (less the 0.15 rad
    margin moveit_servo needs); the eye-in-hand camera's optical axis must
    pass through the target's height at a distance d = (z_t − c_z) / z_z in
    ``d_range``; a base yaw is sampled and the base position follows, so the
    optical axis hits the target exactly. Kept when the base moves at most
    ``max_base_move_m``;
  * arm-only: the base stays where it is, arm joints sampled, kept when the
    target lies within ``cone_deg`` of the optical axis and within ``d_range``.

Both keep the arm's shoulder-to-wrist distance within the whole-body MPC's
reach margin (``reach_max_m``, 0.36 m as ``whole_body_mpc.yaml``): the MPC
penalises a stretched arm and does not reach such poses (on the fake robot it
stalled 25-35 cm short of them). Both are then checked for collision: the
robot's spheres against the hard obstacles (``distance_fn``, ``clearance_m``),
the camera centre and the TCP against the leaves (``leaf_distance_fn``; a view
pose inside a leaf is no view). ``move_time`` estimates the time to get there (turn, drive, turn for
the base, slowest joint for the arm, whichever is longer, plus settling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from scout_piper_whole_body_mpc.costs.terms import JOINT_LIMIT_MARGIN


def camera_pose_fn(model, T_link_cam: Optional[np.ndarray] = None, link_index: int = 6):
    """T_world_cam(X) = link6(X) · T_link_cam (the hand-eye transform; on the
    robot read from TF, in simulation an offset along link6 z)."""
    if T_link_cam is None:
        T_link_cam = np.eye(4)
        T_link_cam[2, 3] = 0.03
    T_link_cam = np.asarray(T_link_cam, float)

    def fn(X):
        return model.world_frames(np.asarray(X, float))[..., link_index, :, :] @ T_link_cam
    return fn


@dataclass
class Candidates:
    X: np.ndarray            # (M, 9) whole-body states
    T: np.ndarray            # (M, 4, 4) camera poses
    arm_only: np.ndarray     # (M,) bool


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def sample_views(model, cam_fn, target: np.ndarray, x_now: np.ndarray, n: int = 120000,
                 d_range: Tuple[float, float] = (0.2, 0.45), max_base_move_m: float = 1.0,
                 cone_deg: float = 10.0, distance_fn: Optional[Callable] = None,
                 leaf_distance_fn: Optional[Callable] = None, clearance_m: float = 0.02,
                 max_candidates: int = 300, reach_max_m: Optional[float] = 0.36,
                 rng: Optional[np.random.Generator] = None) -> Candidates:
    rng = rng or np.random.default_rng(0)
    target = np.asarray(target, float)
    x_now = np.asarray(x_now, float)
    lo, hi = model.kin.lower + JOINT_LIMIT_MARGIN, model.kin.upper - JOINT_LIMIT_MARGIN

    def arm_q(k):
        q = rng.uniform(lo, hi, (k, 6))
        return q if reach_max_m is None else q[model.kin.wrist_extension(q) <= reach_max_m]

    # base-solved
    qb = arm_q(n)
    Xb = np.zeros((len(qb), 9))
    Xb[:, 3:] = qb
    T0 = cam_fn(Xb)
    c, z = T0[:, :3, 3], T0[:, :3, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (target[2] - c[:, 2]) / z[:, 2]
    keep = np.isfinite(d) & (d >= d_range[0]) & (d <= d_range[1])
    Xb, c, z, d = Xb[keep], c[keep], z[keep], d[keep]
    th = x_now[2] + rng.uniform(-np.pi, np.pi, len(Xb))
    hit = c + d[:, None] * z                                      # point the optical axis reaches, base frame
    ct, st = np.cos(th), np.sin(th)
    Xb[:, 2] = th
    Xb[:, 0] = target[0] - (ct * hit[:, 0] - st * hit[:, 1])
    Xb[:, 1] = target[1] - (st * hit[:, 0] + ct * hit[:, 1])
    Xb = Xb[np.linalg.norm(Xb[:, :2] - x_now[:2], axis=1) <= max_base_move_m]

    # arm-only
    qa = arm_q(n)
    Xa = np.tile(x_now, (len(qa), 1))
    Xa[:, 3:] = qa
    Ta = cam_fn(Xa)
    rel = target - Ta[:, :3, 3]
    depth = (rel * Ta[:, :3, 2]).sum(1)
    ang = np.arccos(np.clip(depth / np.maximum(np.linalg.norm(rel, axis=1), 1e-9), -1, 1))
    Xa = Xa[(ang < np.radians(cone_deg)) & (depth >= d_range[0]) & (depth <= d_range[1])]

    X = np.concatenate([Xb, Xa])
    arm_only = np.r_[np.zeros(len(Xb), bool), np.ones(len(Xa), bool)]
    if len(X) == 0:
        return Candidates(np.zeros((0, 9)), np.zeros((0, 4, 4)), np.zeros(0, bool))
    ok = np.ones(len(X), bool)
    if distance_fn is not None:
        C, r = model.collision_spheres(X)
        dist, _ = distance_fn(C.reshape(-1, 3))
        ok &= (np.asarray(dist).reshape(C.shape[:-1]) - r).min(1) > clearance_m
    T = cam_fn(X)
    if leaf_distance_fn is not None:
        tcp = model.tcp_world(X)[:, :3, 3]
        ok &= (leaf_distance_fn(T[:, :3, 3]) > clearance_m) & (leaf_distance_fn(tcp) > clearance_m)
    X, T, arm_only = X[ok], T[ok], arm_only[ok]
    if len(X) > max_candidates:                                   # keep both kinds in proportion, at least a third arm-only
        na = min(int(arm_only.sum()), max(max_candidates // 3, int(max_candidates * arm_only.mean())))
        ia = rng.permutation(np.flatnonzero(arm_only))[:na]
        ib = rng.permutation(np.flatnonzero(~arm_only))[:max_candidates - len(ia)]
        sel = np.r_[ia, ib]
        X, T, arm_only = X[sel], T[sel], arm_only[sel]
    return Candidates(X, T, arm_only)


def axis_turn(model, x_from: np.ndarray, X_to: np.ndarray) -> np.ndarray:
    """Angle (rad) the TCP approach axis turns from ``x_from`` to each of ``X_to``."""
    a0 = model.tcp_world(np.asarray(x_from, float))[:3, 2]
    A = model.tcp_world(np.atleast_2d(X_to))[:, :3, 2]
    return np.arccos(np.clip(A @ a0, -1.0, 1.0))


def move_time(model, x_from: np.ndarray, X_to: np.ndarray, settle_s: float = 0.5) -> np.ndarray:
    """Seconds to move from ``x_from`` to each of ``X_to`` (M, 9)."""
    X_to = np.atleast_2d(X_to)
    v, w, qd = model.scout.v_max, model.scout.omega_max, model.arm.qd_max
    dxy = X_to[:, :2] - x_from[:2]
    dist = np.linalg.norm(dxy, axis=1)
    heading = np.arctan2(dxy[:, 1], dxy[:, 0])
    drive = (np.abs(_wrap(heading - x_from[2])) + np.abs(_wrap(X_to[:, 2] - heading))) / w + dist / v
    turn = np.abs(_wrap(X_to[:, 2] - x_from[2])) / w
    t_base = np.where(dist > 0.02, drive, turn)
    t_arm = np.abs(X_to[:, 3:] - x_from[3:]).max(1) / qd
    return np.maximum(t_base, t_arm) + settle_s
