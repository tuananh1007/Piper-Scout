"""Calibration maths for the robot (P3A.7 / WE1, P0.2.6); no ROS.

* ``excitation_plan`` / ``resample_poses`` / ``identify_slip`` — skid-steer
  slip of the Scout: drive a fixed sequence of straight runs, arcs and turns
  on the spot, measure the pose with an *external* reference (motion capture,
  a fixed camera on an AprilTag, or a total station; wheel odometry cannot
  see slip), and fit ``k_v``, ``k_ω`` of ``dynamics/scout.py`` (``fit_slip``).
* ``pivot_calibration`` — TCP offset: touch one fixed point with the gripper
  tip from several flange orientations; least squares for the tip in the
  flange (link6) frame and the point in the base frame.
* ``hand_eye`` — eye-in-hand camera: solve A X = X B (Park & Martin) from
  flange poses (base → link6) and board poses (camera → board) of the same
  instants; X = link6 → camera.

The ROS recorders are in ``calibration_nodes.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .dynamics.scout import ScoutParams


# ------------------------------------------------------------- base slip
def excitation_plan(v: float = 0.15, w: float = 0.4, seg_s: float = 3.0, pause_s: float = 1.0
                    ) -> List[Tuple[float, float, float]]:
    """(duration s, v m/s, ω rad/s) segments covering straight motion, both
    arc directions and turns on the spot, each followed by a stop. Net
    displacement stays small (forward and back cancel)."""
    moves = [(v, 0.0), (-v, 0.0), (v, w), (-v, -w), (v, -w), (-v, w), (0.0, w), (0.0, -w)]
    plan = []
    for vv, ww in moves:
        plan.append((seg_s, vv, ww))
        plan.append((pause_s, 0.0, 0.0))
    return plan


def _unwrap(a: np.ndarray) -> np.ndarray:
    return np.unwrap(np.asarray(a, float))


def resample_poses(t_query: np.ndarray, t: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Linear interpolation of planar poses (N, 3) at t_query, heading unwrapped."""
    poses = np.asarray(poses, float)
    th = _unwrap(poses[:, 2])
    out = np.column_stack([np.interp(t_query, t, poses[:, 0]), np.interp(t_query, t, poses[:, 1]),
                           np.interp(t_query, t, th)])
    return out


def identify_slip(cmd_t: np.ndarray, cmd_vw: np.ndarray, pose_t: np.ndarray, poses: np.ndarray,
                  dt: float = 0.1, min_speed: float = 0.02, settle_s: float = 0.6) -> Tuple[ScoutParams, dict]:
    """Fit k_v, k_ω from commands (time-stamped, held until the next one) and
    externally measured poses. The commands are resampled every ``dt``. Left
    out of the fit: steps where both commanded speeds are below ``min_speed``
    (stops) and the first ``settle_s`` after every change of command, while
    the base is still accelerating (otherwise its lag reads as slip)."""
    cmd_t, cmd_vw = np.asarray(cmd_t, float), np.asarray(cmd_vw, float)
    t0, t1 = max(cmd_t[0], pose_t[0]), min(cmd_t[-1], pose_t[-1])
    grid = np.arange(t0, t1, dt)
    if len(grid) < 3:
        raise ValueError("not enough overlapping command and pose data")
    k = np.clip(np.searchsorted(cmd_t, grid, side="right") - 1, 0, len(cmd_t) - 1)
    vw = cmd_vw[k][:-1]
    P = resample_poses(grid, pose_t, poses)
    moving = (np.abs(vw[:, 0]) >= min_speed) | (np.abs(vw[:, 1]) >= min_speed)
    change = np.r_[True, np.any(np.abs(np.diff(vw, axis=0)) > 1e-6, axis=1)]
    last_change = grid[:-1][np.maximum.accumulate(np.where(change, np.arange(len(vw)), 0))]
    settled = grid[:-1] - last_change >= settle_s
    keep = np.flatnonzero(moving & settled)
    # fit_slip expects consecutive poses; build pose pairs for the kept steps
    d = np.diff(P, axis=0)[keep]
    th = P[:-1, 2][keep]
    fwd = d[:, 0] * np.cos(th) + d[:, 1] * np.sin(th)
    v_cmd, w_cmd = vw[keep, 0] * dt, vw[keep, 1] * dt
    k_v = float(v_cmd @ fwd / max(v_cmd @ v_cmd, 1e-12))
    k_w = float(w_cmd @ d[:, 2] / max(w_cmd @ w_cmd, 1e-12))
    res_v = fwd - k_v * v_cmd
    res_w = d[:, 2] - k_w * w_cmd
    report = {"k_v": round(k_v, 4), "k_omega": round(k_w, 4), "steps_used": int(len(keep)),
              "rms_forward_m_per_step": float(np.sqrt(np.mean(res_v ** 2))) if len(keep) else None,
              "rms_heading_rad_per_step": float(np.sqrt(np.mean(res_w ** 2))) if len(keep) else None}
    return ScoutParams(k_v=k_v, k_omega=k_w), report


# ------------------------------------------------------------------ TCP
@dataclass
class PivotResult:
    tcp_in_flange: np.ndarray      # (3,) tip position in the link6 frame
    pivot_in_base: np.ndarray      # (3,) touched point in the base frame
    rms_m: float                   # residual of the fixed-point constraint


def pivot_calibration(T_base_flange: Sequence[np.ndarray]) -> PivotResult:
    """R_i t + p_i = c for every pose: [R_i  −I] [t; c] = −p_i, least squares.
    Needs ≥ 3 poses with clearly different orientations (≥ 30° apart)."""
    T = np.asarray(T_base_flange, float)
    if len(T) < 3:
        raise ValueError("pivot calibration needs at least 3 poses")
    spread = max(np.degrees(np.linalg.norm(_log_rot(T[i, :3, :3].T @ T[j, :3, :3])))
                 for i in range(len(T)) for j in range(i + 1, len(T)))
    if spread < 20.0:
        raise ValueError(f"orientations too similar ({spread:.0f} deg apart at most; tilt the flange ≥ 30 deg)")
    A = np.concatenate([np.concatenate([T[i, :3, :3], -np.eye(3)], 1) for i in range(len(T))])
    b = -np.concatenate([T[i, :3, 3] for i in range(len(T))])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = A @ x - b
    return PivotResult(x[:3], x[3:], float(np.sqrt(np.mean(res.reshape(-1, 3) ** 2) * 3)))


# ------------------------------------------------------------- hand-eye
def _log_rot(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th / (2 * np.sin(th)) * v


def _inv(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


@dataclass
class HandEyeResult:
    T_flange_camera: np.ndarray    # (4, 4) link6 -> camera optical frame
    rot_residual_deg: float
    trans_residual_m: float


def hand_eye(T_base_flange: Sequence[np.ndarray], T_camera_board: Sequence[np.ndarray]) -> HandEyeResult:
    """Eye-in-hand calibration (Park & Martin 1994) from paired poses.

    The board is fixed in the base frame: E_i X C_i = E_j X C_j, so with
    A = E_j⁻¹ E_i and B = C_j C_i⁻¹, A X = X B. All pairs (i, j) are used.
    Needs ≥ 3 poses with rotations about at least two different axes."""
    E = [np.asarray(T, float) for T in T_base_flange]
    C = [np.asarray(T, float) for T in T_camera_board]
    if len(E) != len(C) or len(E) < 3:
        raise ValueError("hand-eye needs at least 3 paired poses")
    As, Bs = [], []
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            As.append(_inv(E[j]) @ E[i])
            Bs.append(C[j] @ _inv(C[i]))
    M = np.zeros((3, 3))
    for A, B in zip(As, Bs):
        M += np.outer(_log_rot(B[:3, :3]), _log_rot(A[:3, :3]))
    w, V = np.linalg.eigh(M.T @ M)
    if w.min() < 1e-10:
        raise ValueError("hand-eye poses are degenerate: rotate about at least two different axes")
    R = V @ np.diag(1.0 / np.sqrt(w)) @ V.T @ M.T
    lhs = np.concatenate([A[:3, :3] - np.eye(3) for A in As])
    rhs = np.concatenate([R @ B[:3, 3] - A[:3, 3] for A, B in zip(As, Bs)])
    t, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, t
    rot_err = [np.degrees(np.linalg.norm(_log_rot((A @ X)[:3, :3].T @ (X @ B)[:3, :3]))) for A, B in zip(As, Bs)]
    tr_err = [np.linalg.norm((A @ X)[:3, 3] - (X @ B)[:3, 3]) for A, B in zip(As, Bs)]
    return HandEyeResult(X, float(np.sqrt(np.mean(np.square(rot_err)))), float(np.sqrt(np.mean(np.square(tr_err)))))


def rpy_from_matrix(R: np.ndarray) -> Tuple[float, float, float]:
    """URDF roll-pitch-yaw (fixed axes X, Y, Z) of a rotation matrix."""
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) < 1e-9:
        return float(np.arctan2(-R[1, 2], R[1, 1])), pitch, 0.0
    return float(np.arctan2(R[2, 1], R[2, 2])), pitch, float(np.arctan2(R[1, 0], R[0, 0]))
