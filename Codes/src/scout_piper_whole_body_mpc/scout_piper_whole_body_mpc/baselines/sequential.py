"""W1 baseline: reposition the Scout, then reach with the arm alone.

research/whole_body_mpc §15 (W1) / WE2. The deterministic comparator for the
unified MPC (W3/W4): it may only use the base *before* the arm moves.

  1. ``choose_base_pose`` — candidate base poses facing the goal on rings of
     ``reach_m`` around it, plus the current pose. Each is scored by batched
     damped-least-squares IK of the TCP position (arm frame, joint limits with a
     margin) and, with a ``distance_fn``, by arm + base sphere clearance at the
     IK solution. Among feasible poses the one with the least travel wins, so
     the base does not move when the goal is already reachable.
  2. Base phase — turn / drive / turn unicycle controller to that pose with
     the arm frozen; every command goes through the same safety filter as the
     MPC.
  3. Arm phase — arm-only MPPI (the W0 controller, base frozen) from there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..costs.terms import DistanceFn, WholeBodyCost
from ..dynamics.scout import planar_T
from ..dynamics.whole_body import WholeBodyModel
from ..safety.projection import SafetyFilter
from ..sim import SimLog, run_closed_loop
from ..solvers.mppi import MPPI, MPPIConfig


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class SequentialConfig:
    reach_m: Tuple[float, ...] = (0.45, 0.55, 0.65)           # goal distance from base_link (xy)
    approach_offsets: Tuple[float, ...] = (0.0, -0.4, 0.4, -0.8, 0.8)   # rad around the straight approach
    yaw_weight: float = 0.3                                   # m of travel per rad of turning
    ik_iters: int = 80
    ik_damping: float = 0.05
    ik_tol_m: float = 0.005
    joint_margin: float = 0.05                                # rad inside the joint limits
    clearance_m: float = 0.03                                 # required at the IK solution
    pos_tol_m: float = 0.03
    yaw_tol: float = 0.05
    k_v: float = 1.0                                          # v = k_v ρ
    k_w: float = 2.0                                          # ω = k_w α
    face_tol: float = 0.15                                    # drive only once heading error < this


@dataclass
class BasePlan:
    pose: np.ndarray            # (3,) target base pose
    q: np.ndarray               # (6,) IK arm configuration there
    travel: float               # score: distance + yaw_weight · |Δθ|
    ik_error_m: float


def solve_position_ik(model: WholeBodyModel, bases: np.ndarray, goal_p: np.ndarray,
                      q0: np.ndarray, cfg: SequentialConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Batched DLS IK of the TCP position. bases (N, 3) -> q (N, 6), error (N,)."""
    kin = model.kin
    lo, hi = kin.lower + cfg.joint_margin, kin.upper - cfg.joint_margin
    T_inv = np.linalg.inv(planar_T(bases))                                  # base_T_world
    target = (T_inv @ np.r_[goal_p, 1.0])[:, :3]
    q = np.broadcast_to(np.clip(q0, lo, hi), (len(bases), 6)).copy()
    lam2 = cfg.ik_damping ** 2
    for _ in range(cfg.ik_iters):
        e = target - kin.tcp(q)[:, :3, 3]
        Jv = kin.jacobian(q)[:, :3, :]
        JJt = Jv @ np.swapaxes(Jv, 1, 2) + lam2 * np.eye(3)
        dq = (np.swapaxes(Jv, 1, 2) @ np.linalg.solve(JJt, e[..., None]))[..., 0]
        q = np.clip(q + dq, lo, hi)
    err = np.linalg.norm(target - kin.tcp(q)[:, :3, 3], axis=1)
    return q, err


def choose_base_pose(model: WholeBodyModel, goal_p: np.ndarray, x0: np.ndarray,
                     distance_fn: Optional[DistanceFn] = None,
                     cfg: SequentialConfig = SequentialConfig()) -> Optional[BasePlan]:
    goal_p = np.asarray(goal_p, float)
    b0 = np.asarray(x0[:3], float)
    phi_c = np.arctan2(goal_p[1] - b0[1], goal_p[0] - b0[0])
    cands = [b0]
    for r in cfg.reach_m:
        for off in cfg.approach_offsets:
            phi = phi_c + off
            cands.append(np.r_[goal_p[0] - r * np.cos(phi), goal_p[1] - r * np.sin(phi), phi])
    bases = np.array(cands)
    q, err = solve_position_ik(model, bases, goal_p, np.asarray(x0[3:], float), cfg)
    ok = err < cfg.ik_tol_m
    if distance_fn is not None and ok.any():
        X = np.concatenate([bases, q], axis=1)
        C, r = model.collision_spheres(X)                                    # (N, S, 3)
        d, valid = distance_fn(C.reshape(-1, 3))
        clear = (d.reshape(C.shape[:-1]) - r).min(1)
        ok &= (clear > cfg.clearance_m) & valid.reshape(C.shape[:-1]).all(1)
    if not ok.any():
        return None
    travel = np.linalg.norm(bases[:, :2] - b0[:2], axis=1) + cfg.yaw_weight * np.abs(_wrap(bases[:, 2] - b0[2]))
    travel = np.where(ok, travel, np.inf)
    i = int(np.argmin(travel))
    return BasePlan(pose=bases[i], q=q[i], travel=float(travel[i]), ik_error_m=float(err[i]))


def base_command(b: np.ndarray, target: np.ndarray, model: WholeBodyModel,
                 cfg: SequentialConfig) -> Tuple[np.ndarray, bool]:
    """Turn / drive / turn. Returns ((v, ω), reached)."""
    d = target[:2] - b[:2]
    rho = float(np.hypot(*d))
    if rho > cfg.pos_tol_m:
        alpha = float(_wrap(np.arctan2(d[1], d[0]) - b[2]))
        v = cfg.k_v * rho if abs(alpha) < cfg.face_tol else 0.0
        w = cfg.k_w * alpha
    else:
        dth = float(_wrap(target[2] - b[2]))
        if abs(dth) < cfg.yaw_tol:
            return np.zeros(2), True
        v, w = 0.0, cfg.k_w * dth
    vw = np.clip([v, w], model.u_low[:2], model.u_high[:2])
    return vw, False


@dataclass
class SequentialResult:
    log: SimLog
    plan: Optional[BasePlan]
    switch_step: int            # first arm-phase step (len of the base phase)


def run_sequential(x0: np.ndarray, cost: WholeBodyCost, safety: SafetyFilter, steps: int = 150,
                   seed: int = 0, cfg: SequentialConfig = SequentialConfig(),
                   mppi_cfg: Optional[MPPIConfig] = None, tol_m: float = 0.01) -> SequentialResult:
    m = cost.model
    x = np.asarray(x0, float).copy()
    plan = choose_base_pose(m, cost.goal.p, x, cost.distance_fn, cfg)
    log = SimLog(X=[x.copy()])
    u_prev = np.zeros(8)
    if plan is not None:
        for _ in range(steps):
            vw, reached = base_command(x[:3], plan.pose, m, cfg)
            if reached and np.abs(u_prev[:2]).max() < 1e-6:
                break
            rep = safety.project(x, np.r_[vw, np.zeros(6)], u_prev)
            x = m.rollout(x, rep.u[None, None])[0, 1]
            u_prev = rep.u
            log.X.append(x.copy()); log.U.append(rep.u); log.reasons.append(rep.reason)
            log.min_clearance.append(rep.min_clearance)
    switch = len(log.U)
    cfg_arm = mppi_cfg or MPPIConfig(seed=seed)
    cfg_arm.arm_only = True
    arm = run_closed_loop(x, MPPI(m, cfg_arm), cost, safety, steps=max(steps - switch, 0),
                          freeze_base=True, tol_m=tol_m)
    log.X += arm.X[1:]; log.U += arm.U; log.reasons += arm.reasons
    log.min_clearance += arm.min_clearance
    return SequentialResult(log=log, plan=plan, switch_step=switch)
