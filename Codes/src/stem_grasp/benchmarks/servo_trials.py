#!/usr/bin/env python3
"""Scripted final-approach trials in simulation: IBVS vs MPPI visual servo (P2.3.1).

The hardware-free precursor of the Phase 2B benchmark (ROADMAP P2.3: 50
scripted approach trials, baseline vs MPPI-VS; exit gate: >= 20 % success-rate
gain, >= 30 % lower maximum force). Each trial starts where the whole-body MPC
hands over (the pre-grasp, within the pipeline's 2 cm / 15 deg handoff
tolerance) and runs stem_grasp's own servo-phase logic in a kinematic loop:

  * measurement: the stem mask rendered through the *true* eye-in-hand camera
    (640 x 480, f = 380 px, as the hardware-free checks), the stem feature from
    ``servo_geometry.stem_feature_uv`` at the target's row, the desired pixel
    where the gripper axis crosses the target depth (``servo_geometry.desired_uv``);
    the 3-D target from a grasp-point estimate (bias + noise, refreshed at 5 Hz
    while the camera is farther than the D405's 7 cm, frozen below);
  * approach: ``approach.IterativeApproach`` with the pipeline's defaults;
  * IBVS (``core.FullAdaptiveServoController``, baseline): its camera
    translation plus the approach speed, executed as moveit_servo does (twist
    in the camera frame of the *assumed* hand-eye calibration, damped inverse
    of the arm Jacobian, joint speed limit);
  * MPPI (``visual_servo.MppiVisualServo``, as ``servo_controller: mppi``),
    optionally with a semantic field of the obstacles (``mppi+field``: the
    neighbouring stem; the target stem released as in stem_grasp);
  * force: fingers (open 6 cm, along link6 y) and the gripper's palm against a
    compliant stem (``--stiffness`` N/m); the force is fed back to both
    controllers like the joint-effort estimate (its noise added) and above
    ``max_force_n`` the trial ends (the pipeline's force gate).

Randomised per trial (same draws for every controller): the start offset and
axis error, the hand-eye calibration error, the grasp-point estimate bias,
stem sway (amplitude, frequency), mask jitter and, in half the trials, a
neighbouring stem 5-9 cm beside the target.

Success: the approach ends at the grasp point (AT_GRASP) with the true stem
centred between the fingers (lateral miss <= ``--miss``, along the axis
within 1.5 cm), no collision with the neighbour and the force limit never hit.

    cd Codes/src
    PYTHONPATH=stem_grasp:scout_piper_whole_body_mpc:scout_piper_scene_repr/python \\
        python3 stem_grasp/benchmarks/servo_trials.py --trials 50 --out servo_trials.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from scout_piper_whole_body_mpc.dynamics.piper import PiperKinematics
from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel
from stem_grasp.approach import ApproachConfig, IterativeApproach
from stem_grasp.core import FullAdaptiveServoController
from stem_grasp.servo_geometry import desired_uv, project, stem_feature_uv

K = np.array([[380.0, 0.0, 320.0], [0.0, 380.0, 240.0], [0.0, 0.0, 1.0]])
HW = (480, 640)
STEM_R = 0.004
OPEN_W = 0.06               # gripper opening during the approach (grasp_open_width_m)
FINGER_T = 0.008            # finger thickness
PALM_S = -0.035             # stem this far behind the TCP along the axis touches the palm
GRASP_Z = 0.40


def rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    x, y, z = axis
    c, s, C = np.cos(ang), np.sin(ang), 1 - np.cos(ang)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def T_flange_cam(d_xyz=(0.0, 0.0, 0.0), d_rot=np.eye(3)):
    """link6 -> camera optical frame (optical z along link6 z, 5 cm out), with an error."""
    T = np.eye(4)
    T[:3, :3] = d_rot @ np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = np.array([0.05, 0.0, 0.05]) + np.asarray(d_xyz)
    return T


def ik(kin: PiperKinematics, p: np.ndarray, axis: np.ndarray, q: np.ndarray, iters: int = 300) -> np.ndarray:
    """Joints putting the TCP at ``p`` with its z along ``axis`` (damped least squares)."""
    q = np.asarray(q, float).copy()
    for _ in range(iters):
        T = kin.tcp(q)
        e = np.r_[p - T[:3, 3], np.cross(T[:3, 2], axis)]
        if np.linalg.norm(e) < 1e-6:
            break
        J = kin.jacobian(q)
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), kin.lower + 0.16, kin.upper - 0.16)
    return q


@dataclass
class Trial:
    seed: int
    start_offset: np.ndarray            # TCP lateral offset at the start (m, base frame)
    axis_err_rot: np.ndarray            # (3, 3) start axis error
    calib_xyz: np.ndarray               # hand-eye translation error (m)
    calib_rot: np.ndarray               # hand-eye rotation error
    est_bias: np.ndarray                # grasp-point estimate bias (m)
    sway_amp: float
    sway_hz: float
    sway_phase: float
    neighbour: Optional[np.ndarray] = None   # (x, y) of a second stem
    rng: np.random.Generator = field(default=None)

    @classmethod
    def draw(cls, seed: int) -> "Trial":
        r = np.random.default_rng(seed)
        off = r.normal(size=3)
        off[0] = 0.0                                          # lateral to the approach (base x)
        off *= r.uniform(0.0, 0.02) / max(np.linalg.norm(off), 1e-9)
        ax = r.normal(size=3)
        ax[0] = 0.0
        err = rot(ax if np.linalg.norm(ax) > 1e-9 else [0, 0, 1.0], np.radians(r.uniform(0.0, 10.0)))
        crot = rot(r.normal(size=3), np.radians(abs(r.normal(0.0, 1.5))))
        nb = None
        if r.uniform() < 0.5:
            nb = np.array([0.0, r.choice([-1.0, 1.0]) * r.uniform(0.05, 0.09)]) + [r.uniform(-0.03, 0.03), 0.0]
        return cls(seed, off, err, r.normal(0.0, 0.004, 3), crot, r.normal(0.0, 0.003, 3),
                   float(r.uniform(0.0, 0.006)), float(r.uniform(0.2, 0.8)), float(r.uniform(0, 2 * np.pi)), nb,
                   np.random.default_rng(seed + 10_000))


class Scene:
    """Stem (sway along base y), optional neighbour, renderer and contact model."""

    def __init__(self, trial: Trial, stem_xy=(0.62, 0.0)):
        self.t = trial
        self.stem0 = np.array([stem_xy[0], stem_xy[1]])
        zz = np.linspace(0.25, 0.60, 300)
        ang = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        self._ring = np.stack([np.repeat(STEM_R * np.cos(ang)[None], len(zz), 0).ravel(),
                               np.repeat(STEM_R * np.sin(ang)[None], len(zz), 0).ravel(),
                               np.repeat(zz[:, None], len(ang), 1).ravel()], 1)

    def stem_xy(self, time_s: float) -> np.ndarray:
        s = self.t.sway_amp * np.sin(2 * np.pi * self.t.sway_hz * time_s + self.t.sway_phase)
        return self.stem0 + [0.0, s]

    def grasp_point(self, time_s: float) -> np.ndarray:
        return np.r_[self.stem_xy(time_s), GRASP_Z]

    def stem_offset(self, T_tcp: np.ndarray, time_s: float) -> np.ndarray:
        """Stem axis point at the TCP's height minus the TCP (the stem is vertical)."""
        return np.r_[self.stem_xy(time_s), T_tcp[2, 3]] - T_tcp[:3, 3]

    def mask(self, T_base_cam: np.ndarray, time_s: float) -> np.ndarray:
        pts = self._ring + np.r_[self.stem_xy(time_s), 0.0]
        R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
        pc = (pts - t) @ R
        pc = pc[pc[:, 2] > 0.02]
        jit = self.t.rng.normal(0.0, 1.0, 2)
        u = np.round(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2] + jit[0]).astype(int)
        v = np.round(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2] + jit[1]).astype(int)
        m = np.zeros(HW, bool)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                ok = (u + du >= 0) & (u + du < HW[1]) & (v + dv >= 0) & (v + dv < HW[0])
                m[v[ok] + dv, u[ok] + du] = True
        return m

    def force(self, T_tcp: np.ndarray, time_s: float, stiffness: float) -> float:
        """Spring force of the fingers / palm on the stem (N)."""
        a, y = T_tcp[:3, 2], T_tcp[:3, 1]
        d = self.stem_offset(T_tcp, time_s)
        s = float(d @ a)                                       # stem ahead of the TCP along the axis
        lat = d - s * a
        dy, dx = float(lat @ y), float(np.linalg.norm(lat - (lat @ y) * y))
        pen = 0.0
        if -0.045 < s < 0.012 and dx < 0.01:                   # finger reach along the axis
            pen = max(pen, STEM_R + FINGER_T / 2 - abs(abs(dy) - OPEN_W / 2))
        if s < PALM_S and abs(dy) < OPEN_W / 2:                # palm
            pen = max(pen, PALM_S - s)
        return stiffness * max(pen, 0.0)

    def neighbour_clearance(self, model: WholeBodyModel, q: np.ndarray, T_tcp: np.ndarray) -> float:
        if self.t.neighbour is None:
            return np.inf
        c = self.stem0 + self.t.neighbour
        C, r = model.collision_spheres(np.r_[0.0, 0.0, 0.0, q])
        n = len(model.arm_spheres)
        d = np.linalg.norm(C[:n, :2] - c, axis=1) - r[:n] - STEM_R
        tips = [T_tcp[:3, 3] + sgn * OPEN_W / 2 * T_tcp[:3, 1] for sgn in (-1, 1)]
        d_tips = [np.linalg.norm(p[:2] - c) - FINGER_T / 2 - STEM_R for p in tips]
        return float(min(d.min(), min(d_tips)))

    def neighbour_distance_fn(self):
        """Semantic field of the neighbour (base frame), for ``mppi+field``."""
        if self.t.neighbour is None:
            return None
        c = self.stem0 + self.t.neighbour

        def fn(points):
            p = np.asarray(points, float).reshape(-1, 3)
            return np.linalg.norm(p[:, :2] - c, axis=1) - STEM_R - 0.005, np.ones(len(p), bool)
        return fn


def run_trial(trial: Trial, controller: str, samples: int = 256, dt: float = 0.05, timeout_s: float = 60.0,
              stiffness: float = 150.0, max_force_n: float = 2.0, force_noise_n: float = 0.1,
              contact_threshold_n: float = 0.5, miss_m: float = 0.01, trace: Optional[list] = None,
              align_window_s: float = 0.0) -> dict:
    model = WholeBodyModel()
    kin = model.kin
    scene = Scene(trial)
    g0 = scene.grasp_point(0.0)
    axis0 = trial.axis_err_rot @ np.array([1.0, 0.0, 0.0])
    q = ik(kin, g0 - 0.12 * np.array([1.0, 0.0, 0.0]) + trial.start_offset, axis0,
           np.array([0.0, 1.2, -1.0, 0.0, 0.5, 0.0]))
    T_fc = T_flange_cam()                                      # the URDF's (assumed) hand-eye transform
    T_fc_true = T_flange_cam(trial.calib_xyz, trial.calib_rot)   # the real mount differs
    approach = IterativeApproach(ApproachConfig(contact_force_n=contact_threshold_n, align_window_s=align_window_s),
                                 start_t=0.0)
    ibvs = FullAdaptiveServoController(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                                       lambda_0=0.8, lambda_inf=0.5, rho=0.3)
    mppi = None
    if controller.startswith("mppi"):
        from scout_piper_whole_body_mpc.visual_servo import (MppiVisualServo, ServoTarget,  # noqa: PLC0415
                                                              VisualServoConfig, VisualServoWeights)
        mppi = MppiVisualServo(VisualServoConfig(samples=samples, seed=trial.seed),
                               VisualServoWeights(contact_force_n=contact_threshold_n, max_force_n=max_force_n))
        dist_fn = scene.neighbour_distance_fn() if controller == "mppi+field" else None
    est = g0 + trial.est_bias
    t_est, cam_prev = -1.0, None
    f_max, clear_min, outcome, reason = 0.0, np.inf, None, ""
    t0 = time.perf_counter()
    k = 0
    for k in range(int(timeout_s / dt)):
        now = k * dt
        F6 = kin.link_frames(q)[6]
        T_tcp = kin.tcp(q)
        T_cam_true, T_cam = F6 @ T_fc_true, F6 @ T_fc
        f = scene.force(T_tcp, now, stiffness)
        f_max = max(f_max, f)
        clear_min = min(clear_min, scene.neighbour_clearance(model, q, T_tcp))
        if clear_min < 0.0:
            outcome, reason = "collision", "touched the neighbouring stem"
            break
        if f > max_force_n:
            outcome, reason = "force", f"force {f:.2f} N > {max_force_n} N"
            break
        f_meas = max(f + trial.rng.normal(0.0, force_noise_n), 0.0)
        # grasp-point estimate: refreshed at 5 Hz while depth is usable (camera farther than 7 cm)
        truth = scene.grasp_point(now)
        if now - t_est >= 0.2 and (truth - T_cam_true[:3, 3]) @ T_cam_true[:3, 2] > 0.07:
            est, t_est = truth + trial.est_bias + trial.rng.normal(0.0, 0.001, 3), now
        target_cam = T_cam[:3, :3].T @ (est - T_cam[:3, 3])
        tcp_cam = T_cam[:3, :3].T @ (T_tcp[:3, 3] - T_cam[:3, 3])
        approach_cam = T_cam[:3, :3].T @ T_tcp[:3, 2]
        if target_cam[2] < 0.05:
            outcome, reason = "abort", "target behind the camera"
            break
        mask = scene.mask(T_cam_true, now)
        uv_t = project(K, target_cam)
        desired = desired_uv(K, tcp_cam, approach_cam, float(target_cam[2]))
        desired = np.array([K[0, 2], K[1, 2]]) if desired is None else desired
        raw = stem_feature_uv(mask, uv_t[1], 12) if uv_t is not None else None
        distance = float((est - T_tcp[:3, 3]) @ T_tcp[:3, 2])
        if trace is not None:
            trace.append({"t": now, "err_px": None if raw is None else float(np.linalg.norm(raw - desired)),
                          "distance": distance, "phase": approach.phase, "f": f, "f_meas": f_meas})
        if raw is None:
            st = approach.update(now, float("inf"), distance, 0, f_meas)
            qd = np.zeros(6)
        else:
            err = float(np.linalg.norm(raw - desired))
            st = approach.update(now, err, distance, int(mask.sum()), f_meas, error_uv=raw - desired)
            if st.phase in ("done", "abort"):
                break
            if mppi is None:
                cam_vel = None
                if cam_prev is not None:
                    cam_vel = T_cam[:3, :3].T @ (T_cam[:3, 3] - cam_prev) / dt
                vel, _ = ibvs.step(raw_uv=raw, desired_uv=desired, depth_z=float(target_cam[2]), force_n=f_meas,
                                   commanded_vel=None, dt=dt, camera_vel=cam_vel)
                vel = vel + st.speed * approach_cam
                v_base = T_cam[:3, :3] @ vel                   # moveit_servo: twist in the (assumed) camera frame
                J = kin.jacobian(q)
                qd = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[v_base, 0.0, 0.0, 0.0])
                qd = qd * min(1.0, 0.5 / max(np.abs(qd).max(), 1e-9))
            else:
                tg = ServoTarget(uv_meas=raw, K=K, image_hw=HW, T_flange_cam=T_fc, p_base=est, force_n=f_meas,
                                 desired_distance_m=distance - st.speed * mppi.cfg.horizon * mppi.cfg.dt)
                qd, diag = mppi.step(q, tg, distance_fn=dist_fn)
                if trace is not None:
                    trace[-1].update(plan=diag.get("plan_terms"), nominal=diag.get("nominal_terms"),
                                     safety=diag.get("safety"), dd=tg.desired_distance_m, qd=np.round(qd, 3).tolist())
        if st.phase in ("done", "abort"):
            break
        cam_prev = T_cam[:3, 3]
        q = np.clip(q + dt * qd, kin.lower, kin.upper)
    T_tcp = kin.tcp(q)
    d = scene.stem_offset(T_tcp, k * dt)
    s = float(d @ T_tcp[:3, 2])
    miss = float(np.linalg.norm(d - s * T_tcp[:3, 2]))
    if outcome is None:
        outcome = "done" if approach.phase == "done" else ("abort" if approach.phase == "abort" else "timeout")
        reason = approach.reason
    success = outcome == "done" and miss <= miss_m and abs(s) <= 0.015
    return {"seed": trial.seed, "controller": controller, "success": bool(success), "outcome": outcome,
            "reason": reason, "time_s": round(k * dt, 2), "steps": approach.steps, "miss_mm": round(1000 * miss, 1),
            "axial_mm": round(1000 * s, 1), "max_force_n": round(f_max, 3),
            "neighbour": trial.neighbour is not None,
            "min_clearance_m": None if not np.isfinite(clear_min) else round(clear_min, 3),
            "wall_s": round(time.perf_counter() - t0, 1)}


def summarise(rows):
    out = {}
    for c in sorted({r["controller"] for r in rows}):
        rs = [r for r in rows if r["controller"] == c]
        out[c] = {"trials": len(rs), "success": round(float(np.mean([r["success"] for r in rs])), 3),
                  "max_force_mean_n": round(float(np.mean([r["max_force_n"] for r in rs])), 3),
                  "max_force_p90_n": round(float(np.percentile([r["max_force_n"] for r in rs], 90)), 3),
                  "outcomes": {o: sum(r["outcome"] == o for r in rs) for o in sorted({r["outcome"] for r in rs})},
                  "time_s_median": round(float(np.median([r["time_s"] for r in rs])), 1)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--controllers", default="ibvs,mppi,mppi+field")
    ap.add_argument("--samples", type=int, default=256, help="MPPI samples")
    ap.add_argument("--stiffness", type=float, default=150.0, help="N/m of the stem")
    ap.add_argument("--miss", type=float, default=0.01, help="m: largest lateral miss counted as a grasp")
    ap.add_argument("--align-window", type=float, default=0.0,
                    help="s: approach alignment on the mean error over this window (stem_grasp approach_align_window_sec)")
    ap.add_argument("--out", default="", help="JSON lines per trial")
    a = ap.parse_args()
    rows = []
    out = open(a.out, "w", encoding="utf-8") if a.out else None
    for i in range(a.first_seed, a.first_seed + a.trials):
        trial = Trial.draw(i)
        for c in a.controllers.split(","):
            r = run_trial(trial, c, samples=a.samples, stiffness=a.stiffness, miss_m=a.miss,
                          align_window_s=a.align_window)
            rows.append(r)
            line = json.dumps(r)
            print(line, flush=True)
            if out:
                out.write(line + "\n")
                out.flush()
    s = summarise(rows)
    base = s.get("ibvs")
    for c, v in s.items():
        if base and c != "ibvs":
            v["success_gain_pts"] = round(100 * (v["success"] - base["success"]), 1)
            v["max_force_reduction"] = (round(1 - v["max_force_mean_n"] / base["max_force_mean_n"], 3)
                                        if base["max_force_mean_n"] > 0 else None)
    print(json.dumps({"summary": s}, indent=1), flush=True)
    if out:
        out.write(json.dumps({"summary": s}) + "\n")
        out.close()


if __name__ == "__main__":
    main()
