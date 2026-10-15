"""Reachability filter on every proposed target, with plain-language reasons
(P5.1.4, P5.3.5).

Whatever proposes a target (the rule-based grounding or a VLM), the target
goes through this filter before anything moves. Checks, in order:

  no_depth      no 3-D point for the target ("I can't tell how far away it is")
  too_high /    the grasp point outside the heights the gripper reaches with a
  too_low       roughly horizontal approach from the Scout's top plate (sampled
                once from the arm's kinematics)
  too_thick     a stem wider than the gripper opening allows
  crowded       another hard obstacle (stem, stake, pot) closer to the grasp
                point than half the open gripper plus a margin (the target's
                own stem released by scout_piper_scene_repr ``exclude_target``)
  too_far       no whole-body pose within ``max_base_move_m`` of the robot puts
                the TCP on the pre-grasp point (0.12 m before the target on the
                line from the robot, approach axis along that line)
  no_pose       such poses exist but all of them collide with the mapped
                obstacles
and an advisory that does not reject:
  occluded      the line of sight from the camera to the target crosses leaves
                or other obstacles ("It is behind a leaf; I will look from
                another side first"), which hands over to active perception.

The semantic field is optional: without it only the geometric checks run and
the verdict says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ReachVerdict:
    ok: bool
    code: str
    message: str
    advisories: List[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "code": self.code, "message": self.message, "advisories": self.advisories,
                "details": self.details}


@dataclass
class ReachConfig:
    pregrasp_offset_m: float = 0.12
    gripper_open_m: float = 0.07          # Piper gripper stroke
    crowd_margin_m: float = 0.01
    approach_tilt_deg: float = 30.0       # TCP z axis within this of horizontal
    clearance_m: float = 0.02
    max_base_move_m: float = 2.0
    samples: int = 30000
    exclude_radius_m: float = 0.10


class ReachabilityChecker:
    def __init__(self, model, cfg: Optional[ReachConfig] = None, rng: Optional[np.random.Generator] = None):
        self.m = model
        self.cfg = cfg or ReachConfig()
        self.rng = rng or np.random.default_rng(0)
        self._arm = None

    # ------------------------------------------------------------ arm table
    def _arm_samples(self):
        """Arm configurations with a roughly horizontal TCP axis: (q, TCP in
        the base frame (N, 4, 4)), computed once."""
        if self._arm is None:
            from scout_piper_whole_body_mpc.costs.terms import JOINT_LIMIT_MARGIN  # noqa: PLC0415
            kin = self.m.kin
            q = self.rng.uniform(kin.lower + JOINT_LIMIT_MARGIN, kin.upper - JOINT_LIMIT_MARGIN,
                                 (self.cfg.samples, 6))
            X = np.concatenate([np.zeros((len(q), 3)), q], 1)
            T = self.m.tcp_world(X)
            tilt = np.degrees(np.arcsin(np.clip(np.abs(T[:, 2, 2]), 0, 1)))
            keep = tilt <= self.cfg.approach_tilt_deg
            self._arm = (q[keep], T[keep])
        return self._arm

    def height_range(self):
        _, T = self._arm_samples()
        z = T[:, 2, 3]
        return float(np.percentile(z, 1)), float(np.percentile(z, 99))

    # ---------------------------------------------------------------- check
    def check(self, p_world: Optional[np.ndarray], x_now: np.ndarray, distance_fn=None, snapshot=None,
              camera_pos: Optional[np.ndarray] = None, width_m: Optional[float] = None,
              target_class: str = "") -> ReachVerdict:
        cfg = self.cfg
        if p_world is None or not np.all(np.isfinite(p_world)):
            return ReachVerdict(False, "no_depth", "I can't tell how far away it is (no depth on it). "
                                "Let me look from closer first.")
        p = np.asarray(p_world, float)
        adv: List[str] = []
        a = np.r_[p[:2] - np.asarray(x_now[:2], float), 0.0]
        dist_xy = float(np.linalg.norm(a))
        a = a / max(dist_xy, 1e-9)
        pre = p - cfg.pregrasp_offset_m * a
        z_lo, z_hi = self.height_range()
        if pre[2] > z_hi + 0.02:
            return ReachVerdict(False, "too_high", f"It is too high for me to reach ({p[2]:.2f} m; "
                                f"I reach up to about {z_hi:.2f} m).", details={"z": p[2], "z_max": z_hi})
        if pre[2] < z_lo - 0.02:
            return ReachVerdict(False, "too_low", f"It is too low for me to reach ({p[2]:.2f} m; "
                                f"I reach down to about {z_lo:.2f} m).", details={"z": p[2], "z_min": z_lo})
        if width_m is not None and target_class in ("stem", "branch") and width_m > 0.9 * cfg.gripper_open_m:
            return ReachVerdict(False, "too_thick", f"That stem is too thick for my gripper "
                                f"({100 * width_m:.1f} cm; the gripper opens to {100 * cfg.gripper_open_m:.0f} cm).",
                                details={"width_m": width_m})
        hard_fn = distance_fn
        if snapshot is not None:
            from scout_piper_scene_repr_py.field import exclude_target, snapshot_distance_fn  # noqa: PLC0415
            snap_x = exclude_target(snapshot, p, radius=cfg.exclude_radius_m)
            d_other = snapshot_distance_fn(snap_x, snapshot.stamp, np.inf)(p[None])[0][0]
            need = 0.5 * cfg.gripper_open_m + cfg.crowd_margin_m
            if np.isfinite(d_other) and d_other < need:
                return ReachVerdict(False, "crowded", f"It is too close to another stem ({100 * d_other:.0f} cm) "
                                    "to open the gripper around it safely.", details={"clearance_m": float(d_other)})
            if hard_fn is None:
                hard_fn = snapshot_distance_fn(snapshot, snapshot.stamp, np.inf)
            if camera_pos is not None and self._occluded(snapshot, np.asarray(camera_pos, float), p):
                adv.append("It is behind a leaf; I will look from another side first.")
        X, n_free = self._reach_poses(pre, a, x_now, hard_fn)
        if n_free == 0:
            return ReachVerdict(False, "too_far", f"It is too far away ({dist_xy:.1f} m); I may drive at most "
                                f"{cfg.max_base_move_m:.1f} m for a grasp.", advisories=adv,
                                details={"distance_m": dist_xy})
        if len(X) == 0:
            return ReachVerdict(False, "no_pose", "I can't find a way to reach it without hitting the plants.",
                                advisories=adv)
        msg = "I can reach it." + ("" if snapshot is not None or distance_fn is not None
                                   else " (Checked without the scene map: obstacles not known.)")
        return ReachVerdict(True, "ok", msg, advisories=adv,
                            details={"poses": int(len(X)), "pregrasp": np.round(pre, 4).tolist(),
                                     "approach_axis": np.round(a, 4).tolist()})

    def _reach_poses(self, pre, a, x_now, hard_fn):
        """(collision-free poses, number of poses within the drive range)."""
        q, T = self._arm_samples()
        sel = np.abs(T[:, 2, 3] - pre[2]) < 0.01
        q, T = q[sel], T[sel]
        if not len(q):
            return np.zeros((0, 9)), 0
        # base yaw so the TCP axis points along a; base position puts the TCP on pre
        th = np.arctan2(a[1], a[0]) - np.arctan2(T[:, 1, 2], T[:, 0, 2])
        c, s = np.cos(th), np.sin(th)
        X = np.zeros((len(q), 9))
        X[:, 2] = th
        X[:, 3:] = q
        X[:, 0] = pre[0] - (c * T[:, 0, 3] - s * T[:, 1, 3])
        X[:, 1] = pre[1] - (s * T[:, 0, 3] + c * T[:, 1, 3])
        X = X[np.linalg.norm(X[:, :2] - np.asarray(x_now[:2], float), axis=1) <= self.cfg.max_base_move_m]
        n = len(X)
        if hard_fn is not None and len(X):
            C, r = self.m.collision_spheres(X)
            d, _ = hard_fn(C.reshape(-1, 3))
            X = X[(np.asarray(d).reshape(C.shape[:-1]) - r).min(1) > self.cfg.clearance_m]
        return X, n

    @staticmethod
    def _occluded(snapshot, cam, p, clear_last_m=0.03) -> bool:
        from scout_piper_scene_repr_py.field import FieldSampler  # noqa: PLC0415
        smp = FieldSampler(snapshot)
        L = float(np.linalg.norm(p - cam))
        n = max(int(L / 0.005), 2)
        s = np.linspace(0.0, 1.0, n)[:, None]
        pts = cam + s * (p - cam)
        q = smp.query(pts, snapshot.stamp, np.inf)
        block = 0.5 * snapshot.voxel_size
        hit = q["hard"] < block
        if q["soft"] is not None:
            hit |= q["soft"] < block
        hit &= q["in_bounds"] & ((1.0 - s[:, 0]) * L > clear_last_m)
        return bool(hit.any())
