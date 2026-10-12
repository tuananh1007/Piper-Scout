"""Next-best-view selection (P4.2) and the ablation policies (P4.4).

Per candidate view the planner computes four gains:

  vis      expected target pixels with a depth return (the stem joint's
           surface facing the camera, in the image, surviving the sensor's
           thin-structure dropout, through the occupancy grid)
  fisher   expected entropy drop of the stem joint (½ log det ratio)
  entropy  expected unknown voxels of the approach corridor cleared (those
           the corridor requirement counts: not next to a surface), each
           reached through the occupancy grid and with a depth return behind
           it (``OccupancyGrid.voxel_gain``); without a corridor, unknown
           space Gaussian-weighted around the target
  twin     expected pixels on the plant_twin contact point (on the robot,
           while touching)

and normalises each by its best candidate. The policies:

  fixed_ring  the next point of a ring of views around the target (radius
              0.3 m, ±75° about the approach line), nearest reachable candidate
  random      a random reachable candidate
  geometry    classic NBV: unknown space observed, unweighted over the grid
  target      target-only (JEPA-only): target pixels
  fused       Σ weight·need·gain: α·n_JEPA·vis + β·(n_joint·fisher +
              n_corridor·entropy) + γ·n_twin·twin, where the need n of a
              required term is how far it is from being met,
              (U − 0.5)/0.5 clipped to [0, 1], plus 0.05·U so a met term
              still breaks ties; a term that is not required has n = 0.5·U.
              Weighting by U itself kept spending views on terms already met
              (the joint after one good view) while the corridor stayed
              unknown

each minus ``lambda_t`` per second of motion to reach the view and
``lambda_turn`` per radian the TCP approach axis turns (the whole-body MPC
closes large axis changes slowly or stalls, P3B.7 and the fake-robot check).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .candidates import Candidates, axis_turn, move_time
from .grid import OccupancyGrid
from .stem_joint import StemJointEstimator
from .uncertainty import Uncertainty

POLICIES = ("fixed_ring", "random", "geometry", "target", "fused")


@dataclass
class ViewGains:
    vis: np.ndarray
    fisher: np.ndarray
    entropy: np.ndarray
    entropy_all: np.ndarray
    twin: np.ndarray
    time: np.ndarray


def need_weight(U: float, required: bool) -> float:
    """How much a term still matters for choosing the view (see ``fused``)."""
    if not required:
        return 0.5 * U
    return float(np.clip((U - 0.5) / 0.5, 0.0, 1.0)) + 0.05 * U


def _norm(g: np.ndarray) -> np.ndarray:
    m = float(np.max(g)) if len(g) else 0.0
    return g / m if m > 1e-12 else np.zeros_like(g)


@dataclass
class ViewPlanner:
    model: object
    K: np.ndarray
    image_hw: tuple
    near_m: float = 0.07
    far_m: float = 1.0
    lambda_t: float = 0.02
    lambda_turn: float = 0.15
    p_unknown: float = 0.02       # occupancy of an unknown 1 cm voxel: the canopy's fill fraction
    task_sigma_m: float = 0.06
    ring_radius_m: float = 0.3
    ring_deg: tuple = (-75.0, -45.0, -15.0, 15.0, 45.0, 75.0)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    ring_done: list = field(default_factory=list)
    corridor_radius_m: float = 0.03

    def gains(self, cand: Candidates, x_now: np.ndarray, est: StemJointEstimator, grid: Optional[OccupancyGrid],
              corridor=None, twin_point: Optional[np.ndarray] = None, need=("vis", "fisher", "entropy")) -> ViewGains:
        M = len(cand.X)
        vis, fisher, ent, ent_all, twin = (np.zeros(M) for _ in range(5))
        ignore = est.target_exclusion(0.009)
        p_unk = self.p_unknown
        trans = None if grid is None else (
            lambda o, p: grid.transmittance(o, p, p_unk, stop_short_m=0.0, ignore=ignore))
        I_now = est.information()
        w_task, cvox = None, None
        if grid is not None and "entropy" in need:
            if corridor is not None:
                from .grid import corridor_mask  # noqa: PLC0415
                cvox = corridor_mask(grid, *corridor, radius_m=self.corridor_radius_m,
                                     exclude=est.target_exclusion(0.01)) & ~grid.skin() & grid.unknown
            else:
                w_task = grid.task_weight(est.cyl.p, self.task_sigma_m)
        for i in range(M):
            T = cand.T[i]
            if "vis" in need or "fisher" in need:
                I_v, px = est.predicted_view(T, self.K, self.image_hw, self.near_m, self.far_m, trans)
                vis[i] = px
                fisher[i] = est.fisher_gain(I_v, I_now) if "fisher" in need else 0.0
            if grid is not None and "entropy" in need:
                ent[i] = (grid.voxel_gain(T, self.K, self.image_hw, cvox, p_unk, self.near_m, self.far_m,
                                          ignore=ignore) if cvox is not None else
                          grid.entropy_gain(T, self.K, self.image_hw, w_task, max_range_m=self.far_m, p_unknown=p_unk))
            if grid is not None and "entropy_all" in need:
                ent_all[i] = grid.entropy_gain(T, self.K, self.image_hw, None, max_range_m=self.far_m,
                                               p_unknown=p_unk)
            if twin_point is not None and "twin" in need:
                twin[i] = self._point_visible(T, twin_point, grid, p_unk)
        return ViewGains(vis, fisher, ent, ent_all, twin, move_time(self.model, x_now, cand.X))

    def _point_visible(self, T, p, grid, p_unknown=None) -> float:
        pc = (np.asarray(p) - T[:3, 3]) @ T[:3, :3]
        if pc[2] < self.near_m or pc[2] > self.far_m:
            return 0.0
        u = self.K[0, 0] * pc[0] / pc[2] + self.K[0, 2]
        v = self.K[1, 1] * pc[1] / pc[2] + self.K[1, 2]
        if not (0 <= u < self.image_hw[1] and 0 <= v < self.image_hw[0]):
            return 0.0
        if grid is None:
            return 1.0
        return float(grid.transmittance(T[:3, 3], np.asarray(p), p_unknown, stop_short_m=0.02))

    def choose(self, policy: str, cand: Candidates, x_now: np.ndarray, est: StemJointEstimator,
               grid: Optional[OccupancyGrid], u: Uncertainty, corridor=None, approach_axis=None,
               twin_point=None) -> Optional[Dict[str, object]]:
        """Index of the chosen candidate and its gains, or None."""
        M = len(cand.X)
        if M == 0:
            return None
        t = move_time(self.model, x_now, cand.X)
        turn = axis_turn(self.model, x_now, cand.X)
        if policy == "random":
            i = int(self.rng.integers(M))
            return {"index": i, "score": 0.0, "time": float(t[i])}
        if policy == "fixed_ring":
            return self._ring(cand, est.cyl.p, approach_axis, t)
        need = {"geometry": ("entropy_all",), "target": ("vis",),
                "fused": ("vis", "fisher", "entropy") + (("twin",) if u.twin > 0 else ())}[policy]
        g = self.gains(cand, x_now, est, grid, corridor, twin_point, need)
        if policy == "geometry":
            score = _norm(g.entropy_all)
        elif policy == "target":
            score = _norm(g.vis)
        else:
            w = u.weights
            n = {k: need_weight(getattr(u, k), k in u.required) for k in ("jepa", "joint", "corridor", "twin")}
            score = (w.alpha * n["jepa"] * _norm(g.vis)
                     + w.beta * (n["joint"] * _norm(g.fisher) + n["corridor"] * _norm(g.entropy))
                     + w.gamma * n["twin"] * _norm(g.twin))
        score = score - self.lambda_t * t - self.lambda_turn * turn
        i = int(np.argmax(score))
        return {"index": i, "score": float(score[i]), "time": float(t[i]), "turn_deg": float(np.degrees(turn[i])),
                "vis_px": float(g.vis[i]),
                "fisher": float(g.fisher[i]), "entropy": float(g.entropy[i]), "entropy_all": float(g.entropy_all[i])}

    def _ring(self, cand: Candidates, target: np.ndarray, approach_axis, t: np.ndarray):
        a = np.asarray(approach_axis if approach_axis is not None else [1.0, 0.0, 0.0], float)
        a = np.r_[a[:2], 0.0] / max(np.linalg.norm(a[:2]), 1e-9)
        side = np.cross([0.0, 0.0, 1.0], a)
        cams = cand.T[:, :3, 3]
        for k, deg in enumerate(self.ring_deg):
            if k in self.ring_done:
                continue
            th = np.radians(deg)
            ring_pt = target + self.ring_radius_m * (-np.cos(th) * a + np.sin(th) * side)
            dist = np.linalg.norm(cams - ring_pt, axis=1)
            self.ring_done.append(k)
            if dist.min() < 0.15:
                i = int(np.argmin(dist))
                return {"index": i, "score": float(-dist[i]), "time": float(t[i]), "ring": deg}
        return None
