"""The three uncertainty terms of Phase 4 and their fusion.

U_total = α·U_JEPA + β·U_geometry + γ·U_twin   (ROADMAP.md Phase 4)

Every term is mapped to [0, 1] with 0.5 where its requirement is just met:

  U_joint     σ / (σ + σ_req): σ the stem joint's worst 1-σ lateral axis
              uncertainty (``StemJointEstimator.sigma``), σ_req 1 mm (2σ within
              the 2 mm the grasp tolerates)
  U_corridor  f / (f + f_req): f the unknown fraction of the approach corridor
              (the MPC's ``unknown_policy: no_entry`` cannot pass unknown
              space), f_req 5 %
  U_geometry  max(U_joint, U_corridor)
  U_JEPA      from /piper_jepa/target_state: 1 uninitialised or lost, 0.75
              occluded, 1 − confidence while tracking; in simulation (no
              tracker) 1 − min(1, visible target pixels / n_ref)
  U_twin      from /plant_twin/state: 1 − confidence while the gripper
              touches the plant (1 when the state is older than its limit), 0
              otherwise (deformation does not matter before contact)

U_total (normalised by the weights of the active terms) ranks the terms
when scoring views. Whether to look at all is decided per term: a view is
needed while any required term is above 0.5 (its requirement not met). A
weighted sum alone would let a well-met term hide an unmet one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


def ratio_map(value: float, requirement: float) -> float:
    value = max(float(value), 0.0)
    return value / (value + requirement) if requirement > 0 else float(value > 0)


def u_jepa_from_state(state: Optional[dict], now: Optional[float] = None, max_age_s: float = 1.0) -> float:
    if not state:
        return 1.0
    if now is not None and state.get("stamp") is not None and now - float(state["stamp"]) > max_age_s:
        return 1.0
    status = state.get("status", "uninitialized")
    if status in ("uninitialized", "lost"):
        return 1.0
    if status == "occluded":
        return 0.75
    return float(min(max(1.0 - float(state.get("confidence", 0.0)), 0.0), 1.0))


def u_jepa_from_pixels(target_px: float, n_ref: float = 150.0) -> float:
    return float(1.0 - min(1.0, max(target_px, 0.0) / n_ref))


def u_twin_from_state(state: Optional[dict], now: Optional[float] = None, max_age_s: float = 1.0) -> float:
    if not state or not (state.get("touching") or state.get("pulling")):
        return 0.0
    if now is not None and state.get("stamp") is not None and now - float(state["stamp"]) > max_age_s:
        return 1.0
    return float(min(max(1.0 - float(state.get("confidence", 0.0)), 0.0), 1.0))


@dataclass
class Weights:
    alpha: float = 0.3       # JEPA
    beta: float = 0.5        # geometry
    gamma: float = 0.2       # twin


@dataclass
class Uncertainty:
    joint: float
    corridor: float
    jepa: float
    twin: float
    sigma_m: float = float("nan")
    corridor_unknown: float = float("nan")
    weights: Weights = field(default_factory=Weights)
    required: Sequence[str] = ("joint", "corridor")

    @property
    def geometry(self) -> float:
        return max(self.joint, self.corridor)

    @property
    def total(self) -> float:
        w = self.weights
        num = w.alpha * self.jepa + w.beta * self.geometry + w.gamma * self.twin
        den = w.alpha + w.beta + (w.gamma if self.twin > 0 else 0.0)
        return num / den if den > 0 else 0.0

    def unmet(self) -> list:
        return [k for k in self.required if getattr(self, k) > 0.5]

    @property
    def needs_view(self) -> bool:
        return bool(self.unmet())

    def as_dict(self) -> Dict[str, object]:
        return {"u_total": round(self.total, 4), "u_jepa": round(self.jepa, 4), "u_geometry": round(self.geometry, 4),
                "u_joint": round(self.joint, 4), "u_corridor": round(self.corridor, 4), "u_twin": round(self.twin, 4),
                "axis_sigma_mm": None if self.sigma_m != self.sigma_m else round(1000 * self.sigma_m, 3),
                "corridor_unknown": None if self.corridor_unknown != self.corridor_unknown
                else round(self.corridor_unknown, 4),
                "needs_view": self.needs_view, "unmet": self.unmet()}


def assess(sigma_m: float, corridor_unknown: float, u_jepa: float, u_twin: float = 0.0,
           sigma_req_m: float = 0.001, corridor_req: float = 0.05, weights: Optional[Weights] = None,
           required: Sequence[str] = ("joint", "corridor")) -> Uncertainty:
    return Uncertainty(joint=ratio_map(sigma_m, sigma_req_m), corridor=ratio_map(corridor_unknown, corridor_req),
                       jepa=u_jepa, twin=u_twin, sigma_m=sigma_m, corridor_unknown=corridor_unknown,
                       weights=weights or Weights(), required=tuple(required))
