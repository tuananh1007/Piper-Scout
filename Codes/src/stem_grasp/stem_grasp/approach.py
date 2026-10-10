"""Iterative final approach from the pre-grasp pose to the stem (P0.4.13).

The ROS 1 approach state machine is not in this repository (only its
parameter names are), so this is a new design that keeps those names. It runs
on top of the image-based servo, which keeps the stem on the gripper approach
axis, and adds motion along that axis in steps:

    align --error <= align_tolerance_px for settle_s--> advance
      ^                                                    |
      +-- moved approach_step (step done), or error grew --+
          (step interrupted: resumed after re-aligning, not counted again)
    done:  TCP within approach_distance_tolerance of the grasp point along the
           axis, or contact force >= contact_threshold_n for contact_hold_s
           (one noisy sample of a force estimate is not a contact)
    abort: more than approach_max_steps steps, the TCP farther than
           approach_target_distance when the approach starts, the stem mask
           below approach_min_leaf_pixels for approach_mask_wait_sec, or
           approach_timeout_sec overall

The caller feeds one update per servo step (new mask) and adds
``speed * approach_axis`` to the servo's camera velocity. At AT_GRASP the
pipeline may close the gripper; ``GripperCloseMonitor`` decides when the
measured opening has settled on the stem, and ``RetreatMonitor`` tracks the
retreat after a release or a force-limit violation. ``servo_gate`` holds the
Phase 2B hard gates of the servo phases. This module holds no ROS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ApproachConfig:
    step_m: float = 0.05               # approach_step
    max_steps: int = 8                 # approach_max_steps
    settle_s: float = 0.4              # approach_settle_sec
    distance_tolerance_m: float = 0.01  # approach_distance_tolerance
    max_start_distance_m: float = 0.25  # approach_target_distance
    mask_wait_s: float = 2.0           # approach_mask_wait_sec
    min_mask_pixels: int = 60          # approach_min_leaf_pixels
    align_tolerance_px: float = 8.0    # approach_align_tolerance_px
    speed_mps: float = 0.02            # approach_speed_mps
    timeout_s: float = 60.0            # approach_timeout_sec
    contact_force_n: float = 0.15      # contact_threshold_n
    contact_hold_s: float = 0.1        # contact_hold_sec: the force must stay above it this long


@dataclass
class ApproachStatus:
    phase: str                         # "align", "advance", "done" or "abort"
    speed: float = 0.0                 # m/s along the gripper approach axis
    reason: str = ""


class IterativeApproach:
    """Decides, per servo step, whether to advance the gripper along its axis."""

    def __init__(self, cfg: ApproachConfig, start_t: float) -> None:
        self.cfg = cfg
        self.start_t = start_t
        self.phase = "align"
        self.steps = 0
        self.reason = ""
        self._aligned_since: Optional[float] = None
        self._step_goal_d: Optional[float] = None   # distance at which this step ends
        self._mask_ok_t = start_t
        self._contact_since: Optional[float] = None

    def _end(self, phase: str, reason: str) -> ApproachStatus:
        self.phase, self.reason = phase, reason
        return ApproachStatus(phase, 0.0, reason)

    def update(self, now: float, error_px: float, distance_m: float,
               mask_pixels: int, force_n: float = 0.0) -> ApproachStatus:
        """``distance_m``: grasp point minus TCP along the approach axis."""
        c = self.cfg
        if self.phase in ("done", "abort"):
            return ApproachStatus(self.phase, 0.0, self.reason)
        if self.steps == 0 and distance_m > c.max_start_distance_m:
            return self._end("abort", f"TCP {distance_m:.3f} m from the grasp point, more than "
                                      f"{c.max_start_distance_m} m")
        if force_n >= c.contact_force_n:
            self._contact_since = now if self._contact_since is None else self._contact_since
            if now - self._contact_since >= c.contact_hold_s:
                return self._end("done", f"contact ({force_n:.2f} N for {now - self._contact_since:.2f} s)")
        else:
            self._contact_since = None
        if distance_m <= c.distance_tolerance_m:
            return self._end("done", f"at the grasp point ({1000 * distance_m:.1f} mm)")
        if now - self.start_t > c.timeout_s:
            return self._end("abort", f"no arrival within {c.timeout_s} s")

        # a vanishing stem mask stops the advance; if it stays gone, give up
        if mask_pixels < c.min_mask_pixels:
            if now - self._mask_ok_t > c.mask_wait_s:
                return self._end("abort", f"stem mask under {c.min_mask_pixels} px for {c.mask_wait_s} s")
            self._aligned_since = None
            return ApproachStatus(self.phase, 0.0)
        self._mask_ok_t = now

        if self.phase == "advance":
            if distance_m <= self._step_goal_d or error_px > 3.0 * c.align_tolerance_px:
                self.phase, self._aligned_since = "align", None
                return ApproachStatus("align", 0.0)
            return ApproachStatus("advance", c.speed_mps)

        # align: hold position until the servo has kept the stem on the axis
        if error_px > c.align_tolerance_px:
            self._aligned_since = None
            return ApproachStatus("align", 0.0)
        if self._aligned_since is None:
            self._aligned_since = now
        if now - self._aligned_since < c.settle_s:
            return ApproachStatus("align", 0.0)
        if self._step_goal_d is None or distance_m <= self._step_goal_d:   # next step
            if self.steps >= c.max_steps:
                return self._end("abort", f"not at the grasp point after {c.max_steps} steps")
            self.steps += 1
            self._step_goal_d = distance_m - c.step_m
        self.phase = "advance"
        return ApproachStatus("advance", c.speed_mps)


@dataclass
class GripperCloseMonitor:
    """Decides when a closing gripper has settled (P0.4.13 grasp).

    ``update`` takes the measured opening and returns "wait", "grasped" (the
    opening stopped changing above ``min_object_m``: something is between the
    fingers), "empty" (it closed below ``min_object_m``) or "timeout".
    """

    start_t: float
    settle_s: float = 0.5
    timeout_s: float = 5.0
    min_object_m: float = 0.002
    still_m: float = 0.0005            # change below this counts as settled
    _ref: Optional[float] = None
    _ref_t: Optional[float] = None
    width: Optional[float] = None

    def update(self, now: float, width: Optional[float]) -> str:
        if width is None:
            return "timeout" if now - self.start_t > self.timeout_s else "wait"
        self.width = width
        if self._ref is None or abs(width - self._ref) > self.still_m:
            self._ref, self._ref_t = width, now
        if now - self._ref_t >= self.settle_s and now - self.start_t >= self.settle_s:
            return "grasped" if width > self.min_object_m else "empty"
        return "timeout" if now - self.start_t > self.timeout_s else "wait"


@dataclass
class RetreatMonitor:
    """Tracks the straight retreat out of the plant after a release.

    ``update`` takes the TCP position (world) and returns "wait", "done" once
    it has backed ``distance_m`` along -``axis`` (the gripper approach axis at
    the start), or "timeout".
    """

    start_t: float
    start_tcp: np.ndarray
    axis: np.ndarray
    distance_m: float = 0.10
    timeout_s: float = 15.0
    moved: float = 0.0

    def update(self, now: float, tcp: Optional[np.ndarray]) -> str:
        if tcp is not None:
            self.moved = float((np.asarray(self.start_tcp) - np.asarray(tcp)) @ self.axis)
            if self.moved >= self.distance_m:
                return "done"
        return "timeout" if now - self.start_t > self.timeout_s else "wait"


@dataclass
class ServoGates:
    """Phase 2B hard gates (ROADMAP Phase 2B, research/whole_body_mpc §12)."""
    max_force_n: float = 2.0           # max_force_n: retract
    force_max_age_s: float = 0.5       # force_max_age_sec: a force source that falls silent stops (0 = off)
    lost_target_s: float = 2.0         # servo_lost_target_sec: re-ground after this without a mask (0 = off)


def servo_gate(g: ServoGates, approaching: bool, force_n: float, force_age_s: Optional[float],
               mask_age_s: float) -> Optional[Tuple[str, str]]:
    """The first gate that trips in a servo phase, as (action, reason), or None.

    "retract": the force limit is exceeded (back out along the gripper axis);
    "stop": a force source that was seen has fallen silent (``force_age_s``
    None: never seen, no source configured);
    "reground": the target has been lost for ``lost_target_s`` while servoing
    (back to SCANNING). In the approach a lost mask ends the approach instead
    (``ApproachConfig.mask_wait_s``), so this gate is for SERVOING only."""
    if force_n > g.max_force_n:
        return "retract", f"force limit exceeded ({force_n:.2f} N > {g.max_force_n} N)"
    if force_age_s is not None and g.force_max_age_s > 0 and force_age_s > g.force_max_age_s:
        return "stop", f"no force for {force_age_s:.1f} s (> {g.force_max_age_s} s)"
    if not approaching and g.lost_target_s > 0 and mask_age_s > g.lost_target_s:
        return "reground", f"target lost for {mask_age_s:.1f} s (> {g.lost_target_s} s)"
    return None
