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
           axis, or contact force >= contact_threshold_n
    abort: more than approach_max_steps steps, the TCP farther than
           approach_target_distance when the approach starts, the stem mask
           below approach_min_leaf_pixels for approach_mask_wait_sec, or
           approach_timeout_sec overall

The caller feeds one update per servo step (new mask) and adds
``speed * approach_axis`` to the servo's camera velocity. This module holds
no ROS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
            return self._end("done", f"contact ({force_n:.2f} N)")
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
