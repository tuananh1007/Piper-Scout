# stem_grasp (ROS 2)

ROS 2 Humble port of `stem_grasp_ros1` (the ROS 1 package from the original `piper_ros` workspace on the lab machine; it is not in this repository).

## Status: Phase 0 port

`core.py` (servo math and skeleton/candidate helpers), `moveit_planner.py`,
`segmentation_node`, `pointcloud_node` and `pipeline_node` are ported from the
ROS 1 source. The grasp sequence runs hardware-free, not yet on the robot:

- reach to the pre-grasp pose through the whole-body MPC (P0.4.11,
  `reach_executor: whole_body_mpc`);
- image-based servo (P0.4.12);
- stepwise final approach (P0.4.13, `approach_enabled`).

- closing the gripper on the stem (`grasp_close_gripper`, GRASPING →
  GRASPED);
- releasing it (`~/release`: RELEASING → RETREATING → IDLE; `~/scan` scans
  again).

The grasp point is on the stem axis. The skeleton of a single-view cloud
follows the camera-facing half of the stem, so `core.stem_axis_point` fits a
circle to the cross-section around each skeleton point and moves the point to
its centre (`grasp_point_on_stem_axis`). The pipeline logs the fitted diameter:
7.7 mm for the 8 mm synthetic stem.

Open:
- The MoveIt path is unwired: `moveit_py` has no Humble binary, and
  `moveit_planner.py` returns no plan when it is absent.

## MPPI visual servo (Phase 2B, `servo_controller: mppi`)

`servo_controller: mppi` replaces the IBVS below (which stays the default and
the baseline) with `scout_piper_whole_body_mpc.visual_servo.MppiVisualServo`:
an arm-only MPPI over the six joint velocities, published as `JointJog` on
`servo_joint_cmd_topic` (frame_id `stem_grasp`), with the Scout still. Per new
mask it takes the same geometry as the IBVS (measured stem pixel, the pixel
where the gripper axis crosses the target depth, the metric target), plus the
joint states and the link6 → camera transform from TF. In SERVOING it holds the
distance to the stem; in APPROACHING the stepwise approach (`approach.py`)
sets the distance it should reach over the horizon. Every command passes the
whole-body `SafetyFilter` (velocity, acceleration, joint limits); the force,
mask-age and approach aborts below still apply. `/stem_grasp/servo_status`
then also carries `solve_ms`, the image-prediction `mode` and the cost of each
term for the executed plan and the resolved-rate nominal (`plan_terms`,
`nominal_terms`).

Hardware-free (fake arm and base, `hardware_free_checks.sh --quick --servo
mppi`): the full grasp passes — approach in 12 s, gripper axis 0.02 cm from the
stem centreline, gripper closed on the stem, release and retreat. Not run on
the robot or the GPU yet (MODULE_TASKS.md A13, B7, C9).

**Simulated trials (P2.3.1), `benchmarks/servo_trials.py`.** 50 scripted final
approaches from the MPC handoff, same draws for each controller: start offset
up to 2 cm and 10°, hand-eye error (σ 4 mm, 1.5°), grasp-point estimate bias
(σ 3 mm), stem sway (up to 6 mm at 0.2–0.8 Hz), a neighbouring stem 5–9 cm
away in half the trials, force-estimate noise. The pipeline's approach logic
runs unchanged; the stem mask is rendered through the true camera; the IBVS
twist is executed as moveit_servo does. Success: AT_GRASP with the stem within
1 cm of the gripper axis, no collision, no force-limit hit.

| Approach alignment | IBVS | MPPI | MPPI + field | Median approach (MPPI / IBVS) |
|---|---|---|---|---|
| per sample (before) | 33/50 | 37/50 | 38/50 | 8.6 / 14.9 s |
| 1.5 s window (`approach_align_window_sec`) | 48/50 | 49/50 | 47/50 | 8.9 / 14.6 s |

Every failure before the window was a timeout in the 11 trials whose stem sways
faster than ~15 mm/s: the error never stayed under 8 px for 0.4 s. With the
window the two servos are equal in success (the Phase 2B exit gate's 20-point
gain is not reachable here); the MPPI servo approaches about 40 % faster,
lateral miss median 3.0 vs 4.0 mm. The simulated contact never engages (the
approach stops 1 cm short with the stem between the open fingers), so the
force half of the gate needs the robot. The trials also found two faults, both
fixed: the MPPI servo stalled under hand-eye *translation* error (its target is
now anchored to the measured pixel's ray), and one noisy force sample ended an
approach (`contact_hold_sec`).

    cd Codes/src
    PYTHONPATH=stem_grasp:scout_piper_whole_body_mpc:scout_piper_scene_repr/python \
        python3 stem_grasp/benchmarks/servo_trials.py --trials 50 --align-window 1.5 --out trials.jsonl

## Image-based servo (P0.4.12)

In SERVOING, `_inner_loop` steps `core.FullAdaptiveServoController` once per
new mask (`servo_geometry.py` holds the image geometry):

- **target**: the live `/stem_grasp/target_point` if fresher than
  `target_point_max_age_sec`, else the grasp point chosen before the reach;
- **desired pixel**: where the gripper approach axis (`eef_frame` z, TCP at
  `tcp_offset_m`) crosses the target's depth, from TF. ROS 1 used the image
  centre, but the camera sits beside the gripper;
- **measured pixel**: the stem column in the mask rows around the target's row
  (`servo_row_band_px`);
- **output**: a camera translation, published in `camera_optical_frame`.

Deliberate changes from the ROS 1 maths are listed in the `core.py` docstring:
centred image Jacobian, observer start and time step, and sway measured against
the camera's TF motion. Hardware-free (`grasp_chain_check.py`, INSTALL.md 10.9)
the error falls from 33–65 px to 0.8–5 px in 15 s at four stem positions. The
gains are not tuned on the robot.

The segmentation node skips frames while the joints move
(`segment_only_when_stationary`), except in the pipeline states in
`motion_gate_off_states` (default SERVOING and APPROACHING): the servo and
the final approach move the arm and need a fresh mask at every step.

## Final approach (P0.4.13)

`approach.py` is a new design, because the ROS 1 state machine is not in this
repository; it keeps the ROS 1 `approach_*` parameter names. With
`approach_enabled: true` the reach handoff goes to APPROACHING:

- the servo keeps the stem on the gripper axis;
- once aligned, the gripper advances along its axis in `approach_step` steps,
  re-aligning between steps;
- it stops in AT_GRASP at `approach_distance_tolerance` from the grasp point, or
  on contact (force above `contact_threshold_n` for `contact_hold_sec`);
- alignment is judged on the mean image-error vector over
  `approach_align_window_sec` (1.5 s), so a swaying stem does not stall it;
- it goes to ABORTED (and stops) on too many steps, a lost mask, a start too far
  away, or a timeout.

Hardware-free: AT_GRASP after 3 steps in 17–24 s, with the TCP within 1 cm of
the grasp point (INSTALL.md 10.9). Off by default. It drives the gripper onto
the stem, and the Piper has no force sensor for the contact stop.

With `grasp_close_gripper` the pipeline opens the gripper before the approach
and closes it at AT_GRASP through `piper_servo_bridge` (`~/gripper_cmd`).
`GripperCloseMonitor` (approach.py) reports GRASPED once the measured opening
holds still above `grasp_min_object_m`, and ABORTED if the gripper closed on
nothing or did not settle.

## Nodes

| Node | Executable | Status |
|---|---|---|
| Pipeline orchestrator | `pipeline_node` | Ported: state machine, outer loop (skeleton + candidate selection → `/stem_grasp/target_pose`), inner-loop image-based servo toward the gripper axis (P0.4.12), stepwise final approach (P0.4.13, `approach_enabled`). Reach to the pre-grasp pose through the whole-body MPC with `reach_executor: whole_body_mpc` (P0.4.11, `reach_handoff.py`; default `none` only publishes the pose) |
| Segmentation (YOLO + Grounded-SAM) | `segmentation_node` | Ported (YOLO-seg, Grounded-SAM + target caption, HSV fallback) |
| Point cloud filter | `pointcloud_node` | Ported (`/stem_grasp/filtered_cloud`, `/stem_grasp/leaf_filtered_cloud`) |
| Hot-key stop | `hotkey_stop_and_zero` | Working: `x` publishes zero twists and disables `piper_servo_bridge` (the arm holds its measured pose); stops servo-driven motion only, not the base; zero-arm service not ported |

## Port plan

Remaining port items are tracked under P0.4 in [`../../../PROGRESS.md`](../../../PROGRESS.md);
the original per-node chunking is Step 6 of [`../../PHASE0_CHECKLIST.md`](../../PHASE0_CHECKLIST.md).

## Run standalone

```bash
ros2 launch stem_grasp stem_grasp.launch.py
```

For the integrated arm + base + camera bringup, use
[`scout_piper_bringup`](../scout_piper_bringup/).
