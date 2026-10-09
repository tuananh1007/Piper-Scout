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

## Final approach (P0.4.13)

`approach.py` is a new design, because the ROS 1 state machine is not in this
repository; it keeps the ROS 1 `approach_*` parameter names. With
`approach_enabled: true` the reach handoff goes to APPROACHING:

- the servo keeps the stem on the gripper axis;
- once aligned, the gripper advances along its axis in `approach_step` steps,
  re-aligning between steps;
- it stops in AT_GRASP at `approach_distance_tolerance` from the grasp point;
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
