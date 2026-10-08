# stem_grasp (ROS 2)

ROS 2 Humble port of `stem_grasp_ros1` (the ROS 1 package from the original `piper_ros` workspace on the lab machine; it is not in this repository).

## Status: Phase 0 port

`core.py` (servo math and skeleton/candidate helpers), `moveit_planner.py`,
`segmentation_node`, `pointcloud_node` and most of `pipeline_node` are ported
from the ROS 1 source. Open items are marked `TODO(P0.4.x)` in `pipeline_node.py`:

- **P0.4.11**: plan → execute. Done through the whole-body MPC instead
  (`reach_executor: whole_body_mpc`, `reach_handoff.py`). The MoveIt path is
  still unwired: `moveit_py` has no Humble binary, and `moveit_planner.py`
  returns no plan when it is absent.
- **P0.4.13**: iterative approach state machine (not ported).

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

## Nodes

| Node | Executable | Status |
|---|---|---|
| Pipeline orchestrator | `pipeline_node` | Ported: state machine, outer loop (skeleton + candidate selection → `/stem_grasp/target_pose`), inner-loop image-based servo toward the gripper axis (P0.4.12). Reach to the pre-grasp pose through the whole-body MPC with `reach_executor: whole_body_mpc` (P0.4.11, `reach_handoff.py`; default `none` only publishes the pose); iterative approach not ported (P0.4.13) |
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
