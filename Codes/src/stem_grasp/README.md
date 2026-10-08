# stem_grasp (ROS 2)

ROS 2 Humble port of `stem_grasp_ros1` (the ROS 1 package from the original `piper_ros` workspace on the lab machine; it is not in this repository).

## Status: Phase 0 port

`core.py` (servo math and skeleton/candidate helpers), `moveit_planner.py`,
`segmentation_node`, `pointcloud_node` and most of `pipeline_node` are ported
from the ROS 1 source. Open items are marked `TODO(P0.4.x)` in `pipeline_node.py`:

- **P0.4.11**: plan → execute. The outer loop publishes `/stem_grasp/target_pose`
  but does not call `moveit_planner` yet. `moveit_py` has no Humble binary, and
  `moveit_planner.py` returns no plan when it is absent.
- **P0.4.13**: iterative approach state machine (not ported).

## Nodes

| Node | Executable | Status |
|---|---|---|
| Pipeline orchestrator | `pipeline_node` | Ported: state machine, outer loop (skeleton + candidate selection → `/stem_grasp/target_pose`), inner-loop servo (desired image point still the image centre). Reach to the pre-grasp pose through the whole-body MPC with `reach_executor: whole_body_mpc` (P0.4.11, `reach_handoff.py`; default `none` only publishes the pose); iterative approach not ported (P0.4.13) |
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
