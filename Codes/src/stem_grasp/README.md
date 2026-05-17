# stem_grasp (ROS 2)

ROS 2 Humble port of [`stem_grasp_ros1`](../../../../src/stem_grasp_ros1/).

## Status: Phase 0 skeleton

The package compiles and the nodes start cleanly, but the algorithm bodies
are stubbed. Each stub block carries a `TODO(port-from-ros1)` comment with
the line range in the original ROS 1 source.

## Nodes

| Node | Executable | Status |
|---|---|---|
| Pipeline orchestrator | `pipeline_node` | Skeleton (state machine + timers wired) |
| Segmentation (YOLO + Grounded-SAM) | `segmentation_node` | Empty stub |
| Point cloud filter | `pointcloud_node` | Empty stub |
| Hot-key e-stop | `hotkey_stop_and_zero` | Working (zero-twist publish) |

## Port plan

See [`../../PHASE0_CHECKLIST.md`](../../PHASE0_CHECKLIST.md) — the per-node
port is broken into independent chunks, each ~1–3 days of work.

## Run standalone

```bash
ros2 launch stem_grasp stem_grasp.launch.py
```

For the integrated arm + base + camera bringup, use
[`scout_piper_bringup`](../scout_piper_bringup/).
