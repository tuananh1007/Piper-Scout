# scout_piper_scene_repr (Phase 1)

Per-class semantic 3D scene representation for the Piper+Scout stack.

**Status:** Phase 1 skeleton — package compiles (once nvblox is installed); the
MoveIt 2 collision plugin returns "no collision" placeholders, and the
`class_demux_node` is the only fully-working component.

## What's here

| Path | Status | Purpose |
|---|---|---|
| [`docs/PHASE1_DESIGN.md`](docs/PHASE1_DESIGN.md) | done | Design doc — read first |
| [`python/class_demux_node.py`](python/class_demux_node.py) | working | Fans semantic label image → per-class mask + gated depth |
| [`include/.../semantic_collision_plugin.hpp`](include/scout_piper_scene_repr/semantic_collision_plugin.hpp) | scaffold | MoveIt 2 plugin interface |
| [`src/semantic_collision_plugin.cpp`](src/semantic_collision_plugin.cpp) | scaffold | Plugin impl — TODO P1.4 |
| [`config/semantic_classes.yaml`](config/semantic_classes.yaml) | done | Per-class policy (hard / soft / attractor) |
| [`config/nvblox_per_class.yaml`](config/nvblox_per_class.yaml) | done | Per-class nvblox tuning |
| [`launch/nvblox_semantic.launch.py`](launch/nvblox_semantic.launch.py) | done | Phase 1 v0 launch (demux + 4 nvblox nodes) |
| [`plugin_description.xml`](plugin_description.xml) | done | pluginlib export |

## Prerequisites

1. `isaac_ros_nvblox` installed (Docker per [NVIDIA Isaac ROS docs](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_nvblox/index.html), or native build).
2. The `stem_grasp` segmentation node publishing either:
   - **merged mode**: a single `mono8` label image on `/stem_grasp/semantic_label` (1=stem, 2=branch, 3=leaf, 4=target), OR
   - **separate mode** (Phase 0 fallback): the legacy `/stem_grasp/mask` + `/stem_grasp/target_mask` pair.

## Run (Phase 1 v0)

```bash
# After Phase 0 is up and segmentation is publishing
ros2 launch scout_piper_scene_repr nvblox_semantic.launch.py
# or, while segmentation still emits the legacy two-topic format:
ros2 launch scout_piper_scene_repr nvblox_semantic.launch.py input_mode:=separate
```

## Next steps

See `P1.x` in [`../../../../PROGRESS.md`](../../../../PROGRESS.md).
