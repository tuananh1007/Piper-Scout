# scout_piper_scene_repr (Phase 1)

Per-class semantic 3D scene representation for the Piper+Scout stack.

**Status:** P1.1 RealSense -> nvblox smoke test is working in the dev
container. The MoveIt 2 collision plugin still returns "no collision"
placeholders until P1.4.

## What's here

| Path | Status | Purpose |
|---|---|---|
| [`docs/PHASE1_DESIGN.md`](docs/PHASE1_DESIGN.md) | done | Design doc — read first |
| [`python/class_demux_node.py`](python/class_demux_node.py) | working | Fans semantic label image → per-class mask + gated depth |
| [`include/.../semantic_collision_plugin.hpp`](include/scout_piper_scene_repr/semantic_collision_plugin.hpp) | scaffold | MoveIt 2 plugin interface |
| [`src/semantic_collision_plugin.cpp`](src/semantic_collision_plugin.cpp) | scaffold | Plugin impl — TODO P1.4 |
| [`config/semantic_classes.yaml`](config/semantic_classes.yaml) | done | Per-class policy (hard / soft / attractor) |
| [`config/nvblox_per_class.yaml`](config/nvblox_per_class.yaml) | done | Per-class nvblox tuning |
| [`config/realsense_nvblox.yaml`](config/realsense_nvblox.yaml) | working | Single-camera nvblox smoke-test profile |
| [`launch/realsense_nvblox.launch.py`](launch/realsense_nvblox.launch.py) | working | P1.1.2 RealSense color/depth -> nvblox TSDF/ESDF launch |
| [`launch/nvblox_semantic.launch.py`](launch/nvblox_semantic.launch.py) | done | Phase 1 v0 launch (demux + 4 nvblox nodes) |
| [`plugin_description.xml`](plugin_description.xml) | done | pluginlib export |

## Prerequisites

1. `nvblox_ros` built from source in the dev container; see [`docs/PHASE1_RUNTIME.md`](docs/PHASE1_RUNTIME.md).
2. Intel RealSense D405 publishing aligned depth.
3. For semantic v0, the `stem_grasp` segmentation node publishing either:
   - **merged mode**: a single `mono8` label image on `/stem_grasp/semantic_label` (1=stem, 2=branch, 3=leaf, 4=target), OR
   - **separate mode** (Phase 0 fallback): the legacy `/stem_grasp/mask` + `/stem_grasp/target_mask` pair.

## Run P1.1.2 RealSense -> nvblox

```bash
# Inside the dev container after building and sourcing install/setup.bash:
ros2 launch scout_piper_scene_repr realsense_nvblox.launch.py
```

Quick checks:

```bash
ros2 topic hz /camera/aligned_depth_to_color/image_raw
ros2 topic list | grep nvblox_node
```

In RViz, set the fixed frame to `camera_link` and add the nvblox mesh, TSDF
marker, or static ESDF pointcloud topics.

## Run Phase 1 v0 Semantic Stack

```bash
# After Phase 0 is up and segmentation is publishing:
ros2 launch scout_piper_scene_repr nvblox_semantic.launch.py
# or, while segmentation still emits the legacy two-topic format:
ros2 launch scout_piper_scene_repr nvblox_semantic.launch.py input_mode:=separate
```

## Next steps

See `P1.x` in [`../../../../PROGRESS.md`](../../../../PROGRESS.md).

## CPU semantic map + planner distance query (v0 query backend)

`python/scout_piper_scene_repr_py/` holds the planner-facing query from
[`research/semantic_scene/`](../../../research/semantic_scene/) (contribution C4).
It runs without nvblox and is the reference the nvblox-backed path must match.

| Module | Role |
|---|---|
| `policy.py` | loads `config/semantic_classes.yaml`; adds an `other` class (hard, 1 cm) for depth outside every mask (pots, walls, supports) |
| `voxel_map.py` | `SemanticVoxelMap`: per-class hit evidence + shared free space from aligned depth and class masks. Hits win over free space within a frame (keeps thin stems); free space clears moved leaves; never-seen voxels stay unknown |
| `distance_query.py` | `SemanticDistanceQuery.query(points)`: per-class signed distance, hard minimum with padding, nearest hard class, gradient, `valid` (unknown / out of bounds / stale ⇒ False and hard distance clamped ≤ 0). `sphere_clearance`, `leaf_cost` (ψ with the 2 cm penetration cap ⇒ hard violation), `target_attraction`, grasp mode that releases only the target region |
| `python/scene_query_node.py` | integrates live data, publishes `/scene_repr/voxels` (MarkerArray) and `/scene_repr/map_status` (JSON incl. integrate time) |

Controllers should embed `SemanticVoxelMap` + `SemanticDistanceQuery` in
their own process for high-rate queries.

Measured on a 4-core x86 dev CPU (one 320×240 frame, 0.5 m cube):

| Voxel | Stride | Integrate | ESDF rebuild (5 classes) | Cached query, 512 pts |
|---|---|---|---|---|
| 5 mm | 4 | ≈100 ms | ≈510 ms | ≈26 ms |
| 10 mm | 4 | ≈33 ms | ≈48 ms | ≈4 ms |

So at 5 mm this is a low-rate snapshot for the MPC (research plan kill
criterion K3), not a 30 Hz field. nvblox remains the GPU path.

```bash
cd Codes/src/scout_piper_scene_repr && python -m pytest test -q
```

Tests ray-cast a 1 cm stem, a leaf patch and a wall: thin stem preserved
(>90 % centreline coverage), stem distance within one voxel, unknown space
not reported free, stale geometry invalid, wall hard via `other`, leaf soft,
grasp mode releases only the target region, a moved leaf is cleared.

The MoveIt plugin (`src/semantic_collision_plugin.cpp`) is still the scaffold
that reports no collision; wiring it to this query (or to nvblox) is P1.4.
