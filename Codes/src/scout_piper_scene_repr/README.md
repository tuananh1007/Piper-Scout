# scout_piper_scene_repr (Phase 1)

Per-class semantic 3D scene representation for the Piper+Scout stack.

**Status:** P1.1.2 RealSense -> nvblox smoke test is working in the dev
container. The CPU semantic voxel map + planner distance query (P1.7, see
below) is implemented and tested without ROS. The MoveIt 2 collision plugin
(P1.7.7) checks the robot against that query's distance field on top of FCL;
it is tested against a real MoveIt robot model and planning scene, but not
yet on the robot or on recorded plant scenes (P1.7.6), so treat it as
unvalidated until then.

## What's here

| Path | Status | Purpose |
|---|---|---|
| [`docs/PHASE1_DESIGN.md`](docs/PHASE1_DESIGN.md) | done | Design doc — read first |
| [`python/class_demux_node.py`](python/class_demux_node.py) | working | Fans semantic label image → per-class mask + gated depth; adds `other` (depth outside every mask) and removes the gripper fingers (self-filter) |
| [`python/nvblox_field_bridge.py`](python/nvblox_field_bridge.py) | run against fake class services with the `nvblox_msgs` interface; untested against nvblox | Per-class nvblox ESDFs (`~/get_esdf_and_gradient`) → `/scene_repr/distance_field` + `/scene_repr/target_goal` (P1.3.1, P3.2.2) |
| [`rviz/semantic_scene.rviz`](rviz/semantic_scene.rviz) | not yet opened on a GPU run | Per-class nvblox meshes, CPU voxels, target goal, masks (`nvblox_semantic.launch.py rviz:=true`) |
| [`src/semantic_collision_plugin.cpp`](src/semantic_collision_plugin.cpp) | working, not on hardware | MoveIt 2 collision plugin `"Semantic"`: FCL + semantic distance field (see below) |
| [`src/semantic_field_listener.cpp`](src/semantic_field_listener.cpp) | working, not on hardware | Receives `/scene_repr/distance_field` inside move_group; parameters, TF |
| [`include/.../semantic_distance_field.hpp`](include/scout_piper_scene_repr/semantic_distance_field.hpp) | working | Header-only field sampler + robot sphere cover (C++ twin of `field.py`) |
| [`msg/SemanticDistanceField.msg`](msg/SemanticDistanceField.msg) | done | Distance-field snapshot message |
| [`config/semantic_collision.yaml`](config/semantic_collision.yaml) | done | Plugin parameters (pass to move_group) |
| [`config/semantic_classes.yaml`](config/semantic_classes.yaml) | done | Per-class policy (hard / soft / attractor) |
| [`config/nvblox_per_class.yaml`](config/nvblox_per_class.yaml) | done | Per-class nvblox tuning |
| [`config/realsense_nvblox.yaml`](config/realsense_nvblox.yaml) | working | Single-camera nvblox smoke-test profile |
| [`launch/realsense_nvblox.launch.py`](launch/realsense_nvblox.launch.py) | working | P1.1.2 RealSense color/depth -> nvblox TSDF/ESDF launch |
| [`launch/nvblox_semantic.launch.py`](launch/nvblox_semantic.launch.py) | untested | Phase 1 v0 launch: demux + one `nvblox_ros` node per class (stem, branch, leaf, target, other; `classes:=`) + the field bridge (`field_bridge:=true`) + RViz (`rviz:=true`); not yet run (P1.2.1, MODULE_TASKS.md A4) |
| [`launch/test_class_demux.launch.py`](launch/test_class_demux.launch.py) | working | P1.1.0 plumbing test with `python/test_mask_publisher.py` (no nvblox) |
| [`python/scout_piper_scene_repr_py/`](python/scout_piper_scene_repr_py/) | working | CPU `SemanticVoxelMap` + `SemanticDistanceQuery` (v0 query backend, see below) |
| [`python/scene_query_node.py`](python/scene_query_node.py) | hardware-free checked | Live CPU map (fingers self-filtered); `/scene_repr/voxels`, `/scene_repr/map_status`, `/scene_repr/distance_field`, `/scene_repr/target_goal` |
| [`plugin_description.xml`](plugin_description.xml) | done | pluginlib export (`collision_detection::CollisionPlugin` named `Semantic`) |

## Prerequisites

1. `nvblox_ros` built from source in the dev container; see [`docs/PHASE1_RUNTIME.md`](docs/PHASE1_RUNTIME.md).
2. Intel RealSense publishing aligned depth (the P1.1.2 smoke test used a D405; the URDF models a D435; model to be confirmed).
3. For semantic v0, the `stem_grasp` segmentation node publishing either:
   - **merged mode** (default; `segmentation_node` publishes it since 2026-10-09, `publish_semantic_label`): a single `mono8` label image on `/stem_grasp/semantic_label` (1=stem, 2=branch, 3=leaf, 4=target; YOLO classes mapped with `yolo_class_labels`), OR
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

## nvblox-backed distance field, target goal, self-filter (2026-10-09)

- **Bridge** (`nvblox_field_bridge.py`, started by `nvblox_semantic.launch.py`):
  every 1 s (`rate_hz`) it requests each class mapper's ESDF inside the plant
  box (`grid_center` ± `grid_half_extent_m`, default 0.3 m), merges as soon as
  every answer is in, and
  resamples the grids (stem 3 mm, leaf 10 mm, other 2 cm) onto one 1 cm grid
  (`nvblox_field.merge_class_grids`: hard classes min(d − padding) minus half
  a voxel diagonal, so resampling never makes a thin stem look farther; the
  leaf class as the soft field; a voxel is known when any mapper observed
  it) and publishes the same `SemanticDistanceField` as `scene_query_node`.
  Run one of the two, not both. Status on `/scene_repr/bridge_status`.
  Each answer comes at the mapper's voxel size, 4 bytes per voxel: with the
  3 mm stem and target mappers a 0.6 m box is 8.0 M voxels (32 MB) per class
  and request, a 1 m box 37 M voxels (148 MB); merging the five classes onto
  the 60³ grid took 258 ms on the development PC. `bridge_status` reports the
  answer sizes (`answers`), `request_s` and `merge_ms`; keep the box tight
  around the plant. The whole-body MPC stops when the newest field is older
  than its `field_max_age_s` (2.5 s).
- **`other` class:** depth outside every class mask gets its own mapper (pots,
  walls, supports); its observations also mark the space as observed.
- **Target goal** (`attractor.py`, P1.3.3): the largest connected cluster of
  target voxels, approached horizontally from `base_link`, pre-grasp 12 cm in
  front; `/scene_repr/target_goal` (PoseStamped, z = approach). The MPC follows
  it with `whole_body_mpc.launch.py goal_pose_topic:=/scene_repr/target_goal`.
- **Self-filter** (`self_filter.py`, P1.7.8): boxes on `piper_link7` /
  `piper_link8` (`self_filter_boxes`) projected with the depth intrinsics;
  pixels inside a box and no farther than it are removed before integration.
  The default boxes are estimates: tune them on the robot (MODULE_TASKS.md C1).
- **Offline** (`offline.py`): the map from an exported episode (depth, poses,
  labels), for the recorded-scene benchmark (P1.4.2,
  scout_piper_whole_body_mpc `benchmarks/semantic_vs_occupancy.py --episode`).

## Next steps

[`MODULE_TASKS.md`](../../../MODULE_TASKS.md) A4, A5, B3, C1, C4, C5, D1; see
`P1.x` in [`../../../PROGRESS.md`](../../../PROGRESS.md).

## CPU semantic map + planner distance query (v0 query backend)

`python/scout_piper_scene_repr_py/` holds the planner-facing query from
[`research/semantic_scene/`](../../../research/semantic_scene/) (contribution C4).
It runs without nvblox and is the reference the nvblox-backed path must match.

| Module | Role |
|---|---|
| `policy.py` | loads `config/semantic_classes.yaml`; adds an `other` class (hard, 1 cm) for depth outside every mask (pots, walls, supports) |
| `voxel_map.py` | `SemanticVoxelMap`: per-class hit evidence + shared free space from aligned depth and class masks. Hits win over free space within a frame (keeps thin stems); free space clears moved leaves; never-seen voxels stay unknown |
| `distance_query.py` | `SemanticDistanceQuery.query(points)`: per-class signed distance, hard minimum with padding, nearest hard class, gradient, `valid` (unknown / out of bounds / stale ⇒ False and hard distance clamped ≤ 0). `sphere_clearance`, `leaf_cost` (ψ with the 2 cm penetration cap ⇒ hard violation), `target_attraction`, grasp mode that releases only the target region |
| `ros_integrator.py` | `RosSceneIntegrator`: subscribes to aligned depth, its camera info and `/scene_repr/mask/<class>`, and feeds a `SemanticVoxelMap` in the caller's process (used by `scene_query_node.py` and the whole-body MPC) |
| `field.py` | `export_field` → `DistanceFieldSnapshot` (hard classes merged with their padding, leaf field, per-voxel age) and `fill_msg`; `FieldSampler` is the numpy reference of the C++ sampler |
| `python/scene_query_node.py` | integrates live data, publishes `/scene_repr/voxels` (MarkerArray), `/scene_repr/map_status` (JSON incl. integrate and export time) and `/scene_repr/distance_field` |

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

## MoveIt semantic collision plugin (P1.7.7)

`collision_detector: "Semantic"` in the move_group configuration loads
`CollisionEnvSemantic`: MoveIt's FCL environment (self-collision and
planning-scene objects unchanged) plus a check of the robot against the
distance field that `scene_query_node` publishes.

```text
scene_query_node ── /scene_repr/distance_field ──▶ move_group
  SemanticDistanceQuery          (reliable, transient    semantic_collision node (listener)
  └ field.export_field            local, ≤ field_rate_hz)  └ CollisionEnvSemantic = FCL + field
```

* **Robot geometry.** Each link collision shape is covered once by
  overlapping spheres: a bounding cylinder on the shape's longest axis, split
  into ⌈length / radius⌉ spheres of radius ≤ 1.12 × the cylinder radius. Link
  padding and `sphere_padding_m` are added to each sphere.
* **Per sphere** (`SemanticDistanceField::checkSphere`): collision when the
  merged hard field (stem, branch, `other`, each with its padding) is closer
  than the radius, when the sphere overlaps the leaf by more than
  `max_penetration_m`, or (with `unknown_is_occupied`) when its centre voxel
  was never observed or is older than `max_voxel_age_s`. Outside the grid
  the field says nothing and only FCL applies.
* **No usable field** (none received, older than `max_field_age_s`, or no TF
  from the robot model frame to the field frame) with `require_field: true`
  (default): every robot check reports a collision, so planning stops instead
  of ignoring the plant.
* **Contacts and distances** name the other body `semantic/<class>`
  (`semantic/unknown`, `semantic/stale`). Allowing (`<link>`, `semantic`) in
  the AllowedCollisionMatrix, or listing the link in `ignore_links`, exempts
  a link, for example the fingers during the final grasp.
* **Motion segments.** FCL has no continuous check on Humble; the plugin
  samples both FCL and the field every `continuous_step` of
  `RobotState::distance` along the segment.

Parameters are in [`config/semantic_collision.yaml`](config/semantic_collision.yaml);
pass that file to the move_group process (`--ros-args --params-file ...`),
since the plugin reads them through its own node (`semantic_collision`).

**Size and rate.** A message carries 11 B per voxel: the default
`scene_query_node` grid (1 m cube, 1 cm voxels, 10⁶ voxels) is ≈ 11 MB, and
exporting it took ≈ 0.5 s when the per-class ESDFs must be rebuilt (5 mm,
10⁶-voxel test grid, 4-core x86 dev CPU; ≈ 50 ms for 1.25 × 10⁵ voxels). Shrink
`grid_half_extent_m` around the plant, or lower `field_rate_hz`, on the Orin;
keep `max_field_age_s` above the publish period plus the export time.

**Limitations.**

* Depth that hits the robot itself (fingers in view of the wrist camera)
  lands in `other` and becomes an obstacle; there is no robot self-filter yet.
  Exempt such links via the ACM or `ignore_links` until there is.
* The leaf cap is judged from the signed distance at the sphere centre: a
  single-voxel leaf never reads deeper than half a voxel, so in practice a
  sphere collides with a leaf only when it overlaps it by more than
  `max_penetration_m`.
* No grasp-mode exclusion in the plugin (plan to a pre-grasp; the final
  approach is the servo's job).
* Not yet run on the robot, on recorded plant scenes (P1.7.6) or on the Orin.

**Tests** (`colcon test --packages-select scout_piper_scene_repr`):

* `test/test_field_snapshot.py` — export layout and message round trip;
  the numpy reference sampler against `SemanticDistanceQuery` (never less
  conservative, equal where one class dominates); sphere statuses (free,
  hard, leaf cap, unknown, stale, outside); the C++ sampler
  (`test/field_query_driver.cpp`, compiled by the test) against the numpy
  reference to 1e-9 on 600 points in four configurations; sphere-cover
  completeness.
* `test/test_semantic_collision_plugin.cpp` (gtest) — a real MoveIt robot
  model and a field published on the topic: collision reported while no field
  has arrived; free 5 cm from a stem, colliding inside its padding with a
  `semantic/stem` contact; unknown space collides, outside the grid is free;
  ACM exemption; distance and normal match the geometry; a motion through the
  stem collides although both ends are free; `PlanningScene` loads the plugin
  by name through pluginlib, as move_group does.
