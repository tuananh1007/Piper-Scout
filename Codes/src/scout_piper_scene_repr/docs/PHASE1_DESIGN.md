# Phase 1 Design — Semantic 3D Scene Representation

**Author:** Vu Tuan Anh
**Status:** Draft (Phase 1 kickoff, 2026-05-17)
**Companion code:** [`scout_piper_scene_repr/`](../)
**Roadmap context:** [`../../../../ROADMAP.md#phase-1`](../../../../ROADMAP.md#phase-1)

---

## 1. Problem

MoveIt's Octomap collision world treats every camera point identically:

- It cannot distinguish **stem** (must avoid) from **leaf** (may brush through) from **target peduncle** (must reach).
- Thin stems vanish at typical Octomap voxel resolutions (≥ 1 cm).
- No mechanism to encode "soft" or directional cost — the only output is binary occupied/free.

The current ROS 1 pipeline already extracts class-labeled masks via Grounded-SAM,
but throws this information away when it writes a generic point cloud to
`/static_cloud_out`. We want to *keep* the class labels through to the planner.

## 2. Goal

Replace the Octomap-based collision world with a **per-class signed distance field
(SDF) representation** that:

1. Maintains four (initially) distinct TSDFs/ESDFs: `stem`, `branch`, `leaf`, `target`.
2. Lets the planner apply class-specific cost policies (hard collision vs soft cost
   vs attractor).
3. Runs on GPU (Jetson Orin AGX) at ≥ 30 Hz update rate, sustaining the same depth
   stream rate as the camera.
4. Exposes a MoveIt 2 collision plugin interface so existing planners (OMPL, CHOMP,
   and the Phase 3 cuMotion replacement) can consume it transparently.

## 3. Architecture

```
                 ┌────────────────────────────────────────────────────┐
                 │ Camera (RealSense D435, eye-in-hand)               │
                 │   /camera/color/image_raw                          │
                 │   /camera/depth/image_rect_raw                     │
                 │   /camera/color/camera_info                        │
                 └──────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
                 ┌────────────────────────────────────────────────────┐
                 │ stem_grasp/segmentation_node                       │
                 │   YOLO + Grounded-SAM                              │
                 │   Publishes a single semantic-label image:         │
                 │   /stem_grasp/semantic_label (mono8 or 16UC1)      │
                 │   where pixel value ∈ {0:bg, 1:stem, 2:branch,     │
                 │                        3:leaf, 4:target}           │
                 └──────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
                 ┌────────────────────────────────────────────────────┐
                 │ scout_piper_scene_repr/class_demux_node            │
                 │   Fans the semantic image out into one binary      │
                 │   mask per class:                                  │
                 │     /scene_repr/mask/stem                          │
                 │     /scene_repr/mask/branch                        │
                 │     /scene_repr/mask/leaf                          │
                 │     /scene_repr/mask/target                        │
                 │   Also republishes RGB+depth on per-class topics   │
                 │   gated by each mask (depth pixels outside class   │
                 │   are set to NaN so nvblox skips them).            │
                 └──────────────────┬─────────────────────────────────┘
                                    │ × 4 streams
                                    ▼
                 ┌────────────────────────────────────────────────────┐
                 │ N × isaac_ros_nvblox nodes (one per class)         │
                 │   Each maintains its own TSDF + ESDF.              │
                 │   v0 implementation: stamped instances with        │
                 │   distinct namespaces.                             │
                 │   v1 (research contribution): fork nvblox to       │
                 │   carry a per-voxel class label and avoid the      │
                 │   N-way duplication.                               │
                 │ Outputs:                                           │
                 │   /scene_repr/esdf/{stem,branch,leaf,target}       │
                 │   (nav_msgs/OccupancyGrid + custom MarkerArray)    │
                 └──────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
                 ┌────────────────────────────────────────────────────┐
                 │ scout_piper_scene_repr/semantic_collision_plugin   │
                 │   pluginlib export of moveit_core::CollisionDetector│
                 │   Per-class behavior from semantic_classes.yaml:   │
                 │     stem     -> hard collision, padding 5 mm       │
                 │     branch   -> hard collision, padding 8 mm       │
                 │     leaf     -> SOFT cost (penalty/weight pair),   │
                 │                 padding 0 mm                       │
                 │     target   -> attractor (negative cost) within   │
                 │                 attract_radius_m                   │
                 │   Plugged into move_group via                      │
                 │   collision_detector parameter override.           │
                 └──────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
                 ┌────────────────────────────────────────────────────┐
                 │ MoveIt 2 move_group (and Phase 3 cuMotion)         │
                 │   Existing planners are unchanged; they ask the    │
                 │   plugin for collision/cost at each sample.        │
                 └────────────────────────────────────────────────────┘
```

## 4. Data contracts

### 4.1 Semantic label image

Single channel image, encoded as `mono8`:

| Value | Class | Source in current pipeline |
|---|---|---|
| 0 | background | (none — everything not below) |
| 1 | stem | Grounded-SAM `text_prompt:="stem"` (legacy) |
| 2 | branch | Grounded-SAM `text_prompt:="branch"` (current default) |
| 3 | leaf | Grounded-SAM `target_caption:="leaf"` (current) |
| 4 | target | Grounded-SAM `target_caption` when set to fruit/flower |

When the existing segmentation node publishes separate per-class masks
(`/stem_grasp/mask`, `/stem_grasp/target_mask`), the class_demux_node
**accepts either format**:
- `--input-mode merged` : a single label image (preferred).
- `--input-mode separate` : the legacy two-topic interface (fallback).

### 4.2 Per-class TSDF parameters

Defined in `config/nvblox_per_class.yaml`:

```yaml
stem:
  voxel_size_m: 0.003      # 3 mm — recover thin structure
  max_integration_distance_m: 0.8
  weighting: constant
branch:
  voxel_size_m: 0.005
  max_integration_distance_m: 1.0
  weighting: constant
leaf:
  voxel_size_m: 0.01       # coarser — leaves are large
  max_integration_distance_m: 1.0
  weighting: inverse_square
target:
  voxel_size_m: 0.003
  max_integration_distance_m: 0.5
  weighting: constant
```

### 4.3 Collision plugin policy

Defined in `config/semantic_classes.yaml`:

```yaml
classes:
  stem:
    behavior: hard
    padding_m: 0.005
  branch:
    behavior: hard
    padding_m: 0.008
  leaf:
    behavior: soft
    cost_weight: 50.0            # cost added per metre of penetration
    padding_m: 0.0
    max_penetration_m: 0.02      # > 0.02 m brush-through = treat as hard
  target:
    behavior: attractor
    attract_radius_m: 0.10
    attract_weight: -20.0        # negative cost pulls planner inward
```

## 5. v0 vs v1 implementation

### v0 — N parallel nvblox instances (pragmatic, ~3 weeks)

- Instantiate four `isaac_ros_nvblox` nodes with separate namespaces.
- Each one consumes the masked depth stream for its class only.
- 4× GPU memory and compute overhead vs single-class baseline.
- **Acceptable on Orin AGX 64 GB**: budgeted ~8 GB total nvblox memory in
  [`../../../../ROADMAP.md` §5](../../../../ROADMAP.md#5-compute-budget-on-jetson-orin-agx-64-gb).
- This is the **publishable workshop-paper baseline**.

### v1 — single multi-class nvblox fork (research contribution, ~4 weeks)

- Fork `isaac_ros_nvblox` and add a per-voxel `class_id` field (uint8) alongside
  the existing TSDF distance + weight.
- Modify the integration kernel to write the *highest-confidence* class per voxel
  (or maintain a small per-voxel histogram for robustness).
- Add a `class_id` query API; the MoveIt plugin maps `class_id → behavior` via
  the YAML.
- **This is the contribution to release upstream and publish at ICRA Agri-Robotics.**

The collision plugin's public API is identical between v0 and v1 — the swap is
invisible to MoveIt.

## 6. ROS 2 topics summary

| Topic | Type | Producer | Consumer | Notes |
|---|---|---|---|---|
| `/stem_grasp/semantic_label` | `sensor_msgs/Image` mono8 | `segmentation_node` | `class_demux_node` | Phase 0 dep |
| `/scene_repr/mask/<class>` | `sensor_msgs/Image` mono8 | `class_demux_node` | nvblox-class node | 4 topics |
| `/scene_repr/depth/<class>` | `sensor_msgs/Image` 16UC1 | `class_demux_node` | nvblox-class node | mask-gated |
| `/scene_repr/esdf/<class>` | custom (TBD; likely `OccupancyGrid` slice or custom 3D msg) | nvblox | `semantic_collision_plugin` | per-class |
| `/scene_repr/markers/<class>` | `visualization_msgs/MarkerArray` | nvblox | RViz | debug only |
| `/scene_repr/policy` | `std_msgs/String` (YAML inline) | static_publisher from yaml | `semantic_collision_plugin` | latched |

## 7. MoveIt 2 collision plugin design

The plugin lives in [`src/semantic_collision_plugin.cpp`](../src/semantic_collision_plugin.cpp).

**Pluginlib export class:** `collision_detection::CollisionEnvSemantic`.

**Required overrides** (subset of `moveit_core::CollisionEnv`):
- `checkSelfCollision(...)` — delegate to parent FCL impl.
- `checkRobotCollision(req, res, state)` — main entry point:
  1. For each link, query the four ESDFs at the link's collision shape vertices.
  2. Per class, apply behavior:
     - `hard` → if distance < padding, mark collision.
     - `soft` → accumulate `cost_weight × max(0, penetration)`.
     - `attractor` → accumulate `attract_weight × max(0, attract_radius - distance)`.
  3. Populate `res.collision`, `res.distance`, and (for the attractor) a custom
     `res.contacts` cost annotation.
- `distanceRobot(req, res, state)` — return the **minimum** distance to any hard class.

**Allocator registration:** via `class_loader::class_loader` macros in
[`plugin_description.xml`](../plugin_description.xml).

**Activation:** in the MoveIt config's `move_group.launch.py`, override:
```yaml
move_group:
  collision_detector: "Semantic"
```

## 8. Benchmark plan (Phase 1 exit gate)

20 cluttered-plant scenes recorded as ROS 2 bags (replay-able offline).

| Scenario | Octomap baseline | v0 (per-class nvblox) | v1 (fork) |
|---|---|---|---|
| Plan success rate | — | — | — |
| Mean planning time | — | — | — |
| Mean voxel update time | — | — | — |
| Number of false collisions on leaves | — | — | — |
| Number of missed collisions on stems | — | — | — |

Exit gate: **≥ 30 % reduction in "no plan found" failures** vs Octomap baseline.

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| nvblox's semantic channel API only supports COCO-style classes, not custom | Medium | The class_demux_node treats nvblox as a stateless depth integrator (one instance per class, no semantic channel needed). v1 fork sidesteps this entirely. |
| 4× nvblox memory exceeds Orin AGX budget | Low | Per-class voxel size is tuned (3 mm for stem, 1 cm for leaf); total ≈ 6–8 GB. |
| MoveIt 2 collision plugin ABI changes between minor versions | Low | Pin to MoveIt 2 Humble release; CI test against `humble-binary`. |
| Plugin query latency dominates planning | Medium | Cache ESDF samples per OMPL node; batch queries via SIMD or CUDA in v1. |
| Segmentation node not yet producing label image (Phase 0 blocker) | High | class_demux_node accepts the legacy two-topic input (separate masks) as a fallback. |

## 10. Phase 1 task breakdown

Mirrors the IDs in [`../../../../PROGRESS.md`](../../../../PROGRESS.md):

- **P1.1** Stand up nvblox on the workstation (Docker / native).
- **P1.2** Implement `class_demux_node` (Python — fast iteration).
- **P1.3** Launch v0 — four parallel nvblox instances + class_demux.
- **P1.4** Implement `semantic_collision_plugin` (C++ skeleton already in this package).
- **P1.5** Wire plugin into MoveIt 2 via collision_detector override.
- **P1.6** Record 20-scene benchmark set.
- **P1.7** Run baseline (Octomap) vs v0 comparison.
- **P1.8** Draft workshop paper.
- **P1.9** (Optional) v1 nvblox fork with per-voxel class label.

## 11. Open questions

1. **Label image vs per-class masks** — which does the segmentation node port to?
   The label image is simpler downstream but requires a small refactor in the
   segmentation node. Defer decision until Phase 0 segmentation porting starts.

2. **ESDF query API** — does the v0 nvblox node expose a synchronous query
   service, or do we have to subscribe to a published ESDF and cache it
   in the collision plugin? Investigation TODO in P1.1.

3. **Soft-cost integration with OMPL** — OMPL's standard cost interfaces aren't
   trivial to plumb custom costs through. Confirm whether CHOMP (gradient-based)
   or a MoveIt cost wrapper is the cleaner integration point.
