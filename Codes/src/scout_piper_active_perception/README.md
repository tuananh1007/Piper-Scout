# scout_piper_active_perception — uncertainty-driven next view (Phase 4)

The wrist camera sees a grasp target that is thin (a peduncle or branch a few
millimetres across), often behind leaves, and whose depth the D405 drops when
it projects narrower than a few pixels. This package decides **whether** the
robot needs another view before the grasp and **where** the camera should go,
among poses the whole-body robot can reach, and hands the view to the
whole-body MPC as a pose goal.

ROADMAP.md Phase 4: U_total = α·U_JEPA + β·U_geometry + γ·U_twin, ablations
fixed ring / random / geometry-only / JEPA-only / fused, exit criterion
"fewer views or lower acquisition time at equal or better grasp performance".

## Pieces

| Module | What it does |
|---|---|
| `stem_joint.py` | Stem joint (grasp-point axis) as a cylinder fitted to the target's points (Gauss-Newton, radius prior, trimming); Fisher information of the axis per point (depth noise σ_z = σ₀ + k₂z² along the ray, one pixel across it, projected on the surface normal); per-view camera pose error mapped to the joint parameters and fused per view, I = I_prior + Σ_v (I + I_v P_v)⁻¹ I_v; predicted information of a candidate camera (surface facing it, in the image, pixel count, the sensor's thin-structure survival, transmittance through the map); `axis_sigma` (worst 1-σ lateral axis uncertainty over ±3 cm) |
| `grid.py` | Occupancy view of the semantic scene (from the CPU `SemanticVoxelMap` or the `/scene_repr/distance_field` snapshot): occupied (leaves included), unknown; transmittance with an unknown voxel occupied with the canopy fill fraction (0.02); `voxel_gain`: expected corridor voxels a camera clears (reached *and* a depth return behind them within range); unknown share of the approach corridor without the one-voxel skin in front of surfaces |
| `uncertainty.py` | U_joint = σ/(σ + 1 mm), U_corridor = f/(f + 5 %), U_geometry = max, U_JEPA from `/piper_jepa/target_state` (1 lost or uninitialised, 0.75 occluded, 1 − confidence tracking), U_twin = 1 − `/plant_twin/state` confidence while touching; 0.5 = requirement just met. A view is needed while a required term is above 0.5; U_total is reported and ranks |
| `candidates.py` | Reachable camera poses looking at the target: arm joints sampled (0.15 rad from the limits, shoulder-to-wrist within the MPC's 0.36 m reach margin), base pose solved so the optical axis hits the target at 0.2–0.45 m, or arm-only with the target within 10° of the axis; collision-checked against the map (occupied and unknown block, as the MPC's no_entry policy would); `move_time`, `axis_turn` |
| `nbv.py` | `ViewPlanner`: gains per candidate (target pixels, joint Fisher gain, corridor voxels cleared, twin point), normalised; policies `fixed_ring`, `random`, `geometry` (unknown space seen, unweighted), `target` (JEPA-only: target pixels), `fused` (weights by how far each required term is from being met), minus 0.02 per second of motion and 0.15 per radian of approach-axis turn |
| `scene.py` | Simulated plant (scout_piper_whole_body_mpc plant rows, target moved to the middle of a branch, leaves in front of it): exact ray casting of depth + class labels, D405-like sensor (noise, thin-structure dropout) |
| `sim.py` | One simulated episode: scan pose, first view, then plan / move / render / integrate / refit until the requirements are met; teleport or the whole-body MPPI as executor (P4.3) |
| `vggt.py` | P4.1 helpers: `SparseViewBatcher` (keyframes by baseline / angle, anchor kept), lazy VGGT wrapper (`facebook/VGGT-1B`, or any `model_fn`), metric scale from the D405 depth or the robot's camera baselines, `fuse_depth` (sensor where valid, VGGT where the sensor dropped and VGGT is confident, agreement reported), `DepthFiller` (fills the current frame only while the camera is still at the newest reconstructed keyframe: the scene integrators pair depth with the newest masks) |
| `active_perception_node.py` | ROS 2 node (below) |
| `fused_depth_node.py` | ROS 2 node: keyframes from the wrist camera, VGGT in a worker thread, `/active_perception/fused_depth` (32FC1, the depth frame's header: holes filled at the keyframe, sensor depth otherwise), `~/status`, `~/reconstruct`; `model: stand_in` inpaints instead (tests) |

Why the corridor and not only the joint: in simulation one clear view at
0.2–0.4 m fixes the joint to well under a millimetre (with the pose error,
about 0.5–1 mm). What takes views is seeing past the leaves into the space
the gripper will move through: the MPC's `unknown_policy: no_entry` cannot
enter unknown space, so an approach corridor left unknown blocks the grasp.

The fake-robot check (`scout_piper_bringup/scripts/active_perception_chain_check.py`)
runs the node, the MPC and moveit_servo on the fake arm and base with a
simulated plant rendered from the camera pose in TF: scan pose and two planned
views reached in 3.5-6.6 s, the target in view from both, stem axis error
0.56 mm (σ 0.52 mm), uncertainty falling.

## Node

```
ros2 launch scout_piper_active_perception active_perception.launch.py [auto_publish:=true] [policy:=fused]
ros2 service call /active_perception/plan_view std_srvs/srv/Trigger
# the MPC follows the published view:
ros2 launch scout_piper_whole_body_mpc whole_body_mpc.launch.py execute:=true \
    field_topic:=/scene_repr/distance_field goal_pose_topic:=/active_perception/next_view
```

| Direction | Topic | Type | Role |
|---|---|---|---|
| in | `/scene_repr/distance_field` | SemanticDistanceField | occupied / unknown voxels |
| in | `/piper_jepa/target_point` | PointStamped | grasp target (Stage A memory); a jump > 3 cm resets the joint estimate |
| in | `/piper_jepa/target_state` | String (JSON) | U_JEPA |
| in | `/plant_twin/state` | String (JSON) | U_twin (while touching) |
| in | `/stem_grasp/filtered_cloud` | PointCloud2 | target stem points, cropped 5 cm around the target; a new view only after 3 cm or 8° of camera motion |
| in | `/camera/color/camera_info`, `/odom`, `/joint_states`, TF | | intrinsics, whole-body state, hand-eye (`piper_link6` → `camera_color_optical_frame`) |
| out | `/active_perception/status` | String (JSON) | U terms, `needs_view`, `unmet`, joint estimate, last choice |
| out | `/active_perception/next_view` | PoseStamped | TCP pose of the view (odom), z = approach axis (the MPC's goal-pose convention) |
| out | `/active_perception/next_view_camera` | PoseStamped | camera pose of the view (RViz) |
| srv | `~/plan_view`, `~/reset` | Trigger | plan now; forget the joint estimate |

Parameters in `config/active_perception.yaml`. On the robot the required terms
default to joint, corridor and JEPA.

## Simulated benchmark (P4.3 / P4.4)

```bash
cd Codes/src/scout_piper_active_perception
PYTHONPATH=.:../scout_piper_whole_body_mpc:../scout_piper_scene_repr/python \
    python3 benchmarks/nbv_benchmark.py --scenes 30 --jobs 4 --out nbv.jsonl
#   --executor mpc   moves through the whole-body MPPI (slower)
```

Success = true axis error ≤ 2 mm and approach corridor ≤ 5 % unknown; every
policy starts from the same scan pose and first view, uses the same stopping
rule (required terms met), at most 8 views. Per-view camera pose error 0.5 mm
/ 0.1°, depth noise 0.5 mm + 0.002·z², thin-structure dropout below 6 px.

30 scenes (2026-10-12, teleport executor, medians over scenes). Views are
whole-body poses inside the MPC's reach margin (shoulder-to-wrist ≤ 0.36 m),
scored with 0.02 per second of motion and 0.15 per radian of approach-axis
turn:

| policy | success | views | time [s] | views to success | axis error [mm] | corridor unknown | min. clearance to stems [cm] | leaf penetration |
|---|---|---|---|---|---|---|---|---|
| fixed ring | 19/30 | 3.5 | 23.7 | 4.0 | 1.09 | 4.0 % | 3.1 | 0 |
| random | 18/30 | 7.5 | 40.0 | 3.0 | 0.86 | 3.7 % | 2.2 | 0 |
| geometry-only | 13/30 | 8.0 | 21.0 | 4.0 | 1.27 | 5.5 % | 2.6 | 0 |
| target-only (JEPA-only) | 8/30 | 8.0 | 29.9 | 3.0 | 0.66 | 6.2 % | 2.4 | 0 |
| **fused** | **17/30** | **4.5** | **14.5** | **2.0** | 0.79 | 4.3 % | 4.1 | 0 |

Upper bound, views at any valid arm pose (`--no-reach-margin`; the real MPC
does not reach a stretched arm): fixed ring 24/30 (4 views, 34.5 s), random
21/30 (5, 40.9 s), fused 18/30 (5, 12.7 s), target-only 16/30, geometry-only
15/30.

Reading: the fused policy has the lowest acquisition time (14.5 s against
23.7 s for the fixed ring, 21-40 s for the others) and needs the fewest views
to a ready scene (median 2), but it is not more often ready than the fixed
ring (17 vs 19 of 30, within the scatter of 30 scenes). The exit criterion's
"lower acquisition time" holds in simulation; "equal or better grasp
readiness" is not shown. In these scenes the ring's wide side views are what
clears the approach corridor, and most misses of every policy are corridors
left above 5 % unknown after 8 views: a voxel clears only with a depth return
from behind it, and from the folded-arm poses the MPC reaches there is often
nothing within the D405's range behind the corridor. The stem joint itself is
fixed within one or two good views by every policy except random. Lowering the
turn penalty (0 or 0.05) gave 16/30 with 6-6.5 views; aiming views at corridor
points instead of the target did not help on 10 scenes. The real test is the
ablation on plants (MODULE_TASKS.md C10).

An earlier run that ignored the reach margin and the turn penalty (and drew
fewer candidates) gave fused 23/30 in 2.5 views; it overstated what the MPC
can execute: on the fake robot the MPC stalled 25-35 cm short of such views.

Found on the way (each made the predicted gains disagree with rendering):
an unknown voxel counted as occupied with p = 0.5 or 0.15 blocked every line
of sight through the 20-30 unknown voxels left after one view (the canopy's
fill fraction, 0.02, is the right prior); a corridor voxel only clears with a
depth return behind it; independent per-point noise made one view's pose
error look averaged away (σ 0.3 mm with 2-4 mm true error) and a per-point
inflation overcorrected (σ up to 5 mm with 1 mm error), the per-view fusion is
consistent (true error / σ ≈ 1.1-1.4); weighting terms by U instead of by how
far they are from being met spent views on a joint already known.

## Tests

```bash
cd Codes/src/scout_piper_active_perception && python3 -m pytest test -q
```

Rendering against the capsule surface, sensor dropout, transmittance and the
corridor gain on hand-made grids, skin-free unknown share, joint fit and
σ falling with views, the pose error capping one view's information, predicted
pixels against rendered ones, candidates looking at the target, move time,
uncertainty maps, the view batcher and depth fusion with a fake VGGT, and a
short episode. The node runs end to end on the fake robot in
`scout_piper_bringup/scripts/active_perception_chain_check.py`
(`hardware_free_checks.sh`, check `active_perception`).

## Open

- **VGGT on the GPU / Orin** (P4.1): the wrapper and `fused_depth_node` are
  tested with a stand-in model only (in ROS: 4 keyframes, the stem's missing
  depth filled, scale recovered); inference time, memory and depth quality on
  thin stems are MODULE_TASKS.md A14 (workstation) and B8 (Orin). The semantic
  map takes the fused depth by setting its `depth_topic` to
  `/active_perception/fused_depth` once A14 shows it helps.
- **Robot runs** (P4.3, P4.4): MODULE_TASKS.md C10.
- The simulation's leaves are flat discs and the sensor model is a
  hypothesis (dropout below 6 px, σ_z); the noise parameters should come from
  the D405 on the robot (C10 step 1).
