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
| `candidates.py` | Reachable camera poses looking at the target: arm joints sampled (0.15 rad from the limits), base pose solved so the optical axis hits the target at 0.2–0.45 m, or arm-only with the target within 10° of the axis; collision-checked against the map (occupied and unknown block, as the MPC's no_entry policy would); `move_time` |
| `nbv.py` | `ViewPlanner`: gains per candidate (target pixels, joint Fisher gain, corridor voxels cleared, twin point), normalised; policies `fixed_ring`, `random`, `geometry` (unknown space seen, unweighted), `target` (JEPA-only: target pixels), `fused` (weights by how far each required term is from being met), minus 0.02 per second of motion |
| `scene.py` | Simulated plant (scout_piper_whole_body_mpc plant rows, target moved to the middle of a branch, leaves in front of it): exact ray casting of depth + class labels, D405-like sensor (noise, thin-structure dropout) |
| `sim.py` | One simulated episode: scan pose, first view, then plan / move / render / integrate / refit until the requirements are met; teleport or the whole-body MPPI as executor (P4.3) |
| `vggt.py` | P4.1 helpers: `SparseViewBatcher` (keyframes by baseline / angle, anchor kept), lazy VGGT wrapper (`facebook/VGGT-1B`, or any `model_fn`), metric scale from the D405 depth or the robot's camera baselines, `fuse_depth` (sensor where valid, VGGT where the sensor dropped and VGGT is confident, agreement reported) |
| `active_perception_node.py` | ROS 2 node (below) |

Why the corridor and not only the joint: in simulation one clear view at
0.2–0.4 m fixes the joint to well under a millimetre (with the pose error,
about 0.5–1 mm). What takes views is seeing past the leaves into the space
the gripper will move through: the MPC's `unknown_policy: no_entry` cannot
enter unknown space, so an approach corridor left unknown blocks the grasp.

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

30 scenes (2026-10-12, teleport executor; medians over scenes):

| policy | success | views | time [s] | views to success | axis error [mm] | corridor unknown | min. clearance to stems [cm] | leaf penetration |
|---|---|---|---|---|---|---|---|---|
| fixed ring | 14/30 | 3.0 | 19.2 | 4.0 | 0.94 | 4.8 % | 2.6 | 0 |
| random | 21/30 | 6.0 | 43.6 | 3.0 | 1.03 | 3.2 % | 2.2 | 0 |
| geometry-only | 24/30 | 4.0 | 19.6 | 3.0 | 0.61 | 3.6 % | 2.0 | 0 |
| target-only (JEPA-only) | 18/30 | 3.0 | 19.4 | 2.0 | 0.70 | 4.7 % | 2.2 | 0 |
| **fused** | **23/30** | **2.5** | **13.2** | **2.0** | 0.82 | 3.5 % | 4.0 | 0 |

The fused policy needs a median 2.5 views and 13 s, against 4 views and 20 s
for geometry-only NBV, at one success fewer (23 vs 24 of 30): the exit
criterion's "fewer views or lower acquisition time" holds, "equal grasp
readiness" within one scene. The fixed ring runs out of reachable ring
points (it stops after its six), random wanders (6 views, 44 s) and
target-only views fix the joint fastest but leave the corridor unknown. The
misses of all policies are mostly corridors that stay above 5 % unknown
within 8 views (space between dense leaves).

Found on the way (each made the predicted gains disagree with rendering):
an unknown voxel counted as occupied with p = 0.5 or 0.15 blocked every line
of sight through the 20-30 unknown voxels left after one view (the canopy's
fill fraction, 0.02, is the right prior); a corridor voxel only clears with
a depth return behind it; independent per-point noise made one view's
pose error look averaged away (σ 0.3 mm with 2-4 mm true error) and a
per-point inflation overcorrected (σ up to 5 mm with 1 mm error), the
per-view fusion is consistent (true error / σ ≈ 1.1-1.4); weighting terms by
U instead of by how far they are from being met spent views on a joint
already known.

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

- **VGGT on the GPU / Orin** (P4.1): the wrapper is tested with a fake model
  only; inference time, memory and depth quality on thin stems are
  MODULE_TASKS.md A14 (workstation) and B8 (Orin). The fused depth is not yet
  wired into the semantic map (a remap of the aligned depth topic once A14
  shows it helps).
- **Robot runs** (P4.3, P4.4): MODULE_TASKS.md C10.
- The simulation's leaves are flat discs and the sensor model is a
  hypothesis (dropout below 6 px, σ_z); the noise parameters should come from
  the D405 on the robot (C10 step 1).
