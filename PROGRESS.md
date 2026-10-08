# Piper + Scout Project — Detailed Progress Tracker

> **Living document.** Update this file when any task is completed, blocked,
> or rescoped. Each phase has its own section with task-level granularity.
> The companion [`ROADMAP.md`](ROADMAP.md) is the high-level strategic plan;
> this file is the day-to-day execution log.

**Last updated:** 2026-10-08 (P0.4.13 stem_grasp stepwise final approach to AT_GRASP; P0.4.12 stem_grasp image-based servo toward the gripper axis, six servo bugs, camera placeholder orientation; whole-body MPC reach margin; P0.4.11 stem_grasp reach through the whole-body MPC, three stem_grasp porting bugs; P3A.11 whole-body MPC execute mode end to end on fake drivers, stop on exit; INSTALL.md TODOs resolved: D405 URDF, realsense-ros 4.58.4, CPU-only build closure, Nav2 blocker; P0.5 moveit_servo wired through the servo bridge; P0.3.13 arm command isolation; P3B.1–P3B.7 Piper-JEPA Stage B/C on a synthetic world; P3A.6, P3A.9 whole-body MPC; P1.7.7 MoveIt semantic collision plugin)

## Legend

- ☐ `pending` — Not started
- ◐ `in_progress` — Active work
- ☑ `done` — Completed and validated
- ✗ `blocked` — Waiting on something (note the blocker)
- ⊝ `deferred` — Intentionally postponed (note to which phase)

---

## Phase 0 — ROS 2 Humble migration + Scout integration

**Start:** 2026-05-16  ·  **Target finish:** 2026-07-15  ·  **Status:** ◐ in_progress

### P0.1 — Workspace bootstrap

| ID | Task | Status | Notes |
|---|---|---|---|
| P0.1.1 | Create `Piper_Scout_ws/Codes/` colcon workspace skeleton | ☑ | 2026-05-16 |
| P0.1.2 | Write `repos.yaml` vcs-import manifest for upstream packages | ☑ | 2026-05-16; pins humble/main branches |
| P0.1.3 | Write workspace `README.md` + `PHASE0_CHECKLIST.md` | ☑ | 2026-05-16 |
| P0.1.4 | Add `.gitignore`, `build_workspace.sh`, `setup_env.sh` helpers | ☑ | 2026-05-16 |
| P0.1.5 | Run `vcs import src < repos.yaml` and verify all four repos clone | ☑ | 2026-05-17 — 4/4 cloned (piper_ros@humble, scout_ros2@humble, scout_nav2@main, realsense-ros@ros2-development) |
| P0.1.6 | `rosdep install --from-paths src --ignore-src -r -y` | ⊝ | Pre-baked in Docker image; not needed at host level |
| P0.1.7 | First successful `colcon build --symlink-install` | ☐ | Will run inside dev container after `./docker/build_dev.sh` |
| P0.1.8 | `vcstool` + `colcon` installed via `pip3 --user` | ☑ | 2026-05-17 |
| P0.1.9 | Docker + nvidia-container-toolkit install script | ☑ | 2026-05-17 — `Codes/scripts/install_docker_nvidia.sh` |
| P0.1.10 | Phase 0 dev Dockerfile + compose + build script | ☑ | 2026-05-17 — `Codes/docker/{Dockerfile.dev,compose.dev.yml,build_dev.sh,README.md}` |
| P0.1.11 | Fix upstream package-name references after vcs import | ☑ | 2026-05-17 — `piper_humble`, `piper_with_gripper_moveit`, `scout_nav2/nav2.launch.py` |

### P0.2 — Unified URDF (scout_piper_description)

| ID | Task | Status | Notes |
|---|---|---|---|
| P0.2.1 | Package skeleton (package.xml, CMakeLists.txt) | ☑ | ament_cmake |
| P0.2.2 | `scout_piper.urdf.xacro` composing scout_description + piper_description + realsense | ☑ | Verify upstream paths after vcs import. 2026-10-07: camera switched to `sensor_d405` (the mounted D405); `piper_gazebo` declared as a dependency (the arm xacro reads its controller config) |
| P0.2.3 | Confirm no TF name collisions between Piper's `base_link` and Scout's `base_link` | ☑ | 2026-05-17 — forked piper xacro into `scout_piper_description/urdf/_piper_arm.xacro`. All 10 links + 10 joints + 8 transmissions + 8 gazebo refs prefixed via `scripts/fork_piper_arm.py` (re-runnable when upstream updates). Unified URDF now: Scout → piper_mount_link → piper_base_link → … → piper_link6 → camera_link (eye-in-hand) |
| P0.2.4 | `view_robot.launch.py` + RViz config — visualize unified model | ☑ | Needs upstream meshes |
| P0.2.5 | Visual sanity: arm reaches expected workspace from Scout top plate | ☐ | After P0.2.3. 2026-10-08 — placeholder camera pose fixed: with `cam_rpy` 0 the D405 looked along `piper_link6` x, across the gripper; now `0 -1.5708 0` points it along the approach axis (link6 z). SRDF collision matrix regenerated for it (camera vs gripper links now checked); still a placeholder until P0.2.6 |
| P0.2.6 | Add hand-eye TF (camera → link6) from existing calibration_samples.yaml | ☐ | Port `calibration_transform.py` from the ROS 1 `piper_ros` workspace on the lab machine (not in this repository) |
| P0.2.7 | Re-do hand-eye calibration on integrated rig (arm on Scout) | ☐ | Required if mount differs from ROS 1 setup |

### P0.3 — Bringup integration (scout_piper_bringup)

| ID | Task | Status | Notes |
|---|---|---|---|
| P0.3.1 | Package skeleton | ☑ | |
| P0.3.2 | `system.yaml` mirroring all ROS 1 launch args | ☑ | Source of truth for tuning |
| P0.3.3 | `full_system.launch.py` with toggle args per-subsystem | ☑ | Phase 0 conservative defaults |
| P0.3.4 | RViz config for live debugging | ☑ | |
| P0.3.5 | Verify arm driver alone launches and arm is enable-able | ☐ | Requires hardware + CAN |
| P0.3.6 | Verify Scout base driver alone launches and `/cmd_vel` moves wheels | ☐ | Requires hardware + CAN1 |
| P0.3.7 | Verify both drivers run simultaneously without CAN contention | ☐ | |
| P0.3.8 | RealSense launches with aligned depth + point cloud | ☐ | |
| P0.3.9 | MoveIt 2 demo plans a canned home→pose motion | ☐ | Run separately: `ros2 launch piper_with_gripper_moveit demo.launch.py` |
| P0.3.10 | All six subsystems running concurrently with stable TF tree | ◐ | 2026-05-17 — verified for: RSP (unified URDF), stem_grasp (pipeline/segmentation/pointcloud), RViz, jsp_gui. Pending: arm/base/camera (need hardware). Phase 0 exit gate. |
| P0.3.11 | Hardware-free defaults — `full_system.launch.py` boots without CAN/USB | ☑ | 2026-05-17 — bringup_arm/base/camera default false; jsp_gui auto-spawned for URDF sliders |
| P0.3.12 | MoveIt include separated due to URDF conflict | ☑ | 2026-05-17 — MoveIt demo brings its own RSP that fights our unified URDF; documented; Phase 3A unifies via cuMotion |
| P0.3.13 | Arm command isolation: driver commands off `/joint_states`, feedback relayed as `piper_joint1..8`, no sliders with the arm | ☑ | 2026-10-07 — upstream launch made the driver execute `/joint_states` (sliders → all-zero pose at full speed); `full_system.launch.py` now starts the driver with `arm_command_topic` (default `/piper/joint_cmd`) + `piper_joint_state_relay.py`; launch + relay tests and an rclpy end-to-end check, not yet on hardware |

### P0.4 — stem_grasp ROS 2 port

| ID | Task | Status | Notes |
|---|---|---|---|
| P0.4.1 | Package skeleton (ament_python, setup.py, entry_points) | ☑ | |
| P0.4.2 | `pipeline_node.py` scaffold: params, subs, pubs, timers, state machine | ☑ | Algorithm bodies filled in by P0.4.8–P0.4.13 (2026-10-08: reach, servo and final approach hardware-free) |
| P0.4.3 | `segmentation_node.py` scaffold | ☑ | Filled in by P0.4.15–P0.4.16 (2026-05-17) |
| P0.4.4 | `pointcloud_node.py` scaffold | ☑ | Filled in by P0.4.17 (2026-05-17) |
| P0.4.5 | `hotkey_stop_and_zero.py` ported (working) | ☑ | Zero-twist publish; service call still TODO |
| P0.4.6 | `core.py` (servo math) — port classes from ROS 1 verbatim | ☑ | 2026-05-17 — full port: 3 classes (`StemVelocityObserver`, `OnlineJacobianEstimator`, `FullAdaptiveServoController`) + 3 helpers |
| P0.4.7 | `moveit_planner.py` — port to `moveit_py` API | ☑ | 2026-05-17 — wrapper ready; runtime requires `moveit_py`, which has no Humble binary (build moveit2 from source; see `Codes/docker/Dockerfile.dev`) |
| P0.4.8 | `pipeline_node._on_stem_mask` — centroid extraction | ☑ | 2026-05-17 — centroid of all mask pixels via numpy where (largest-blob selection not implemented) |
| P0.4.9 | `pipeline_node._on_target_point` — frame conversion + cache | ☑ | 2026-05-17 — TF-based transform to planning_frame |
| P0.4.10 | `pipeline_node._outer_loop` — skeleton + candidate selection | ☑ | 2026-05-17 — uses core.skeletonize_plant_points + extract_main_stem. 2026-10-08 — three porting bugs fixed, found by the hardware-free grasp chain: clouds are read with `read_points_numpy` (Humble's `read_points` returns a structured array and the old code crashed on the first cloud); the main stem is searched along z of `planning_frame` (`skeleton_vertical_axis: 2`; the ROS 1 default was the optical-frame y, which found 2–4 points instead of the stem); candidates approach from the arm's side (`approach_hint`; `cross(stem_dir, x)` meant "from the camera" only in the optical frame) |
| P0.4.11 | `pipeline_node._outer_loop` — plan_and_execute + state transition | ☑ | 2026-10-08 — through the whole-body MPC instead of MoveIt (`reach_executor: whole_body_mpc`): SCANNING → REACHING (pre-grasp pose → `/whole_body_mpc/goal_pose`) → MPC "reached" for 1 s → cancel → SERVOING (`reach_handoff.py`). Hardware-free (`grasp_chain_check.py`): reach in 7–12 s, TCP within 0.2–1 cm of the pre-grasp, approach axis within 15°, MPC idle and silent after the handoff. The moveit_py path stays unwired; not run on hardware |
| P0.4.12 | `pipeline_node._inner_loop` — visual servo step | ☑ | 2026-05-17 — drives `core.FullAdaptiveServoController` and publishes TwistStamped. 2026-10-08 — servoes the stem onto the gripper approach axis (`servo_geometry.py`): the desired pixel is the projection of the axis point at the target's depth (from TF; ROS 1 used the image centre), and the measured pixel is the stem column at the target's image row. Fixed: twists were sent with no frame although the velocity is in the camera optical frame; the depth was a constant; the observer started at pixel (0, 0); the image Jacobian used raw u instead of u − cx; the loop stepped at 100 Hz on 15 Hz masks with a fixed 10 ms observer step; sway was not measured against the camera's own motion (now from TF), which halved the gain every other step. The servo resets per phase and stops commanding on a mask older than 0.3 s. `servo_lambda_inf` 0.07 → 0.5 (moveit_servo realises ~55 % of a commanded velocity here). Hardware-free (`grasp_chain_check.py`, four stem positions): 33–65 px → 0.8–5 px in 15 s, gripper axis 0.04–0.25 cm from the grasp point, no servo halt. Not run on hardware; gains untuned |
| P0.4.13 | Iterative approach state machine | ☑ | 2026-10-08 — new design, since the ROS 1 source (lines ~1070–1220) is not in this repository; it keeps the ROS 1 `approach_*` names (`approach.py`). With `approach_enabled` the reach handoff goes to APPROACHING. The servo keeps the stem on the gripper axis, and once the error stays ≤ 8 px for `approach_settle_sec`, the gripper advances one `approach_step` at 2 cm/s, re-aligning between steps. It ends in AT_GRASP within `approach_distance_tolerance`, or on contact force, and in ABORTED on a start beyond `approach_target_distance`, `approach_max_steps`, a lost mask or a timeout. Hardware-free (`grasp_chain_check.py` with the approach on, seven runs at four stem positions): AT_GRASP after 3 steps in 17–24 s, TCP 0.85–0.97 cm short of the grasp point along the axis and 0.06–0.84 cm off it, the arm holding. Off by default: no force sensor on the Piper, and nothing closes the gripper yet. Not run on hardware |
| P0.4.14 | Multi-view capture ring + ICP merge | ⊝ | Deferred to Phase 4 (replaced by NBV+VGGT) |
| P0.4.15 | `segmentation_node` YOLO seg path | ☑ | 2026-05-17 — full port |
| P0.4.16 | `segmentation_node` Grounded-SAM + target_caption path | ☑ | 2026-05-17 — full port incl. context prompts, union mode, target backprojection |
| P0.4.17 | `pointcloud_node` mask-gated cloud filtering + `/static_cloud_out` | ☑ | 2026-05-17 — full port; output_frame configurable |
| P0.4.18 | `pipeline_node` smoke test (no hardware) | ☑ | `test_pipeline_smoke.py` (expanded 2026-05-17 with servo + skeleton tests) |
| P0.4.19 | End-to-end smoke: pipeline runs on bag file of ROS 1 data | ☐ | Use `rosbags` converter |

### P0.5 — moveit_servo wiring

The ROS 1 stack published twist commands to a topic with no subscriber.
ROS 2 has `moveit_servo` as a first-class node — we actually wire it up.

| ID | Task | Status | Notes |
|---|---|---|---|
| P0.5.1 | Add `moveit_servo` node to bringup launch | ☑ | 2026-10-07 — `bringup_servo:=true`: `servo_node` + `piper_servo_bridge` (starts disabled; joint-limit, 0.1 rad step and 30 % speed clamps; holds the measured gripper) → `/piper/joint_cmd`; unified SRDF `config/moveit/scout_piper.srdf` (collision matrix computed like the Setup Assistant, cross-checked with upstream) |
| P0.5.2 | Servo YAML config tuned for Piper (vel/accel limits, scaling) | ◐ | 2026-10-07 — `config/moveit/servo.yaml`: speed units, 50 Hz, 0.25 s timeout, singularity thresholds 45 / 100 from the Piper Jacobian over the URDF (Panda's 17 / 30 halt the Piper almost everywhere); speeds and thresholds still to tune on the robot |
| P0.5.3 | Verify a manually-published TwistStamped actually moves the EE | ◐ | 2026-10-07 — verified on the fake arm (`fake_arm:=true` + `servo_chain_check.py`: +z twist raises the flange, joint jog, latches, hold; ~63 % of the commanded displacement in that loop); hardware pending |
| P0.5.4 | Hot-key stop disables the servo bridge | ☑ | 2026-10-07 — `hotkey_stop_and_zero` `x`: zero twists + `/piper_servo_bridge/enable false`; verified on the fake arm; base not covered |

### P0.6 — Nav2 stand-up (basic, defer fancy mapping to later)

| ID | Task | Status | Notes |
|---|---|---|---|
| P0.6.1 | Drop in `scout_nav2` launches via include | ☑ | Wired in `full_system.launch.py` (default off) |
| P0.6.2 | Verify Scout drives 2 m to a goal pose with arm folded | ☐ | Blocked: `scout_nav2` is configured for an Ouster 3D lidar (`/ouster/points`, `/ouster/scan`, `/odometry`) and a site map, and `full_system.launch.py` starts its simulation configuration (`simulation` defaults to true). Needs a scan source on the base, a map and parameters (INSTALL.md 10.8) |
| P0.6.3 | `goal_pose` rviz tool functional | ☐ | |

### P0.7 — Phase 0 regression test

| ID | Task | Status | Notes |
|---|---|---|---|
| P0.7.1 | Capture a "scan → candidate → plan" episode on a static plant in ROS 1 | ☐ | Reference behavior |
| P0.7.2 | Replay the same scene in ROS 2 stack, compare candidate poses | ☐ | Tolerance: 2 cm position, 5° orientation |
| P0.7.3 | Document any regressions in this file | ☐ | |

### P0 exit criteria

- All checked tasks in P0.3, P0.5, P0.6, P0.7 are ☑.
- `scout_piper_bringup` launches the full system in < 10 s wall time on the workstation.
- A static plant produces the same grasp candidate poses (within tolerance) in ROS 2 vs ROS 1.
- Hand-eye calibration verified on the integrated Scout+Piper rig.

---

## Phase 1 — Semantic RGB-D scene representation + deformable plant state

**Target start:** 2026-07-15  ·  **Target finish:** 2026-09-15  ·  **Status:** ◐ in_progress — RealSense→nvblox smoke path (2026-05-18), `plant_twin` fitter + node (2026-09-29) and CPU semantic distance query (2026-10-06) done; per-class nvblox SDFs, MoveIt plugin, benchmark and hardware runs open

**Companion docs:** [`Codes/src/scout_piper_scene_repr/docs/PHASE1_DESIGN.md`](Codes/src/scout_piper_scene_repr/docs/PHASE1_DESIGN.md)

### 1A — Semantic nvblox (`scout_piper_scene_repr`)

#### P1.0 — Scaffolding (ahead of schedule, 2026-05-17)

| ID | Task | Status | Notes |
|---|---|---|---|
| P1.0.1 | `scout_piper_scene_repr` package skeleton (ament_cmake C++ + Python) | ☑ | 2026-05-17 |
| P1.0.2 | Phase 1 design doc | ☑ | 2026-05-17 — `docs/PHASE1_DESIGN.md` |
| P1.0.3 | `class_demux_node.py` (merged + separate input modes) | ☑ | 2026-05-17 — fully functional |
| P1.0.4 | `semantic_collision_plugin` C++ skeleton + pluginlib export | ☑ | 2026-05-17 — compiles; methods return "no collision" placeholders |
| P1.0.5 | `semantic_classes.yaml` (per-class policy: hard / soft / attractor) | ☑ | 2026-05-17 |
| P1.0.6 | `nvblox_per_class.yaml` (per-class voxel size + weighting) | ☑ | 2026-05-17 |
| P1.0.7 | `nvblox_semantic.launch.py` (4 nvblox instances + demux) | ☑ | 2026-05-17 — v0 design |
| P1.0.8 | Wire scene_repr into `scout_piper_bringup` (off by default) | ☑ | 2026-05-17 — `bringup_scene_repr:=false` |

#### P1.1 — nvblox integration

| ID | Task | Status | Notes |
|---|---|---|---|
| P1.1.0 | Plumbing smoke test (synthetic mask + class_demux, no nvblox) | ☑ | 2026-05-17 — `test_class_demux.launch.py` + `test_mask_publisher.py` |
| P1.1.1 | Install `isaac_ros_nvblox` on workstation | ☑ | 2026-05-18 — source build validated in dev container; `nvblox_ros`, `nvblox_msgs`, and example bringup packages discoverable after sourcing `install/setup.bash`. |
| P1.1.2 | Feed RealSense color+depth to nvblox; verify TSDF reconstruction | ☑ | 2026-05-18 — added `realsense_nvblox.launch.py` + camera-frame nvblox config; D405 aligned depth runs ~29-30 Hz and nvblox publishes TSDF/mesh/ESDF topics while allocating GPU TSDF blocks from live depth. |
| P1.1.3 | Benchmark nvblox update rate on Orin AGX target | ☐ | Goal: < 33 ms |
| P1.1.4 | Wire semantic mask channel from `segmentation_node` to nvblox | ☐ | v0 uses mask-gated depth (no fork) |

#### P1.2 — Per-class SDFs (v0)

| ID | Task | Status | Notes |
|---|---|---|---|
| P1.2.1 | Stand up 4 parallel TSDFs (stem / branch / leaf / target) | ☐ | 2026-10-07 — launch moved to `nvblox_ros` (smoke-test setup + per-class overrides); not yet run |
| P1.2.2 | Per-class inflation / padding configurable via YAML | ☑ | `semantic_classes.yaml` 2026-05-17 |
| P1.2.3 | Visualize each SDF separately in RViz | ☐ | |

#### P1.3 — MoveIt 2 collision plugin

| ID | Task | Status | Notes |
|---|---|---|---|
| P1.3.1 | Custom collision plugin reading nvblox SDFs | ◐ | 2026-10-07 — plugin reads the CPU field (P1.7.7); nvblox per-class ESDFs still to feed it |
| P1.3.2 | Hard collision for stem/branch; soft cost for leaf | ◐ | 2026-10-07 — hard classes with padding + leaf penetration cap in the plugin; graded leaf cost only in the Python query / MPC (MoveIt collision is boolean) |
| P1.3.3 | Target SDF exposed as attractor for goal generation | ☐ | Policy schema in place |

#### P1.4 — Benchmark

| ID | Task | Status | Notes |
|---|---|---|---|
| P1.4.1 | 20-scene cluttered-plant test set (recorded bags) | ☐ | |
| P1.4.2 | Compare planning success: Octomap vs ours | ☐ | Goal: ≥ 30 % failure reduction |

#### P1.5 — v1 fork (research contribution)

| ID | Task | Status | Notes |
|---|---|---|---|
| P1.5.1 | Fork nvblox with per-voxel class_id | ☐ | Workshop / ICRA Agri-Robotics contribution |
| P1.5.2 | Modify CUDA integration kernel | ☐ | |
| P1.5.3 | Benchmark v0 vs v1 GPU memory + latency | ☐ | |

#### P1.7 — Semantic scene: planner distance query (CPU v0 backend)

| ID | Task | Status | Notes |
|---|---|---|---|
| P1.7.1 | `SemanticVoxelMap`: per-class evidence, shared free space, unknown ≠ free, thin-stem-safe carving | ☑ | 2026-10-06 |
| P1.7.2 | `SemanticDistanceQuery`: signed distance, hard min + padding, gradient, validity/freshness, leaf ψ with cap, grasp-mode exclusion | ☑ | 2026-10-06 — ≤ 1 voxel error on synthetic stem |
| P1.7.3 | `other` class for non-plant depth | ☑ | 2026-10-06 — demux only fed plant classes, so pots/walls were invisible to planning |
| P1.7.4 | All depth consumers on `/camera/aligned_depth_to_color/image_raw` | ☑ | 2026-10-06 |
| P1.7.5 | `scene_query_node` (markers + status) | ☑ | 2026-10-06 — untested on hardware |
| P1.7.6 | S1/S3 runs on real thin-structure scenes; Orin timing | ☐ | |
| P1.7.8 | Robot self-filter for depth (fingers in view become `other` obstacles) | ☐ | until then exempt the finger links (ACM / `ignore_links`) |
| P1.7.7 | MoveIt plugin uses the query (replace stub) — P1.4 | ☑ | 2026-10-07 — `Semantic` = FCL + `/scene_repr/distance_field` (sphere-covered links, unknown/stale = obstacle, no field ⇒ collision); gtest with a real MoveIt model + pluginlib load; not on hardware. Also fixed: plugin was exported as an allocator, which MoveIt cannot load |

### 1B — `plant_twin` integration

#### P1.6 — Deformable leaf + stem twin (`plant_twin`)

| ID | Task | Status | Notes |
|---|---|---|---|
| P1.6.1 | Leaf model: outline+holes mesh, rigid/bend split, stretch + rest + temporal + contact residuals | ☑ | 2026-09-29 — `plant_twin/leaf.py`; point-to-plane data term needed (nearest-vertex stalls half a cell) |
| P1.6.2 | Stem model: Catmull-Rom centreline, base anchor, tip→leaf, length, smooth, stationary-before-pull | ☑ | 2026-09-29 — `plant_twin/stem.py` |
| P1.6.3 | Alternating leaf/stem fitter + synthetic pull-sequence tests | ☑ | 2026-09-29 — 7 tests, no ROS needed |
| P1.6.4 | ROS 2 node: clouds + F/T + `piper_joint7` + fingertip TF → RViz markers | ☑ | 2026-09-29 — `twin_node.py`; untested on hardware |
| P1.6.5 | Leaf outline + holes from segmentation mask contours (replace disc init) | ☑ | 2026-09-29 — `plant_twin/outline.py`, RETR_CCOMP + depth back-projection into PCA plane |
| P1.6.6 | Texture from first frame | ☑ | 2026-09-29 — per-vertex colour from the rgb cloud (`texture_from_cloud`); UV image map still open |
| P1.6.7 | Per-class clouds | ☑ | 2026-09-29 — reuse `pointcloud_node`'s `/stem_grasp/{leaf_filtered,filtered}_cloud`, TF'd to planning frame |
| P1.6.8 | Analytic Jacobians | ☑ | 2026-09-29 — leaf (SO(3) right Jacobian + RBF bend) and stem (linear basis), FD-verified; ~60 ms/frame at 10 evals with velocity warm start, ~4 mm tracking error at 5 mm/frame. 30 Hz still needs custom GN or GPU |
| P1.6.9 | Hardware run: RViz check of textured leaf + stem during a pull | ☐ | Needs arm + camera + F/T |
| P1.6.10 | Fused residual+Jacobian `evaluate` + own LM solver on normal equations | ☑ | 2026-09-29 — `solver.py`; ~28 ms/frame, 1.6 mm at 5 mm/frame; TRF/FD kept as references |

### P1 exit criteria

- Leaf-aware planning visibly avoids leaves where Octomap planning failed.
- nvblox update < 33 ms on Orin AGX.
- Workshop paper draft ready.

---

## Phase 2A — V-JEPA 2.1 dense temporal target state

**Target start:** 2026-09-15  ·  **Target finish:** 2026-11-15  ·  **Status:** ◐ in_progress — Stage A target memory shipped 2026-10-06 (P2A.1–P2A.4); E1 dataset and go/no-go benchmark (P2A.5–P2A.6) open

### P2A — Piper-JEPA Stage A: dense target memory (`scout_piper_jepa`)

| ID | Task | Status | Notes |
|---|---|---|---|
| P2A.1 | Encoder interface: V-JEPA (lazy torch.hub) + numpy reference encoder | ☑ | 2026-10-06 — hub entry point is config; not yet run on GPU |
| P2A.2 | Target memory: gated softmax, mean/cov, entropy, confidence, occlusion/lost, 3-D from aligned depth | ☑ | 2026-10-06 — gate + coasting needed: ungated memory jumps to an identical twin during occlusion |
| P2A.3 | E1 metrics + episode export/eval tools | ☑ | 2026-10-06 |
| P2A.4 | ROS 2 node with re-grounding via `/piper_jepa/init_mask` | ☑ | 2026-10-06 — untested on hardware |
| P2A.5 | Record E1 dataset and annotate target/distractor masks | ☐ | |
| P2A.6 | Run V-JEPA 2 vs 2.1 vs baselines (T0–T4) on E1 | ☐ | go/no-go gate |

---

## Phase 2B — Safety-bounded local MPPI visual servo

**Target start:** 2026-10-15  ·  **Target finish:** 2027-01-15  ·  **Status:** ☐ not started

### P2.1 — MPPI core
| ID | Task | Status | Notes |
|---|---|---|---|
| P2.1.1 | CUDA MPPI sampler (512 traj × 20 step) on Orin | ☐ | |
| P2.1.2 | Image-Jacobian linearization + fallback to online estimator | ☐ | |
| P2.1.3 | Cost terms: image error, visibility, manipulability, joint barrier, force | ☐ | |
| P2.1.4 | Soft cost from Phase 1 leaf SDF | ☐ | Couples to P1 |

### P2.2 — Integration with moveit_servo
| ID | Task | Status | Notes |
|---|---|---|---|
| P2.2.1 | Publish `JointJog` instead of TwistStamped | ☐ | More direct |
| P2.2.2 | Latency budget validation (< 10 ms per cycle) | ☐ | |

### P2.3 — Benchmark
| ID | Task | Status | Notes |
|---|---|---|---|
| P2.3.1 | 50 scripted approach trials, baseline vs MPPI-VS | ☐ | |
| P2.3.2 | ≥ 20 % success-rate gain, ≥ 30 % max-force reduction | ☐ | Exit gate |

---

## Phase 3A — Geometry-only whole-body GPU MPC (Piper + Scout)

**Target start:** 2026-12-15  ·  **Target finish:** 2027-04-15  ·  **Status:** ◐ in_progress — CPU geometry-only baseline shipped 2026-10-06, ahead of schedule (P3A.1–P3A.6, W0/W1 comparators); hardware runs, Orin timing and the cuRobo/GPU port (P3.1) open

### P3A — Geometry-only whole-body MPC (`scout_piper_whole_body_mpc`)

| ID | Task | Status | Notes |
|---|---|---|---|
| P3A.1 | Unicycle base model + skid-steer slip identification (`fit_slip`) | ☑ | 2026-10-06 — WE1 needs real floor data |
| P3A.2 | Piper FK/Jacobian from the URDF (sync test) | ☑ | 2026-10-06 — TCP offset to calibrate |
| P3A.3 | J_geo costs, MPPI with smooth noise, arm-only W0 mode | ☑ | 2026-10-06 — white noise froze or drifted the controller; W0 must not plan with the base |
| P3A.4 | Safety filter (limits, one-step clearance, watchdog) | ☑ | 2026-10-06 |
| P3A.5 | Semantic-scene adapter + dry-run ROS node | ☑ | 2026-10-06 — untested on hardware |
| P3A.6 | Escape obstacle local minima (gradient refinement, candidate B) | ☑ | 2026-10-07 — not a minimum: the MPPI average was worse than "stop" (elitism fixes), sampling missed the last cm (knot-space gradient refinement), planner and filter shared d_safe and deadlocked (planner margin +1 cm); O1 8/8 seeds ≤ 1 cm |
| P3A.7 | WE1 slip identification + hand-eye/TCP calibration on hardware | ☐ | |
| P3A.8 | Orin timing (WE7) | ☐ | ≈60 ms/step on x86 dev CPU (≈85 ms with an obstacle field) |
| P3A.9 | W1 sequential baseline (IK base pose → base phase → arm-only MPPI) | ☑ | 2026-10-07 — R3: W1 3/3 in 150 steps, 0.85 m base travel vs W3 3/3 in 59–92 steps, 0.65 m (offline, synthetic) |
| P3A.10 | W2 holistic / reactive QP baseline | ☐ | |
| P3A.11 | Execute mode end to end through servo, hardware-free | ☑ | 2026-10-07 — `fake_scout_base.py` + `fake_base:=true`, `execute:=` launch argument, `mpc_chain_check.py`: 10 runs on 4 goals reached in 3.6–14 s, 0.87–0.99 cm TCP error from TF, base stopped, no servo halt. Found and fixed: stopping the node mid-motion left the base driving (no Scout command timeout) — the node now sends zero commands on exit. 2026-10-08 — reach margin (`w_reach`, `reach_max_m` 0.36 m shoulder-to-wrist): far goals ended with the arm stretched (0.54 m on R3), so the stem servo halted at the singularity; now the base covers the rest (0.35 m on R3). Bringup helper nodes (relay, bridge, fake drivers) exit cleanly on Ctrl-C (one relay run died with an RCLError in rclpy's shutdown race) |

### P3.1 — cuRobo extension
| ID | Task | Status | Notes |
|---|---|---|---|
| P3.1.1 | Decide: planar virtual joint vs. forked CUDA kernel | ☐ | Pick higher-novelty path |
| P3.1.2 | Build 8-DoF kinematic chain in cuRobo | ☐ | |
| P3.1.3 | Non-holonomic constraint penalty for differential drive | ☐ | |

### P3.2 — Cost design
| ID | Task | Status | Notes |
|---|---|---|---|
| P3.2.1 | Reach + manipulability + base-laziness + visibility costs | ◐ | 2026-10-06 — goal, manipulability and base-motion costs in P3A.3; visibility open (Phase 3B J_visibility) |
| P3.2.2 | Phase 1 semantic SDFs integrated as cost potentials | ◐ | 2026-10-06 — CPU `SemanticDistanceQuery` via `scene_adapter.py` (P3A.5); nvblox-backed field open |
| P3.2.3 | Safety filter: clip whole-body command through Phase 2B MPPI | ☐ | standalone limits/clearance/watchdog filter done in P3A.4 |

### P3.3 — Benchmark
| ID | Task | Status | Notes |
|---|---|---|---|
| P3.3.1 | 30 cluttered-plant scenes; arm-only vs sequential vs holistic-QP vs ours | ☐ | |
| P3.3.2 | ≥ 50 % previously-unreachable targets reachable via base motion | ☐ | Exit gate |

---

## Phase 3B — Piper-JEPA predictive whole-body MPC

**Target start:** 2027-02-15  ·  **Target finish:** 2027-07-15  ·  **Status:** ◐ in_progress — Stage B predictor and the Stage C cost hook exist on a synthetic world (2026-10-07, ahead of schedule); no robot data, V-JEPA features or GPU timing yet

Plan: [`ROADMAP.md`](ROADMAP.md) §3 (Phase 3B) and [`research/piper_jepa/`](research/piper_jepa/) (Stages B–C).

| ID | Task | Status | Notes |
|---|---|---|---|
| P3B.1 | Synthetic dense-feature world (z-buffered eye-in-hand renderer, flower + twin + leaf) | ☑ | 2026-10-07 — test bed only, not evidence |
| P3B.2 | §12 read-outs (û, entropy, identity, visibility) from predicted features | ☑ | 2026-10-07 — spatial prior needed against the identical twin |
| P3B.3 | Γ from whole-body states (batched) for training data and MPC rollouts | ☑ | 2026-10-07 |
| P3B.4 | Learned predictor P0 / P2 / P3 + target-weighted training (§10–11) | ☑ | 2026-10-07 — CPU torch; P1 (V-JEPA 2-AC) needs the real encoder |
| P3B.5 | E3 metrics + synthetic E3 benchmark | ☑ | 2026-10-07 — synthetic, one seed: P3 E_target(4) 13.0 px vs 18.0 persistence / 21.9 P2; see `scout_piper_jepa/README.md` |
| P3B.6 | `JepaVisibilityCost` in `WholeBodyCost.extra`, geometry-anchored read-out | ☑ | 2026-10-07 — anchor fixes the identical-twin failure of the plain §12 read-out |
| P3B.7 | Synthetic closed loop C2 vs C3-oracle | ◐ | 2026-10-07 — inconclusive: C3 trades goal error for visibility without a reliable gain; needs a feasibility check, near-goal constraint, more samples |
| P3B.8 | Record and annotate E3 robot episodes (arm-only, base-only, combined, occlusion) | ☐ | bags → features + Γ + masks |
| P3B.9 | Train P2/P3 on V-JEPA 2.1 features (GPU) | ☐ | after P2A.6 go/no-go |
| P3B.10 | Predictor latency on the Orin (samples × horizon) | ☐ | CPU oracle render ≈ 2 s per control step at 64 samples |
| P3B.11 | C3 with the learned predictor in the loop; E5 visibility-sensitive scenes | ☐ | |

---

## Phase 4 — Uncertainty-driven active perception

**Target start:** 2027-05-15  ·  **Target finish:** 2027-08-15  ·  **Status:** ☐ not started

| ID | Task | Status | Notes |
|---|---|---|---|
| P4.1 | Deploy VGGT on Orin with sparse-view batching | ☐ | ~3 GB GPU |
| P4.2 | FisherRF-style information-gain on stem skeleton joint | ☐ | |
| P4.3 | Closed loop: VGGT → NBV → whole-body MPC → re-VGGT | ☐ | |
| P4.4 | Ablation: random / fixed-ring / ours | ☐ | |

---

## Phase 5 — Language/VLM + operator GUI for non-experts

**Target start:** 2027-07-15  ·  **Target finish:** 2027-11-15  ·  **Status:** ☐ not started

### P5.1 — VLA on Orin
| ID | Task | Status | Notes |
|---|---|---|---|
| P5.1.1 | Choose deployment model (π₀ default; Octo-Small fallback) | ☐ | |
| P5.1.2 | TRT-LLM / INT8 quantize, measure latency on Orin | ☐ | |
| P5.1.3 | Sub-goal generator service: image + instruction → SE(3) + class + rationale | ☐ | |
| P5.1.4 | PhysVLM-style reachability filter on every VLA-proposed target | ☐ | Safety gate |

### P5.2 — Open-vocab perception
| ID | Task | Status | Notes |
|---|---|---|---|
| P5.2.1 | Grounding-DINO + SAM-2 swap for fixed YOLO classes | ☐ | |
| P5.2.2 | Keep YOLO fallback for known classes | ☐ | |

### P5.3 — Operator GUI (laptop)
| ID | Task | Status | Notes |
|---|---|---|---|
| P5.3.1 | Tauri app scaffold, ROS 2 bridge via DDS | ☐ | |
| P5.3.2 | Tabs: Drive, Inspect, Command, Watch | ☐ | |
| P5.3.3 | Voice input (whisper-cpp) + voice output (Piper TTS) | ☐ | |
| P5.3.4 | Confirmation loop with 2-s countdown override | ☐ | |
| P5.3.5 | Plain-language target rejections ("Behind a leaf...") | ☐ | |
| P5.3.6 | Wire abort button to existing hot-key stop | ☐ | |

### P5.4 — User study
| ID | Task | Status | Notes |
|---|---|---|---|
| P5.4.1 | IRB / ethics approval | ☐ | |
| P5.4.2 | 10 non-expert participants, 3 task families | ☐ | |
| P5.4.3 | NASA-TLX + trust survey + task-success metrics | ☐ | |
| P5.4.4 | Paper draft (HRI / RO-MAN) | ☐ | |

---

## How to update this file

When you complete a task:
1. Flip ☐ → ☑ on the task row.
2. Add the date (YYYY-MM-DD) in the Notes column.
3. If the task uncovered a follow-up, add a new row below it.
4. If the task is blocked, change to ✗ and note the blocker.
5. Bump the **Last updated** date at the top.
6. Commit with a one-line message describing what shipped.
