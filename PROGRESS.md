# Piper + Scout Project — Detailed Progress Tracker

> **Living document.** Update this file when any task is completed, blocked,
> or rescoped. Each phase has its own section with task-level granularity.
> The companion [`ROADMAP.md`](ROADMAP.md) is the high-level strategic plan;
> this file is the day-to-day execution log.

**Last updated:** 2026-10-10 (P2.1.5 contact force from the Piper's joint efforts: estimator node, calibration recorder, relay and fake-driver efforts, stem_grasp force topic and tare; Phase 2B MPPI visual servo; P3B.7 constrained visibility cost; 2026-10-09: MODULE_TASKS.md; semantic scene: label image, nvblox ESDF bridge, target goal, self-filter, RViz, semantic vs occupancy; whole-body MPC: torch / GPU backend, W2 QP, 30-scene benchmark, calibration tools; Piper-JEPA: E1 runner, export and annotation, training, latency, predictive MPC node; TEST_PROCEDURE.md and `hardware_free_checks.sh`; stem_grasp segmentation no longer pauses during the servo and approach; P0.1.13 Jetson AGX Orin deployment path; stem_grasp grasp point on the stem axis, release and retreat; P0.4.13 grasp: gripper closes on the stem, wrist room for the final approach; P0.1 memory-safe build and runtime for the 16 GB workstation; P0.4.13 stem_grasp stepwise final approach to AT_GRASP; P0.4.12 stem_grasp image-based servo toward the gripper axis, six servo bugs, camera placeholder orientation; whole-body MPC reach margin; P0.4.11 stem_grasp reach through the whole-body MPC, three stem_grasp porting bugs; P3A.11 whole-body MPC execute mode end to end on fake drivers, stop on exit; INSTALL.md TODOs resolved: D405 URDF, realsense-ros 4.58.4, CPU-only build closure, Nav2 blocker; P0.5 moveit_servo wired through the servo bridge; P0.3.13 arm command isolation; P3B.1–P3B.7 Piper-JEPA Stage B/C on a synthetic world; P3A.6, P3A.9 whole-body MPC; P1.7.7 MoveIt semantic collision plugin)

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
| P0.1.12 | Memory-safe build and runtime for the 16 GB lab workstation (Ryzen 7 3700X, RTX 3060 12 GB) | ☑ | 2026-10-08 — the PC froze on early steps; a plain `colcon build` on 16 threads starts far more compilers than 16 GB holds. Measured peak per compiler: rclcpp nodes 0.83 GB, the MoveIt plugin 0.92 GB, nvblox CUDA 0.98–1.24 GB (pinned `nvblox_core`, 37 CUDA files, 641 s on one core for sm_86; the five default Isaac architectures take 2.3–3.3× longer with the same memory). `scripts/colcon_build_safe.sh` (also used by `build_workspace.sh`) caps compilers at (free − 3 GB) / 1.5 GB, reads the container's cgroup limit, and builds CUDA only for the local GPU. The dev container is capped at 12 GB with no swap (`DEV_MEM_LIMIT`). INSTALL.md: earlyoom, OOM diagnosis, measured runtime (Phase 0 stack 1.0 GB, ~3 cores). `nvblox_semantic.launch.py classes:=` / `scene_classes:=` to run fewer nvblox processes; one BLAS thread for the MPC and pipeline. The build limits are not yet run on the workstation itself |
| P0.1.13 | Jetson AGX Orin 64 GB deployment path (INSTALL.md Path A) | ◐ | 2026-10-09 — written and checked off the robot, not run on the Orin. JetPack 6.1/6.2 (L4T r36.4, CUDA 12.6, VPI 3: what Isaac ROS release-3.2 is built for), native ROS 2 Humble arm64, NVIDIA's Jetson torch wheel. `patch_upstream.sh` now applies its VPI 4 ports only for VPI 4 (and reverts them for VPI 3), and checks the GXF prebuilts of the build architecture (`gxf_jetpack61` on aarch64); `colcon_build_safe.sh` compiles CUDA for 8.7 on a JetPack 6 Jetson; `scripts/check_jetson.sh` (read-only) reports L4T/JetPack, CUDA, VPI, power mode, memory, gs_usb/mttcan, CAN, RealSense, ROS and torch-with-CUDA; `full_system.launch.py` takes `piper_can_port` / `scout_can_port` (the Orin's onboard CAN may hold can0/can1); the MPC has `profile:=orin` (128 MPPI samples: 32 vs 62 ms per solve on x86) and warns when 5 of the last 20 solves overrun 90 % of the period. Workstation as operator station over the LAN (RViz, rosbags, training). `scripts/hardware_free_checks.sh` runs all seven hardware-free chain checks (servo, two MPC goals, four grasp runs) with a PASS/FAIL summary (`--profile orin`, `--quick`; all PASS here); `TEST_PROCEDURE.md` gives the test sequence for the workstation, the Orin and the robot with a results log; `print_tcp.py` prints the TCP and a ready MPC goal command for the robot tests. To do on the Orin: run check_jetson.sh, the build, the hardware-free chains, then the robot (TEST_PROCEDURE.md parts J and R) |

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
| P0.4.11 | `pipeline_node._outer_loop` — plan_and_execute + state transition | ☑ | 2026-10-08 — through the whole-body MPC instead of MoveIt (`reach_executor: whole_body_mpc`): SCANNING → REACHING (pre-grasp pose → `/whole_body_mpc/goal_pose`) → MPC "reached" for 1 s → cancel → SERVOING (`reach_handoff.py`). Hardware-free (`grasp_chain_check.py`): reach in 7–12 s, TCP within 0.2–1 cm of the pre-grasp, approach axis within 15°, MPC idle and silent after the handoff. The moveit_py path stays unwired; not run on hardware 2026-10-09 — handoff at the MPC's reported TCP error ≤ 2 cm and axis error ≤ 15° (`reach_handoff_tolerance_m`, `reach_handoff_angle_deg`) as well as its own 1 cm "reached": the MPC crept at 1.1–1.3 cm for seconds (reach up to 48 s); now 6–8 s at four stem positions |
| P0.4.12 | `pipeline_node._inner_loop` — visual servo step | ☑ | 2026-05-17 — drives `core.FullAdaptiveServoController` and publishes TwistStamped. 2026-10-08 — servoes the stem onto the gripper approach axis (`servo_geometry.py`): the desired pixel is the projection of the axis point at the target's depth (from TF; ROS 1 used the image centre), and the measured pixel is the stem column at the target's image row. Fixed: twists were sent with no frame although the velocity is in the camera optical frame; the depth was a constant; the observer started at pixel (0, 0); the image Jacobian used raw u instead of u − cx; the loop stepped at 100 Hz on 15 Hz masks with a fixed 10 ms observer step; sway was not measured against the camera's own motion (now from TF), which halved the gain every other step. The servo resets per phase and stops commanding on a mask older than 0.3 s. `servo_lambda_inf` 0.07 → 0.5 (moveit_servo realises ~55 % of a commanded velocity here). Hardware-free (`grasp_chain_check.py`, four stem positions): 33–65 px → 0.8–5 px in 15 s, gripper axis 0.04–0.25 cm from the grasp point, no servo halt. Not run on hardware; gains untuned. 2026-10-09 — found by review for the robot: the segmentation node's motion gate (`segment_only_when_stationary`, ROS 1) skips every frame while the joints move, so on the robot the servo would have stopped after its first step for lack of masks (the hardware-free checks feed masks directly). The gate is now off in the states of `motion_gate_off_states` (SERVOING, APPROACHING); unit test added |
| P0.4.13 | Iterative approach state machine | ☑ | 2026-10-08 — new design, since the ROS 1 source (lines ~1070–1220) is not in this repository; it keeps the ROS 1 `approach_*` names (`approach.py`). With `approach_enabled` the reach handoff goes to APPROACHING. The servo keeps the stem on the gripper axis, and once the error stays ≤ 8 px for `approach_settle_sec`, the gripper advances one `approach_step` at 2 cm/s, re-aligning between steps. It ends in AT_GRASP within `approach_distance_tolerance`, or on contact force, and in ABORTED on a start beyond `approach_target_distance`, `approach_max_steps`, a lost mask or a timeout. Hardware-free (`grasp_chain_check.py` with the approach on, seven runs at four stem positions): AT_GRASP after 3 steps in 17–24 s, TCP 0.85–0.97 cm short of the grasp point along the axis and 0.06–0.84 cm off it, the arm holding. Off by default: no force sensor on the Piper, and nothing closes the gripper yet. Not run on hardware 2026-10-09 — grasp: `grasp_close_gripper` opens the gripper before the approach and closes it at AT_GRASP through a new `piper_servo_bridge` input (`~/gripper_cmd`; enabled bridge only, kept through disabling, grip force 0.5), then GRASPED when the opening settles on the stem (`GripperCloseMonitor`), ABORTED if it closed on nothing. The fake driver's `object_width_m` stands in for the stem. Found and fixed: the approach drove the wrist toward its singularity (condition number 59, servo status 1), so the MPC now keeps the wrist room for the advance (`reach_max_advanced_m`); the pipeline publishes every state transition (AT_GRASP lasted under one 2 Hz state message). Hardware-free, grasp on, four stem positions: GRASPED, gripper axis 0.03–0.19 cm from the stem centreline, gripper settled at 8.0 mm, no servo slowdown. Known: the grasp point lies a few mm in front of the stem centreline (skeleton of the camera-facing half); nothing releases the stem yet 2026-10-09 (later) — grasp point centred on the stem axis (`core.stem_axis_point`: circle fit to the cross-section; the skeleton of a single-view cloud lies on the camera-facing surface; 7.7 mm fitted for the 8 mm stem); release: `~/release` opens the gripper and backs out 10 cm along the gripper axis to IDLE, `~/scan` scans again; checked hardware-free (`grasp_chain_check.py --release`) |
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
| P1.1.4 | Wire semantic mask channel from `segmentation_node` to nvblox | ◐ | v0 uses mask-gated depth (no fork) 2026-10-09 — segmentation_node publishes the merged label image `/stem_grasp/semantic_label` (stem 1, branch 2, leaf 3, target 4; YOLO class map `yolo_class_labels`), so `nvblox_semantic.launch.py` runs in its default merged mode; class_demux adds `other` (depth outside every mask). Unit-tested; to run: MODULE_TASKS.md A4 |

#### P1.2 — Per-class SDFs (v0)

| ID | Task | Status | Notes |
|---|---|---|---|
| P1.2.1 | Stand up 4 parallel TSDFs (stem / branch / leaf / target) | ◐ | 2026-10-07 — launch moved to `nvblox_ros` (smoke-test setup + per-class overrides); not yet run 2026-10-09 — mappers for stem, branch, leaf, target and `other` (pots, walls); `classes:=` subset; still not run (no GPU here): MODULE_TASKS.md A4 / B3 |
| P1.2.2 | Per-class inflation / padding configurable via YAML | ☑ | `semantic_classes.yaml` 2026-05-17 |
| P1.2.3 | Visualize each SDF separately in RViz | ◐ | 2026-10-09 — `rviz/semantic_scene.rviz` (per-class nvblox meshes, CPU voxels, target goal, masks), `nvblox_semantic.launch.py rviz:=true`; plugin class taken from nvblox_rviz_plugin 3.2; to check on the first GPU run (MODULE_TASKS.md A4) |

#### P1.3 — MoveIt 2 collision plugin

| ID | Task | Status | Notes |
|---|---|---|---|
| P1.3.1 | Custom collision plugin reading nvblox SDFs | ◐ | 2026-10-07 — plugin reads the CPU field (P1.7.7); nvblox per-class ESDFs still to feed it 2026-10-09 — `nvblox_field_bridge.py`: per-class ESDFs (`~/get_esdf_and_gradient`, esdf_mode 3d) resampled to one grid with the class policies and published as `/scene_repr/distance_field`, the message the plugin and the MPC already read (`nvblox_field.py`, unit-tested on synthetic ESDF responses). 2026-10-09 review — the node crashed at start (it assigned rclpy's read-only `Node.clients`): fixed; it now merges as soon as every answer is in, stamps the field with the request time and reports the answer sizes; against fake stem / target / other services (3 mm, 0.6 m box: 8.0 M voxels, 32 MB per class) request 0.1 s, merge 0.15 s, a field every 1 s, 0.3 s old on arrival, target goal published. Not run against nvblox yet (MODULE_TASKS.md A4) |
| P1.3.2 | Hard collision for stem/branch; soft cost for leaf | ◐ | 2026-10-07 — hard classes with padding + leaf penetration cap in the plugin; graded leaf cost only in the Python query / MPC (MoveIt collision is boolean) |
| P1.3.3 | Target SDF exposed as attractor for goal generation | ☑ | Policy schema in place. 2026-10-09 — `attractor.py`: largest target cluster → grasp point, horizontal approach from the base, pre-grasp 12 cm before it; published as `/scene_repr/target_goal` (PoseStamped, z = approach) by scene_query_node and the nvblox bridge; the MPC follows it with `goal_pose_topic:=/scene_repr/target_goal`. Unit-tested; robot run in MODULE_TASKS.md C4 |

#### P1.4 — Benchmark

| ID | Task | Status | Notes |
|---|---|---|---|
| P1.4.1 | 20-scene cluttered-plant test set (recorded bags) | ◐ | 2026-10-09 — `scripts/record_bag.sh scene` records the topics; bag → episode export with depth, poses and labels (scout_piper_jepa episode.py); offline map from an episode (`offline.py`). The 20 scenes are still to record (MODULE_TASKS.md C5) |
| P1.4.2 | Compare planning success: Octomap vs ours | ◐ | Goal: ≥ 30 % failure reduction. 2026-10-09 — `benchmarks/semantic_vs_occupancy.py` (scout_piper_whole_body_mpc): same MPC, leaves soft vs hard. Synthetic, 20 scenes: occupancy 17/20, semantic 18/20, i.e. 33 % fewer failures, but one scene of difference: weak evidence. `--episode` runs the same comparison on recorded scenes (MODULE_TASKS.md D1) |

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
| P1.7.8 | Robot self-filter for depth (fingers in view become `other` obstacles) | ◐ | until then exempt the finger links (ACM / `ignore_links`) 2026-10-09 — `self_filter.py`: boxes on piper_link7/8 projected into the depth image, pixels no farther than the box removed before integration, in scene_query_node (CPU map) and class_demux (nvblox); `self_filtered_px` in map_status. Box sizes approximate: tune on the robot (MODULE_TASKS.md C1) |
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
| P1.6.9 | Hardware run: RViz check of textured leaf + stem during a pull | ☐ | Needs arm + camera + a force (F/T sensor or the calibrated joint-effort estimate, P2.1.5) |
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
| P2A.4 | ROS 2 node with re-grounding via `/piper_jepa/init_mask` | ☑ | 2026-10-06 — untested on hardware 2026-10-10 review — the 3-D estimate needs the camera pose: without TF it returned a camera-frame point that `target_state_node` published as `odom` and the C3 node used as its world anchor; now it reports none. |
| P2A.5 | Record E1 dataset and annotate target/distractor masks | ◐ | 2026-10-09 — tools: `record_bag.sh e1`; `episode export` with depth, intrinsics, camera poses, states, framewise masks, labels; `annotate` (PNG import, keyframe polygons with interpolation, distractors). Recording and annotation pending (MODULE_TASKS.md A10 pilot, C6) 2026-10-09 review — the export now matches masks and labels recorded after their colour frame (segmentation latency; before, most were dropped), and saving polygons keeps the masks of frames outside the annotated keyframes. |
| P2A.6 | Run V-JEPA 2 vs 2.1 vs baselines (T0–T4) on E1 | ◐ | go/no-go gate. 2026-10-09 — `scout_piper_jepa.e1`: T0 framewise segmentation, T1 Lucas–Kanade, T2 DINOv2 + memory (new `DinoV2Encoder`), T3 / T4 V-JEPA 2 / 2.1 + memory (`config/e1_methods.yaml`), weighted E1 scores and the H1 rule (T4 vs best of T0–T2). Tested on synthetic episodes; V-JEPA / DINOv2 not run (no GPU here): MODULE_TASKS.md A6, D2 |

---

## Phase 2B — Safety-bounded local MPPI visual servo

**Target start:** 2026-10-15  ·  **Target finish:** 2027-01-15  ·  **Status:** ◐ in progress — MPPI visual servo hardware-free (2026-10-10); GPU timing and robot runs open

### P2.1 — MPPI core
| ID | Task | Status | Notes |
|---|---|---|---|
| P2.1.1 | CUDA MPPI sampler (512 traj × 20 step) on Orin | ◐ | 2026-10-10 — `scout_piper_whole_body_mpc/visual_servo.py`: arm-only MPPI (the whole-body MPPI, `arm_only`) with a torch twin of the cost, so `TorchMPPI` samples on CUDA; `benchmarks/visual_servo_timing.py`. CPU here: numpy ~50 ms at 256 samples, torch-CPU ~27 ms at 512 × 20; CUDA / Orin open (MODULE_TASKS.md A13, B7) |
| P2.1.2 | Image-Jacobian linearization + fallback to online estimator | ◐ | 2026-10-10 — projective prediction (FK + link6 → camera + pinhole, measured-pixel bias absorbs calibration error) or `mode: jacobian` (model linearised at q0); without a metric target the online Broyden estimate (`ImageJacobianEstimator`) takes over. Simulated eye-in-hand loop: centring 45 → < 3 px, 8 mm hand-eye error absorbed, continues after depth loss (tests) |
| P2.1.3 | Cost terms: image error, visibility, manipulability, joint barrier, force | ◐ | 2026-10-10 — image (pseudo-Huber toward the gripper-axis pixel), view barrier, joint barrier, manipulability, smoothness, contact force (above contact_threshold_n moving closer costs), approach distance; resolved-rate nominal seeds the sampling (without it the last centimetre of the approach stalled: 1 mm sideways ≈ 4 px near the stem); per-term costs in `/stem_grasp/servo_status`. Force: stem_grasp passes `contact_threshold_n` / `max_force_n` to the cost; the force comes from the joint-effort estimate (P2.1.5) |
| P2.1.5 | Contact force without an F/T sensor (joint-effort estimate) | ◐ | 2026-10-10 — `dynamics/effort.py`: τ = a(g(q) − JᵀF) + b + c·tanh(q̇/v_s) + d·q̇ per joint, gravity from the URDF inertials (gripper lumped on link6), fit on free motion (joints 1 and 6 carry no gravity load and take the median gain), weighted least-squares F, tare, per-axis noise. `effort_force_node` → `/ft_sensor/raw` + `~/status` (3-σ threshold) + `~/tare`; `calibrate_effort` (moves to a checked centre, 0.15 rad from the limits, 5 cm above the top plate; multi-sine; held-out residual); the relay passes driver efforts; the fake driver simulates efforts and a TCP force (`external_force_n`); stem_grasp `force_topic`, tare when a servo phase starts, stop on a stale force; bringup `bringup_force_estimate`. Hardware-free (`effort_chain`, `grasp_force_abort`): 0.04 N error on 3 N at rest, calibration gains 1.00 with 0.02 N·m held-out residual (3-σ 0.32 N), ≤ 0.08 N while servoing, 3 N aborts in 0.2 s. Open: real driver efforts and noise (MODULE_TASKS.md C2 step 4) |
| P2.1.4 | Soft cost from Phase 1 leaf SDF | ◐ | Couples to P1 2026-10-10 — `distance_fn` clearance term (semantic scene) in the cost and the safety filter; not wired in stem_grasp yet (needs the grasp-mode exclusion of the target stem) |

### P2.2 — Integration with moveit_servo
| ID | Task | Status | Notes |
|---|---|---|---|
| P2.2.1 | Publish `JointJog` instead of TwistStamped | ◐ | More direct 2026-10-10 — stem_grasp `servo_controller: mppi` publishes `JointJog` (frame_id `stem_grasp`) on `/servo_node/delta_joint_cmds`; hardware-free `--servo mppi`: full grasp passes (approach 12 s, axis 0.02 cm from the stem centreline, close, release, retreat). Robot open (MODULE_TASKS.md C9) |
| P2.2.2 | Latency budget validation (< 10 ms per cycle) | ◐ | 2026-10-10 — `visual_servo_timing.py` measures the cycle (cost, refinement, safety filter); 10 ms needs CUDA (A13, B7) |

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
| P3A.7 | WE1 slip identification + hand-eye/TCP calibration on hardware | ◐ | 2026-10-09 — `calibration.py` + ROS recorders: `calibrate_slip` (excitation plan, external pose topic, settling excluded; fake base with slip 0.9 / 0.75 → 0.879 / 0.741 with 1.5 s segments), `calibrate_tcp` (pivot), `calibrate_hand_eye` (Park–Martin, ArUco or a pose topic). Robot runs pending (MODULE_TASKS.md C2) |
| P3A.8 | Orin timing (WE7) | ◐ | ≈60 ms/step on x86 dev CPU (≈85 ms with an obstacle field) 2026-10-09 — `benchmarks/timing.py` (numpy / torch-cpu / torch-cuda, with and without the plant field). 4-core x86 sandbox, one thread: numpy 256 samples 93 ms (232 ms with the field), torch-cpu 27 ms (68 ms). PC GPU and Orin numbers pending (MODULE_TASKS.md A2, B2) |
| P3A.9 | W1 sequential baseline (IK base pose → base phase → arm-only MPPI) | ☑ | 2026-10-07 — R3: W1 3/3 in 150 steps, 0.85 m base travel vs W3 3/3 in 59–92 steps, 0.65 m (offline, synthetic) |
| P3A.10 | W2 holistic / reactive QP baseline | ☑ | 2026-10-09 — `baselines/reactive_qp.py`: one-step QP over (v, ω, q̇) with lazy base, joint limits, linearised clearance and optional reach margin (SLSQP, 2–4 ms per step). R1–R3 and O1 reached (0.9–1.0 cm); 30 plant scenes 28/30 (20/22 arm-unreachable), stalling at two obstacles where the MPC's horizon helps |
| P3A.11 | Execute mode end to end through servo, hardware-free | ☑ | 2026-10-07 — `fake_scout_base.py` + `fake_base:=true`, `execute:=` launch argument, `mpc_chain_check.py`: 10 runs on 4 goals reached in 3.6–14 s, 0.87–0.99 cm TCP error from TF, base stopped, no servo halt. Found and fixed: stopping the node mid-motion left the base driving (no Scout command timeout) — the node now sends zero commands on exit. 2026-10-08 — reach margin (`w_reach`, `reach_max_m` 0.36 m shoulder-to-wrist): far goals ended with the arm stretched (0.54 m on R3), so the stem servo halted at the singularity; now the base covers the rest (0.35 m on R3). Bringup helper nodes (relay, bridge, fake drivers) exit cleanly on Ctrl-C (one relay run died with an RCLError in rclpy's shutdown race) |

### P3.1 — cuRobo extension
| ID | Task | Status | Notes |
|---|---|---|---|
| P3.1.1 | Decide: planar virtual joint vs. forked CUDA kernel | ☑ | Pick higher-novelty path. 2026-10-09 — decision: own batched torch rollout with the base as (v, ω) inputs instead of planar virtual joints in a forked cuRobo kernel (exact non-holonomic model, one cost implementation, nothing to build for the Orin); collision through the semantic distance-field snapshot instead of cuRobo + nvblox (torch_backend.py docstring) |
| P3.1.2 | Build 8-DoF kinematic chain in cuRobo | ☑ | 2026-10-09 — replaced per P3.1.1: `torch_backend.py` (TorchModel FK / rollout / spheres / manipulability, TorchCost with every J_geo term, TorchMPPI with autograd refinement, TorchGridField on the device); term-by-term parity with numpy and closed-loop R1 / R3 / obstacle tests (`test_torch_backend.py`); node `backend: torch`, profiles `gpu` / `orin_gpu` |
| P3.1.3 | Non-holonomic constraint penalty for differential drive | ☑ | 2026-10-09 — not needed: the unicycle rollout enforces the non-holonomic constraint exactly (v along the heading only), in numpy and torch |

### P3.2 — Cost design
| ID | Task | Status | Notes |
|---|---|---|---|
| P3.2.1 | Reach + manipulability + base-laziness + visibility costs | ☑ | 2026-10-06 — goal, manipulability and base-motion costs in P3A.3; visibility open (Phase 3B J_visibility) 2026-10-09 — visibility cost: Piper-JEPA `JepaVisibilityCost` through `WholeBodyCost.extra` (P3B.6), in the node through `extra_terms` (predictive_mpc_node) |
| P3.2.2 | Phase 1 semantic SDFs integrated as cost potentials | ◐ | 2026-10-06 — CPU `SemanticDistanceQuery` via `scene_adapter.py` (P3A.5); nvblox-backed field open 2026-10-09 — the node plans on `/scene_repr/distance_field` (`field_topic`; CPU map or the nvblox bridge), copied to the GPU for the torch backend; points outside the plant grid count as free (they made the safety filter stop the base before); `unknown_policy: no_entry` lets spheres move inside never-observed space but not into it. Hardware-free: scene_query_node → field → MPC used it every cycle. 2026-10-09 review — field staleness limit `field_max_age_s` (2.5 s; fields at 1 Hz arrive up to 1.9 s old, over the 1 s `max_geometry_age_s` used before) and a stop until the first field arrives (it planned without geometry before). nvblox-backed run pending (MODULE_TASKS.md A5, C4) |
| P3.2.3 | Safety filter: clip whole-body command through Phase 2B MPPI | ☐ | standalone limits/clearance/watchdog filter done in P3A.4 |

### P3.3 — Benchmark
| ID | Task | Status | Notes |
|---|---|---|---|
| P3.3.1 | 30 cluttered-plant scenes; arm-only vs sequential vs holistic-QP vs ours | ☑ | 2026-10-09 — `benchmarks/plant_scenes.py` + `scenes.py`: 30 synthetic plant rows (capsule stems / branches, pots, disc leaves, targets at peduncles). W0 8/30, W1 30/30 (median 150 steps, 0.38 m base), W2 28/30 (42 steps, 0.24 m), W3 29/30 (60 steps, 0.21 m) |
| P3.3.2 | ≥ 50 % previously-unreachable targets reachable via base motion | ◐ | Exit gate. 2026-10-09 — synthetic: W3 reaches 21 of 22 arm-unreachable targets (95 %, gate ≥ 50 % passed); robot confirmation pending (MODULE_TASKS.md C3–C4) |

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
| P3B.7 | Synthetic closed loop C2 vs C3-oracle | ☑ | 2026-10-07 — inconclusive: C3 trades goal error for visibility without a reliable gain; needs a feasibility check, near-goal constraint, more samples. 2026-10-10 — all three done: 52 % of goal poses see the flower; visibility as `WholeBodyCost.secondary` (≤ 1 cm of horizon-mean goal error, ≤ 20 % of the remaining distance near the goal); 64 / 128 samples; `--hard` starts where C2 loses the flower. Result (oracle): goal error now as C2 (0.4–1.0 cm vs 2.6–15 cm additive); on hard starts the flower stays in view 55–79 % vs 17–31 % (C2), convergence 14–22 vs 45–120 steps; the end view and tracker identity are not rescued (flower hidden at the C2 end pose for these approach directions): the end pose must be chosen for the view (Codes/src/scout_piper_jepa/README.md) |
| P3B.8 | Record and annotate E3 robot episodes (arm-only, base-only, combined, occlusion) | ◐ | bags → features + Γ + masks. 2026-10-09 — tools ready (`record_bag.sh e3`, export with states and camera poses, annotation); end to end on the fake robot: 37 frames, states for 34, poses for 35, base 0.6 m. Recording pending (MODULE_TASKS.md C7) |
| P3B.9 | Train P2/P3 on V-JEPA 2.1 features (GPU) | ◐ | after P2A.6 go/no-go. 2026-10-09 — `scout_piper_jepa.train`: feature caching per episode, Γ from states, masks to the grid, episode split, P0 / P2 / P3 training on CUDA (`device`), E3 scores vs persistence, checkpoints. Synthetic smoke run on CPU only; GPU runs MODULE_TASKS.md A7, D3 2026-10-09 review — the checkpoint records the training frame interval (`step_s`); export E3 episodes with `--stride 6` (0.2 s at 30 fps). 2026-10-10 review — P2 / P3 take the joint angles as input as in §10 (the docstring said so, the network did not; `--no-state` ablates); `--proj-dim N` projects V-JEPA features onto their top N principal directions, stored in the checkpoint. Synthetic check, reduced (300 episodes, 600 steps, 1 seed, 73 test windows): P3 with / without the joint angles E_target@4 14.98 / 15.18 px (persistence 16.14), target_L1@4 2.78 / 2.85, id_acc@4 0.26 / 0.35, i.e. no clear difference at this size; D3 decides on robot data. |
| P3B.10 | Predictor latency on the Orin (samples × horizon) | ◐ | CPU oracle render ≈ 2 s per control step at 64 samples. 2026-10-09 — `scout_piper_jepa.latency`: rollout and predictive-cost time per sample count (fp16 option). CPU here: 64 samples × 4 steps ≈ 250 ms, too slow for 10 Hz on a CPU; GPU numbers MODULE_TASKS.md A8 / B4 2026-10-09 review — the benchmark rolls out over the 20-step MPC horizon at the model's predictor step (default 2 MPC steps: 10 predictor steps). 2026-10-10 review — feature size decides the cost: 256 samples × 10 steps × 16 × 16 × 1024 is 2.7 GB of predicted features per MPPI iteration; `latency.py --grid/--feat-dim/--input-dim` measures it before training (CPU, 32 samples: 1270 ms raw, 440 ms projected to 64); the shared context is copied to the device once; read-outs in float32. |
| P3B.11 | C3 with the learned predictor in the loop; E5 visibility-sensitive scenes | ◐ | 2026-10-09 — `predictive_mpc_node` (C3): the MPC node + Stage A memory + learned predictor + visibility cost in one process, camera pose from FK and the TF hand-eye, `jepa_ground` grounds the target. Hardware-free: cost active in 216/216 cycles once grounded. Learned model and robot runs pending (MODULE_TASKS.md B6, C8) 2026-10-09 review — `jepa_stride` 0 (default) is taken from the checkpoint's `step_s`, with a warning when no stride matches; `visibility_mpc.py --stride` likewise (it used 4 with models trained at 2). |

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
