# Piper + Scout Project — Detailed Progress Tracker

> **Living document.** Update this file when any task is completed, blocked,
> or rescoped. Each phase has its own section with task-level granularity.
> The companion [`ROADMAP.md`](ROADMAP.md) is the high-level strategic plan;
> this file is the day-to-day execution log.

**Last updated:** 2026-05-16 (Phase 0 scaffolding shipped)

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
| P0.1.5 | Run `vcs import src < repos.yaml` and verify all four repos clone | ☐ | User action — needs network |
| P0.1.6 | `rosdep install --from-paths src --ignore-src -r -y` | ☐ | User action |
| P0.1.7 | First successful `colcon build --symlink-install` | ☐ | Confirms upstream + ours compile together |

### P0.2 — Unified URDF (scout_piper_description)

| ID | Task | Status | Notes |
|---|---|---|---|
| P0.2.1 | Package skeleton (package.xml, CMakeLists.txt) | ☑ | ament_cmake |
| P0.2.2 | `scout_piper.urdf.xacro` composing scout_description + piper_description + realsense | ☑ | Verify upstream paths after vcs import |
| P0.2.3 | Confirm no TF name collisions between Piper's `base_link` and Scout's `base_link` | ☐ | May require prefix arg on piper xacro |
| P0.2.4 | `view_robot.launch.py` + RViz config — visualize unified model | ☑ | Needs upstream meshes |
| P0.2.5 | Visual sanity: arm reaches expected workspace from Scout top plate | ☐ | After P0.2.3 |
| P0.2.6 | Add hand-eye TF (camera → link6) from existing calibration_samples.yaml | ☐ | Port [`../calibration_transform.py`](../calibration_transform.py) |
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
| P0.3.9 | MoveIt 2 demo plans a canned home→pose motion | ☐ | |
| P0.3.10 | All six subsystems running concurrently with stable TF tree | ☐ | Phase 0 exit gate |

### P0.4 — stem_grasp ROS 2 port

| ID | Task | Status | Notes |
|---|---|---|---|
| P0.4.1 | Package skeleton (ament_python, setup.py, entry_points) | ☑ | |
| P0.4.2 | `pipeline_node.py` scaffold: params, subs, pubs, timers, state machine | ☑ | Empty algorithm bodies |
| P0.4.3 | `segmentation_node.py` scaffold | ☑ | Empty stub |
| P0.4.4 | `pointcloud_node.py` scaffold | ☑ | Empty stub |
| P0.4.5 | `hotkey_stop_and_zero.py` ported (working) | ☑ | Zero-twist publish; service call still TODO |
| P0.4.6 | `core.py` (servo math) — port classes from ROS 1 verbatim | ☐ | Pure-math; no ROS deps |
| P0.4.7 | `moveit_planner.py` — port to `moveit_py` API | ☐ | ROS 1 used moveit_commander |
| P0.4.8 | `pipeline_node._on_stem_mask` — centroid extraction | ☐ | ROS 1 lines ~520–580 |
| P0.4.9 | `pipeline_node._on_target_point` — frame conversion + cache | ☐ | ROS 1 lines ~610–650 |
| P0.4.10 | `pipeline_node._outer_loop` — skeleton + candidate selection | ☐ | ROS 1 lines ~660–800 |
| P0.4.11 | `pipeline_node._outer_loop` — plan_and_execute + state transition | ☐ | ROS 1 lines ~830–850 |
| P0.4.12 | `pipeline_node._inner_loop` — visual servo step | ☐ | ROS 1 lines ~871–901 |
| P0.4.13 | Iterative approach state machine | ☐ | ROS 1 lines ~1070–1220 |
| P0.4.14 | Multi-view capture ring + ICP merge | ⊝ | Deferred to Phase 4 (replaced by NBV+VGGT) |
| P0.4.15 | `segmentation_node` YOLO seg path | ☐ | |
| P0.4.16 | `segmentation_node` Grounded-SAM + target_caption path | ☐ | |
| P0.4.17 | `pointcloud_node` mask-gated cloud filtering + `/static_cloud_out` | ☐ | |
| P0.4.18 | `pipeline_node` smoke test (no hardware) | ☑ | `test_pipeline_smoke.py` |
| P0.4.19 | End-to-end smoke: pipeline runs on bag file of ROS 1 data | ☐ | Use `rosbags` converter |

### P0.5 — moveit_servo wiring

The ROS 1 stack published twist commands to a topic with no subscriber.
ROS 2 has `moveit_servo` as a first-class node — we actually wire it up.

| ID | Task | Status | Notes |
|---|---|---|---|
| P0.5.1 | Add `moveit_servo` node to bringup launch | ☐ | After upstream piper_moveit is verified |
| P0.5.2 | Servo YAML config tuned for Piper (vel/accel limits, scaling) | ☐ | |
| P0.5.3 | Verify a manually-published TwistStamped actually moves the EE | ☐ | Phase 0 servo handoff validation |

### P0.6 — Nav2 stand-up (basic, defer fancy mapping to later)

| ID | Task | Status | Notes |
|---|---|---|---|
| P0.6.1 | Drop in `scout_nav2` launches via include | ☑ | Wired in `full_system.launch.py` (default off) |
| P0.6.2 | Verify Scout drives 2 m to a goal pose with arm folded | ☐ | Needs odom + 2D costmap |
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

## Phase 1 — Semantic 3D scene representation (nvblox)

**Target start:** 2026-07-15  ·  **Target finish:** 2026-09-15  ·  **Status:** ☐ not started

### P1.1 — nvblox integration

| ID | Task | Status | Notes |
|---|---|---|---|
| P1.1.1 | Install `isaac_ros_nvblox` on workstation | ☐ | Docker or native |
| P1.1.2 | Feed RealSense color+depth to nvblox; verify TSDF reconstruction | ☐ | |
| P1.1.3 | Benchmark nvblox update rate on Orin AGX target | ☐ | Goal: < 33 ms |
| P1.1.4 | Wire semantic mask channel from `segmentation_node` to nvblox | ☐ | May need fork for 4-class support |

### P1.2 — Per-class SDFs

| ID | Task | Status | Notes |
|---|---|---|---|
| P1.2.1 | Stand up 4 parallel TSDFs (stem / branch / leaf / target) | ☐ | |
| P1.2.2 | Per-class inflation / padding configurable via YAML | ☐ | |
| P1.2.3 | Visualize each SDF separately in RViz | ☐ | |

### P1.3 — MoveIt 2 collision plugin

| ID | Task | Status | Notes |
|---|---|---|---|
| P1.3.1 | Custom collision plugin reading nvblox SDFs | ☐ | C++ |
| P1.3.2 | Hard collision for stem/branch; soft cost for leaf | ☐ | |
| P1.3.3 | Target SDF exposed as attractor for goal generation | ☐ | |

### P1.4 — Benchmark

| ID | Task | Status | Notes |
|---|---|---|---|
| P1.4.1 | 20-scene cluttered-plant test set (recorded bags) | ☐ | |
| P1.4.2 | Compare planning success: Octomap vs ours | ☐ | Goal: ≥ 30 % failure reduction |

### P1 exit criteria

- Leaf-aware planning visibly avoids leaves where Octomap planning failed.
- nvblox update < 33 ms on Orin AGX.
- Workshop paper draft ready.

---

## Phase 2 — MPPI visual-predictive servo

**Target start:** 2026-09-15  ·  **Target finish:** 2026-12-15  ·  **Status:** ☐ not started

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

## Phase 3 — Whole-body GPU MPC (Piper + Scout)

**Target start:** 2026-12-15  ·  **Target finish:** 2027-04-15  ·  **Status:** ☐ not started

### P3.1 — cuRobo extension
| ID | Task | Status | Notes |
|---|---|---|---|
| P3.1.1 | Decide: planar virtual joint vs. forked CUDA kernel | ☐ | Pick higher-novelty path |
| P3.1.2 | Build 8-DoF kinematic chain in cuRobo | ☐ | |
| P3.1.3 | Non-holonomic constraint penalty for differential drive | ☐ | |

### P3.2 — Cost design
| ID | Task | Status | Notes |
|---|---|---|---|
| P3.2.1 | Reach + manipulability + base-laziness + visibility costs | ☐ | |
| P3.2.2 | Phase 1 semantic SDFs integrated as cost potentials | ☐ | |
| P3.2.3 | Safety filter: clip whole-body command through Phase 2 MPPI | ☐ | |

### P3.3 — Benchmark
| ID | Task | Status | Notes |
|---|---|---|---|
| P3.3.1 | 30 cluttered-plant scenes; arm-only vs sequential vs holistic-QP vs ours | ☐ | |
| P3.3.2 | ≥ 50 % previously-unreachable targets reachable via base motion | ☐ | Exit gate |

---

## Phase 4 — Active perception (VGGT + task-aware NBV)

**Target start:** 2027-04-15  ·  **Target finish:** 2027-06-15  ·  **Status:** ☐ not started

| ID | Task | Status | Notes |
|---|---|---|---|
| P4.1 | Deploy VGGT on Orin with sparse-view batching | ☐ | ~3 GB GPU |
| P4.2 | FisherRF-style information-gain on stem skeleton joint | ☐ | |
| P4.3 | Closed loop: VGGT → NBV → whole-body MPC → re-VGGT | ☐ | |
| P4.4 | Ablation: random / fixed-ring / ours | ☐ | |

---

## Phase 5 — VLA + operator GUI for non-expert use

**Target start:** 2027-06-15  ·  **Target finish:** 2027-11-15  ·  **Status:** ☐ not started

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
