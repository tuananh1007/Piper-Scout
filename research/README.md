# Piper-Scout Research Program

This directory separates the major research threads within the broader Piper-Scout system. Each subdirectory is a research track with its own hypotheses, experiments, paper scope, and implementation links.

The system-level engineering sequence remains in [`../ROADMAP.md`](../ROADMAP.md); task status lives in [`../PROGRESS.md`](../PROGRESS.md).

## Structure

| Track | Scope | Roadmap phase | Implementation |
|---|---|---|---|
| [`semantic_scene/`](semantic_scene/) | semantic RGB-D geometry, nvblox TSDF/ESDF, class-aware collision/clearance | 1A | [`Codes/src/scout_piper_scene_repr/`](../Codes/src/scout_piper_scene_repr/) |
| [`plant_twin/`](plant_twin/) | explicit deformable leaf/stem twin, contact-conditioned deformation | 1B | [`Codes/src/plant_twin/`](../Codes/src/plant_twin/) |
| [`piper_jepa/`](piper_jepa/) | dense target memory, action-conditioned prediction, visibility/identity-aware whole-body control | 2A, 3B | [`Codes/src/scout_piper_jepa/`](../Codes/src/scout_piper_jepa/) (Stage A; Stage B/C on a synthetic world) |
| [`whole_body_mpc/`](whole_body_mpc/) | deterministic non-holonomic Scout + Piper whole-body MPC baselines | 3A | [`Codes/src/scout_piper_whole_body_mpc/`](../Codes/src/scout_piper_whole_body_mpc/) |
| [`active_perception/`](active_perception/) | uncertainty-driven next-best-view, task-aware multi-view perception | 4 | — |

These tracks are components of one system. They may support separate papers or be combined when the evidence is stronger as one integrated contribution.

---

## Shared platform facts

Every track states its assumptions against this list. When a fact changes, update it here and in `ROADMAP.md` first.

| Item | Current fact | Consequence for the research docs |
|---|---|---|
| Base | AgileX Scout 2.0, **4WD skid-steer**, commanded through `scout_ros2` with `/cmd_vel` = $(v,\omega)$ | Modelled as a unicycle / differential drive. Skid-steer adds lateral slip and an effective track width that must be identified on the real floor (MPC experiment WE1). |
| Arm | AgileX Piper, 6 revolute DoF + parallel gripper (`piper_joint7`/`joint8`), mounted on the Scout top plate (`piper_mount_link`) | FK starts from the URDF mount transform. Arm motion is commanded through `moveit_servo`: Cartesian twists on `/servo_node/delta_twist_cmds` (`stem_grasp`) and joint velocities as `JointJog` on `/servo_node/delta_joint_cmds` (`scout_piper_whole_body_mpc` with `execute: true`; servo must accept joint commands). |
| Camera | Intel RealSense, **eye-in-hand** on `piper_link6` | **Model to be confirmed.** The URDF and earlier docs say D435; the Phase 1 nvblox smoke test was run with a D405. The minimum usable depth differs (≈7 cm for D405 vs a few tens of cm for D435), which decides whether depth is usable in the final grasp approach. |
| Depth alignment | Bringup enables `align_depth`; every depth consumer (`stem_grasp`, `plant_twin`, `scout_piper_scene_repr` incl. the nvblox launches, `scout_piper_jepa`, bringup `system.yaml`) defaults to `/camera/aligned_depth_to_color/image_raw` | Masks and depth share the colour frame (P1.7.4, 2026-10-06). Any new depth consumer must use the aligned topic. |
| Force sensing | Code subscribes to `/ft_sensor/raw` (`WrenchStamped`) | **No wrist F/T sensor is in the hardware list.** Force gates and contact detection need either an added wrist F/T sensor or a calibrated estimate from Piper joint efforts. Until then, force-based claims are not testable. |
| Compute | Jetson AGX Orin 64 GB on-robot; operator laptop for GUI / optional VLM | All timing claims must be measured on the Orin; desktop or dev-container numbers are indicative only. |
| Semantic collision plugin | `scout_piper_scene_repr` MoveIt plugin `Semantic` = FCL + the CPU semantic distance field that `scene_query_node` publishes (P1.7.7); the CPU per-class query (`SemanticVoxelMap` + `SemanticDistanceQuery`, P1.7.1–P1.7.3) is tested on synthetic scenes; `scout_piper_whole_body_mpc` can embed it in-process | Tested only in software (real MoveIt model, synthetic field). Not a safety layer until validated on recorded scenes and the robot (P1.7.6); depth on the robot itself is not filtered yet (P1.7.8). |
| `plant_twin` | One leaf + its stem per instance; Gauss-Newton/LM fit ≈28 ms/frame on a 4-core x86 dev CPU; RViz markers + petiole point; untested on hardware | It is a **current-state fitter**, not a forward simulator. Future-horizon deformation costs need an explicit approximation (see each track's `J_deform` note). Fit confidence and timestamp are not yet published. |

---

## ID registry

IDs are local to a track. When citing an ID from another track, prefix it with the track tag, for example `SS:G3`, `WB:W4`, `PJ:C2`.

| Tag | Track | Methods | Experiments | Other IDs |
|---|---|---|---|---|
| `SS` | semantic_scene | G0–G4 | S1–S6 | H1–H4 hypotheses, K1–K3 kill criteria |
| `WB` | whole_body_mpc | W0–W5 | WE1–WE7 | A1–A5 ablations, H1–H5, K1–K3 |
| `PJ` | piper_jepa | T0–T4 tracking, P0–P3 prediction, C0–C4 control | E1–E9 | M0–M5 motion conditions, SF0–SF3 safety stack, D0–D2 `plant_twin` ablation, GF1–GF5 grounding failures, H1–H5 |
| `RM` | ROADMAP §8 | A–F full-system ablation | — | — |

### Shared scene classes (all tracks)

| Class | Definition |
|---|---|
| R1 | target comfortably reachable by the arm alone |
| R2 | target near the arm's workspace boundary |
| R3 | target unreachable without Scout repositioning |
| R4 | several geometrically feasible whole-body paths in clutter that differ substantially in target visibility |

R4 is where geometry-only control and Piper-JEPA should differ; no deterministic `W*` controller sees visibility.

### Method correspondences

| Behaviour | Roadmap §8 | Whole-body MPC | Piper-JEPA |
|---|---|---|---|
| legacy / current arm-only pipeline | A | W0 | C0 |
| sequential Scout then arm | — | W1 | C1 |
| holistic / reactive QP | — | W2 | — |
| unified MPC, binary geometry | — | W3 | — |
| geometry-only whole-body MPC (semantic) | B | W4 | C2 |
| JEPA-aware whole-body MPC, no local handoff | C (predictor P1), D (P2), E (P3) | W5 | C3 |
| Piper-JEPA + safety projection + local servo | F | W5 + handoff | C4 |
