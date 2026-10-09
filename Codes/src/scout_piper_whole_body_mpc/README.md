# scout_piper_whole_body_mpc — geometry-only whole-body MPC

Coordinates the Scout 2.0 base and the Piper arm toward a grasp point with
explicit non-holonomic dynamics, semantic clearance and a safety filter.
Research plan: [`research/whole_body_mpc/`](../../../research/whole_body_mpc/).
This is the deterministic baseline (W3/W4) that Piper-JEPA must beat; the
arm-only (W0) and sequential base-then-arm (W1) comparators live alongside it.

## Pieces

| Module | Role |
|---|---|
| `dynamics/scout.py` | unicycle with skid-steer corrections `k_v`, `k_ω`; `fit_slip` identifies them from commanded (v, ω) and externally measured poses (WE1) |
| `dynamics/piper.py` | batched FK / geometric Jacobian / manipulability from the project URDF (table checked against `_piper_arm.xacro` by a test); TCP offset is a parameter to calibrate |
| `dynamics/whole_body.py` | x = [x_b, y_b, θ_b, q], u = [v, ω, q̇]; rollout, world frames, collision spheres (arm + base) |
| `costs/terms.py` | J_geo: goal (+ optional orientation / approach axis), smooth collision penalty, unknown-space penalty, leaf soft cost, manipulability, joint-limit margin, base motion, smoothness; `extra` hook for Piper-JEPA J_vis / J_id |
| `solvers/mppi.py` | MPPI with temporally smooth (knot-interpolated) noise, warm start, `arm_only` mode for the W0 baseline; elitism (`keep_best`) and gradient refinement in knot space (`refine_iters`, candidate B) |
| `baselines/sequential.py` | W1: base pose from batched DLS IK (current pose first, then rings facing the goal; clearance-checked), turn/drive/turn base phase with the arm frozen, then arm-only MPPI |
| `baselines/reactive_qp.py` | W2: one-step holistic QP over (v, ω, q̇) — lazy base, joint limits, linearised sphere clearance, optional reach margin (SciPy SLSQP, 2–4 ms per step) |
| `safety/projection.py` | box + rate limits, one-step joint limits, one-step hard clearance (scales the command toward zero), watchdog on stale state/geometry; unknown space: `unknown_policy` "stop" (any sphere in unknown space) or "no_entry" (spheres may move inside never-observed space, not into it) |
| `scene_adapter.py` | `semantic_distance_fn` / `semantic_leaf_fn` over `scout_piper_scene_repr`'s query (outside the plant grid = free); analytic spheres, capsules (stems), discs (leaves); `gridded` voxel fields for fast synthetic queries |
| `scenes.py` | synthetic cluttered-plant rows for P3.3 (capsule stems / branches, pots, disc leaves, targets at peduncles, pre-grasp goals) |
| `torch_backend.py` | the same model, cost and MPPI in torch (CUDA on the RTX 3060 / Orin, or CPU); `TorchGridField` puts the semantic field on the device; autograd refinement |
| `calibration.py`, `calibration_nodes.py` | slip identification from an excitation plan and an external pose reference, TCP pivot calibration, eye-in-hand calibration (Park–Martin); ROS recorders `calibrate_slip`, `calibrate_tcp`, `calibrate_hand_eye` |
| `sim.py`, `benchmarks/` | closed-loop offline runs: `reachability.py` (R1–R3, O1), `plant_scenes.py` (W0–W3 on 30 plant scenes), `semantic_vs_occupancy.py` (P1.4.2, synthetic or recorded scenes), `timing.py` (numpy / torch / CUDA solve time) |
| `controller_node.py` | ROS 2 node; **dry run by default**; `backend: torch` for the GPU; `extra_terms` hook (Piper-JEPA's predictive node) |

## ROS interface

```text
in   /odom                          nav_msgs/Odometry
in   /joint_states                  sensor_msgs/JointState (joint_names param)
in   /whole_body_mpc/goal           geometry_msgs/PointStamped in world_frame
in   /whole_body_mpc/goal_pose      geometry_msgs/PoseStamped in world_frame; z axis = approach direction
in   /whole_body_mpc/cancel         std_msgs/Empty: drop the goal, stop, mode "idle"
in   geometry                       field_topic (SemanticDistanceField from scene_query_node or the
                                   nvblox bridge) or the in-process semantic map (use_semantic_scene)
out  /whole_body_mpc/preview/cmd_vel, /whole_body_mpc/preview/joint_jog   (execute: false)
out  /cmd_vel, /servo_node/delta_joint_cmds (control_msgs/JointJog)     (execute: true)
out  /whole_body_mpc/status         std_msgs/String JSON (mode, error, safety, timing, ages)
out  /whole_body_mpc/plan           nav_msgs/Path of the predicted TCP
```

Set `execute: true` (`ros2 launch scout_piper_whole_body_mpc whole_body_mpc.launch.py
execute:=true`) only after the Phase 0 E-stop / stop-and-zero validation. Arm
velocities go to `moveit_servo` as `JointJog` and reach the Piper through
`piper_servo_bridge` (scout_piper_bringup, `bringup_servo:=true`); the joint
states come from the bringup's relay as `piper_joint1..6`.

**Pose goals and handoff.** A `goal_pose` adds the approach axis as a soft
cost (`w_orient`) and to the `reached` test (`approach_tolerance_deg`, 15°);
re-sending the same goal keeps the solver's warm start. `stem_grasp` uses this
to reach its pre-grasp pose and then cancels the MPC so its image-based servo
has `moveit_servo` to itself (scout_piper_bringup `grasp_chain_check.py`,
INSTALL.md 10.9).

**Reach margin (2026-10-08).** The node adds `w_reach` (1e4) times the squared
excess of the shoulder-to-wrist distance over `reach_max_m` (0.36 m; the arm's
maximum is 0.537 m). The distance depends only on the elbow angle, so the term
costs about 1.5 ms per solve. Without it a far goal ended with the arm
stretched (0.54 m on R3), which leaves the stem servo and the final approach no
room to advance the gripper, and moveit_servo then halted at the singularity.
With it the base covers the rest: 0.35 m on R3, with the base travelling 0.81 m
instead of 0.65 m. `CostWeights.reach` defaults to 0, so the offline benchmark
below is unchanged; `test_reach_margin_moves_the_base_instead_of_stretching_the_arm`
covers the term.

**Room for the final approach (2026-10-09).** After the reach, stem_grasp
advances the gripper `pose_goal_advance_m` (0.12 m) along its axis with the
orientation fixed. That straightens the Piper's wrist (q5 → 0 is its
singularity): one hardware-free run ended at a Jacobian condition number of 59,
and moveit_servo slowed (status 1). For pose goals the cost adds
`w_reach_advanced` (1e3) times the squared excess of the shoulder-to-wrist
distance *after* that advance over `reach_max_advanced_m` (0.40 m). It is
computed at the end of the horizon without inverse kinematics: the shoulder
does not move, and the wrist centre translates with the TCP. Offline, the
distance after the advance went from 0.45–0.48 m to 0.35–0.40 m, with reach in
52–69 steps instead of 41–49. At the full `w_reach` weight (1e4) the term kept
one goal from converging within 250 steps. Hardware-free, the stem grasp at
four positions then ran without a servo slowdown.

**Jetson AGX Orin (`profile:=orin`, 2026-10-09).** The MPPI solve is numpy on
the CPU. With the reach terms, 256 samples took 62 ms per 100 ms step on one
2.1 GHz x86 core; 192 took 41 ms, 128 took 32 ms, and 128 with one refinement
step 27 ms. `whole_body_mpc.launch.py profile:=orin` loads
`config/whole_body_mpc_orin.yaml` (128 samples) after the main config. The
hardware-free MPC chain reached its goal in 6.8 s with it (median solve 44 ms
in the full ROS stack), and the stem grasp passed at two positions. The node
warns when 5 of the last 20 solves take over 90 % of the period. Orin timings are still to be
measured (INSTALL.md Path A).

**Stopping.** When the node exits (Ctrl-C, SIGTERM from `ros2 launch`, an
exception) it publishes zero base and joint velocities before shutting down.
Neither `scout_ros2` nor `ugv_sdk` stops the Scout when `/cmd_vel` stops
arriving: without this, stopping the node mid-motion left the (fake) base
driving at 0.2 m/s. A stale input (`max_state_age_s`) makes the safety filter
send zeros while the node runs; a crash that kills the process outright still
cannot stop the base.

**Hardware-free execute mode (2026-10-07).** With `fake_arm:=true
fake_base:=true bringup_servo:=true` in `full_system.launch.py`,
`scout_piper_bringup`'s `mpc_chain_check.py` runs the whole chain (MPC →
`/cmd_vel` + servo → bridge → fake arm/base → relay/odom → MPC). On a 4-core x86
CPU, ten runs on four goals 0.55–1.0 m out (base motion up to 0.33 m) all
reached the MPC tolerance in 3.6–14 s, with 0.87–0.99 cm TCP error measured
independently from TF, the base at rest and no servo halt; MPPI solve 49–64 ms
median, ≤ 185 ms worst case. The fake drivers are kinematic stand-ins (no slip,
no arm dynamics), so this checks plumbing, frames, timing and stopping, not
tracking on the robot.

**Profiles.** `whole_body_mpc.launch.py profile:=default | orin | gpu | orin_gpu`
loads `config/whole_body_mpc_<profile>.yaml` after the main config: `orin`
128 numpy samples; `gpu` the torch backend on CUDA with 1024 samples;
`orin_gpu` the torch backend on the Orin's GPU with 512. Set the sample
counts from `benchmarks/timing.py` on each machine (MODULE_TASKS.md A2, B2).
`field_topic:=` and `goal_pose_topic:=` override those parameters from the
command line (e.g. `goal_pose_topic:=/scene_repr/target_goal`).

## Torch / GPU backend (P3.1, 2026-10-09)

`torch_backend.py` is a line-by-line port of the model, every cost term and
the MPPI solver (elitism, warm start; the gradient refinement uses autograd
instead of finite differences). The base stays two non-holonomic inputs
(v, ω) of an exact unicycle rollout, so no cuRobo fork, virtual planar
joints or non-holonomic penalty are needed (P3.1.1–P3.1.3). Geometry: the
semantic field snapshot on the device (`TorchGridField`, trilinear, unknown
voxels invalid), or the numpy distance function evaluated on the CPU.
`test_torch_backend.py` checks rollout, FK, spheres, manipulability and every
cost term against numpy (float64, 1e-6) and closed-loop R1, R3 and an
obstacle on the device field.

`benchmarks/timing.py`, 4-core x86 sandbox without a GPU, one thread, median
ms per solve (2 iterations, 2 refinement steps):

| Samples | numpy | numpy, plant field | torch CPU | torch CPU, plant field |
|---|---|---|---|---|
| 128 | 59 | 144 | 22 | 46 |
| 256 | 93 | 232 | 27 | 68 |
| 512 | 164 | 409 | 39 | 115 |
| 1024 | 327 | 796 | 69 | 202 |

CUDA numbers come from the workstation and the Orin (MODULE_TASKS.md A2, B2).

## Cluttered plant scenes, W0–W3 (P3.3, 2026-10-09)

`benchmarks/plant_scenes.py --scenes 30`: synthetic rows of 1–3 potted
plants (capsule stems, stakes and branches, disc leaves) 0.75–1.7 m ahead,
the goal 12 cm before a peduncle 0.25–0.65 m high; the planners and the
safety filter use the scene on a 1 cm voxel grid, the reported clearances
the exact geometry. 22 of the 30 targets are out of the arm's reach from the
start (no collision-free IK within 5 mm from 16 seeds).

| Method | All | Arm-reachable | Arm-unreachable | Median steps | Median base travel |
|---|---|---|---|---|---|
| W0 arm-only | 8/30 | 8/8 | 0/22 | 91 | 0 |
| W1 sequential | 30/30 | 8/8 | 22/22 | 150 | 0.38 m |
| W2 reactive QP | 28/30 | 8/8 | 20/22 | 42 | 0.24 m |
| W3 unified MPPI | 29/30 | 8/8 | 21/22 | 60 | 0.21 m |

Exit gate P3.3.2 (≥ 50 % of the arm-unreachable targets reached): W3 95 %,
passed on synthetic scenes. W2 stalls at the clearance boundary in two scenes
(the one-step QP has no horizon); W3 ends 4.6 cm short in one (P09) at the
planning margin; W1 reaches all but needs the full 150 steps. Synthetic
geometry, kinematic model: robot runs decide.

`benchmarks/semantic_vs_occupancy.py` (P1.4.2) runs W3 with leaves soft
(semantic policy) or hard (occupancy only) on the same scenes: 18/20 vs 17/20,
one scene of difference. With `--episode` it builds the semantic map from
recorded scenes instead (scout_piper_scene_repr `offline.py`).

## Calibration (P3A.7, 2026-10-09)

- `ros2 run scout_piper_whole_body_mpc calibrate_slip [--execute] --pose-topic <external pose>`:
  drives `calibration.excitation_plan` (straight, arcs, turns on the spot,
  ~50 s, returns near the start), records commands and poses, fits `k_v`,
  `k_omega` without the settling after each command change. On the fake base
  with slip 0.9 / 0.75 it found 0.879 / 0.741 with 1.5 s segments.
- `calibrate_tcp`: Enter captures the flange pose while the gripper tip
  touches one fixed point; least-squares pivot fit → `tcp_offset_m`; refuses
  orientation spreads below 20°.
- `calibrate_hand_eye --aruco --marker-length <m>` (or `--board-topic`):
  flange and board poses → link6 → camera transform as a URDF `<origin>`.

## Measured (offline, synthetic, 4-core x86 dev CPU)

`PYTHONPATH=. python3 benchmarks/reachability.py --seeds 3` (2026-10-07):

| Scene | W0 arm-only | W1 sequential | W3 unified |
|---|---|---|---|
| R1 reachable | 3/3, 5–7 mm, 136–150 steps | = W0 (base not needed) | 3/3, 2–10 mm, 29–99 steps, base 2–4 cm |
| R2 workspace edge | 3/3, 3–8 mm, 84–141 steps | = W0 (base not needed) | 3/3, 1–10 mm, 39–150 steps, base 13–18 cm |
| R3 needs base | 0/3 (≈0.63 m short) | 3/3, 3–8 mm, 150 steps, base 0.85 m | 3/3, 10 mm, 59–92 steps, base 0.64–0.66 m |
| O1 obstacle on the path | 3/3, 8–10 mm, min clearance ≥ 3.3 cm | = W0 (base not needed) | 3/3, 1–9 mm, min clearance ≥ 2.9 cm |

Steps are 0.1 s control cycles until the TCP is within 1 cm and the command
has settled (150 = the cap). Success is < 2 cm. On R3 the unified controller
reaches the goal in about half the time with 0.2 m less base travel than the
sequential baseline: the first (synthetic, offline) WE2 evidence for
coordination.

≈60 ms per control step without geometry, ≈85 ms with the obstacle field
(256 samples × 20 steps × 2 iterations, elitism, 2 refinement steps, plus the
safety filter) — under the 100 ms budget of 10 Hz here; the Orin CPU will be
slower and semantic queries add cost. `refine_iters: 0` saves ≈15 ms.

## Converging near obstacles (P3A.6)

Plain MPPI used to end 3–5 cm short of the O1 goal with zero collision cost
and every command passed by the safety filter. Two causes, three fixes:

- **The weighted average was worse than its own samples.** Near the goal
  the averaged sequence often cost more than holding still, so each executed
  command carried sampling noise and the arm wandered around the goal.
  *Elitism:* the averaged sequence is scored and replaced by the best sample
  when that is cheaper, so a solve never returns worse than "stop".
- **Sampling cannot resolve the last centimetre.** *Gradient refinement:*
  `refine_iters` steps of gradient descent on the same cost in the 4-knot
  noise space (central differences, one batch of 64 rollouts, then a batched
  line search); a step is kept only if it lowers the cost.
- **Planner and filter used the same margin.** Plans grazed the filter's
  2 cm boundary and the filter refused every move, freezing the arm for
  ~120 steps on one seed. The planner now penalises clearance below
  `d_safe + plan_margin_m` (3 cm), above the filter's `d_safe` (2 cm).

Measured on O1 (final TCP error after ≤ 150 steps): plain MPPI 2.8–5.0 cm
on 4/4 seeds; elitism + refinement with equal margins 6/8 seeds ≤ 1 cm and
two deadlocked against the filter (4.8 cm, 26 cm); all three fixes 8/8 seeds
≤ 1 cm with no filter interventions and minimum clearance ≥ 2.9 cm.

## Known limitations

- The obstacle cases are synthetic spheres; real thin stems and leaves come
  through the semantic scene adapter and have not been run yet.
- Kinematic model only; the skid-steer correction is two scalars until WE1
  data says otherwise.
- Not run on hardware; the tests cover the numpy core and the torch backend (no ROS needed).

## Tests

```bash
cd Codes/src/scout_piper_whole_body_mpc && PYTHONPATH=../scout_piper_scene_repr/python python -m pytest test -q
# without the PYTHONPATH the semantic-scene adapter test is skipped
```
