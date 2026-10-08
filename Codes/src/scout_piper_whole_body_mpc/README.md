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
| `safety/projection.py` | box + rate limits, one-step joint limits, one-step hard clearance (scales the command toward zero), watchdog on stale state/geometry and unknown space at the robot |
| `scene_adapter.py` | `semantic_distance_fn` / `semantic_leaf_fn` over `scout_piper_scene_repr`'s query; analytic sphere fields for synthetic scenes |
| `sim.py`, `benchmarks/reachability.py` | closed-loop offline runs (WE2): W0, W1, W3 on R1–R3 and the obstacle scene O1 |
| `controller_node.py` | ROS 2 node; **dry run by default** |

## ROS interface

```text
in   /odom                          nav_msgs/Odometry
in   /joint_states                  sensor_msgs/JointState (joint_names param)
in   /whole_body_mpc/goal           geometry_msgs/PointStamped in world_frame
in   geometry                       in-process semantic map (use_semantic_scene: true)
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
- W2 (holistic / reactive QP) is not implemented.
- Not run on hardware; the tests cover the numpy core only (no ROS needed).

## Tests

```bash
cd Codes/src/scout_piper_whole_body_mpc && PYTHONPATH=../scout_piper_scene_repr/python python -m pytest test -q
# without the PYTHONPATH the semantic-scene adapter test is skipped
```
