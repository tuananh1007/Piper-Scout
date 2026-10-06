# scout_piper_whole_body_mpc — geometry-only whole-body MPC

Coordinates the Scout 2.0 base and the Piper arm toward a grasp point with
explicit non-holonomic dynamics, semantic clearance and a safety filter.
Research plan: [`research/whole_body_mpc/`](../../../research/whole_body_mpc/).
This is the deterministic baseline (W3/W4) that Piper-JEPA must beat.

## Pieces

| Module | Role |
|---|---|
| `dynamics/scout.py` | unicycle with skid-steer corrections `k_v`, `k_ω`; `fit_slip` identifies them from commanded (v, ω) and externally measured poses (WE1) |
| `dynamics/piper.py` | batched FK / geometric Jacobian / manipulability from the project URDF (table checked against `_piper_arm.xacro` by a test); TCP offset is a parameter to calibrate |
| `dynamics/whole_body.py` | x = [x_b, y_b, θ_b, q], u = [v, ω, q̇]; rollout, world frames, collision spheres (arm + base) |
| `costs/terms.py` | J_geo: goal (+ optional orientation / approach axis), smooth collision penalty, unknown-space penalty, leaf soft cost, manipulability, joint-limit margin, base motion, smoothness; `extra` hook for Piper-JEPA J_vis / J_id |
| `solvers/mppi.py` | MPPI with temporally smooth (knot-interpolated) noise, warm start, `arm_only` mode for the W0 baseline |
| `safety/projection.py` | box + rate limits, one-step joint limits, one-step hard clearance (scales the command toward zero), watchdog on stale state/geometry and unknown space at the robot |
| `scene_adapter.py` | `semantic_distance_fn` / `semantic_leaf_fn` over `scout_piper_scene_repr`'s query; analytic sphere fields for synthetic scenes |
| `sim.py`, `benchmarks/reachability.py` | closed-loop offline runs (WE2) |
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

Set `execute: true` only after the Phase 0 E-stop / stop-and-zero validation.
Arm velocities go to `moveit_servo` as `JointJog`, so servo must accept joint
commands. Check `joint_names` against what the Piper driver actually
publishes (the URDF prefixes `piper_`).

## Measured (offline, synthetic, 4-core x86 dev CPU)

`PYTHONPATH=. python3 benchmarks/reachability.py --seeds 2`:

| Scene | W0 arm-only | W3 unified | Unified base travel |
|---|---|---|---|
| R1 reachable | 2/2 (5–10 mm) | 2/2 (9–10 mm) | 8–11 cm |
| R2 workspace edge | 2/2 (15–18 mm, 150 steps) | 2/2 (9–10 mm, 42–53 steps) | ≈0.2 m |
| R3 needs base | 0/2 (≈0.62 m short) | 2/2 (6–7 mm) | ≈0.7 m |

≈60 ms per control step (256 samples × 20 steps × 2 iterations, plus the
safety filter) without semantic geometry — under the 100 ms budget of 10 Hz
here; the Orin CPU will be slower and semantic queries add cost.

## Known limitations

- **Local minima around obstacles.** With an obstacle on the straight path,
  MPPI detours safely but stalls 3–4 cm short (test documents this). A
  gradient refinement (solver candidate B) is the planned fix.
- R1 still drives the base 8–11 cm; `w_base` trades this against R3 speed.
- Kinematic model only; the skid-steer correction is two scalars until WE1
  data says otherwise.
- Not run on hardware; no ROS in the dev container.

## Tests

```bash
cd Codes && python -m pytest src/scout_piper_whole_body_mpc/test -q
```
