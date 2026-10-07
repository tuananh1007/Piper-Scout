# Whole-Body MPC Research Plan

**Working title:** *Non-Holonomic Whole-Body MPC for Mobile Manipulation in Thin-Structure Plant Environments*  
**Track:** deterministic Scout + Piper control  
**Role:** strongest geometry-only controller and baseline for Piper-JEPA Stage C  
**Platform:** AgileX Scout 2.0 (4WD skid-steer) + AgileX Piper 6-DoF arm; shared facts in [`../README.md`](../README.md)

---

## 1. Research objective

This track asks:

> **Can a unified non-holonomic MPC coordinate the Scout base and Piper arm to improve reachability and manipulation efficiency in cluttered plant scenes relative to arm-only and sequential base-then-arm strategies, while maintaining explicit semantic clearance constraints?**

This controller is intentionally deterministic. It must be strong enough that Piper-JEPA later has a credible baseline.

---

## 2. Why this is a separate research problem

The mobile base and arm cannot be treated as independent modules when:
- the arm is near workspace limits;
- base motion changes arm manipulability;
- obstacles constrain both base and arm simultaneously;
- moving the base can shorten or lengthen the arm path;
- a sequential base pose may be locally poor.

A unified controller can trade base motion against arm motion continuously.

---

## 3. State and control

Scout planar pose:

```math
b=[x_b,y_b,\theta_b]^T.
```

Arm configuration:

```math
q\in\mathbb R^6.
```

Combined state:

```math
x=
[x_b,y_b,\theta_b,q^T]^T.
```

Control:

```math
u=
[v,\omega,\dot q^T]^T
\in\mathbb R^8.
```

Scout dynamics (unicycle model of the skid-steer base; slip and effective track width are identified in WE1):

```math
x_{b,t+1}
=
x_{b,t}
+
\Delta t\,v_t\cos\theta_t,
```

```math
y_{b,t+1}
=
y_{b,t}
+
\Delta t\,v_t\sin\theta_t,
```

```math
\theta_{t+1}
=
\theta_t
+
\Delta t\,\omega_t.
```

Arm:

```math
q_{t+1}
=
q_t+\Delta t\,\dot q_t.
```

The final controller must preserve this non-holonomic structure rather than treating base $x,y$ as independently actuated joints.

Arm kinematics include the fixed mount transform from the URDF (`base_link → piper_mount_link → piper_base_link`), so end-effector and camera poses are computed in the Scout frame and then in `odom`.

### Actuation path

| Control | Executed by | Note |
|---|---|---|
| $(v,\omega)$ | `scout_ros2`, `/cmd_vel` | skid-steer low-level controller tracks the unicycle command |
| $\dot q$ | `moveit_servo` joint jog: `control_msgs/JointJog` velocities on `/servo_node/delta_joint_cmds` (`/whole_body_mpc/preview/joint_jog` while `execute: false`) | the `stem_grasp` servo publishes Cartesian twists on `/servo_node/delta_twist_cmds`; servo must also accept joint commands |

The MPC rate, the servo rate and the Scout command rate are different loops; WE7 measures the end-to-end delay.

---

## 4. Hypotheses

### H1 — Reachability

Unified whole-body MPC converts more arm-unreachable targets into reachable targets than arm-only control.

### H2 — Coordination

Unified control produces lower task time/path cost than sequential base repositioning followed by arm planning.

### H3 — Manipulability-aware motion

A base-motion regularizer plus arm manipulability term produces interpretable behavior: use the arm when feasible; move the base when required or beneficial.

### H4 — Semantic geometry integration

Class-aware hard/soft semantic geometry improves feasibility relative to binary hard-obstacle treatment without increasing hard-structure collisions.

### H5 — Embedded feasibility

GPU-batched MPC/MPPI can sustain a useful receding-horizon rate on Jetson AGX Orin.

---

## 5. Baseline controller objective

Define

```math
J_{\mathrm{geo}}
=
w_gJ_{\mathrm{goal}}
+
w_cJ_{\mathrm{collision}}
+
w_lJ_{\mathrm{leaf}}
+
w_mJ_{\mathrm{manip}}
+
w_bJ_{\mathrm{base}}
+
w_sJ_{\mathrm{smooth}}.
```

Optional later term:

```math
+w_dJ_{\mathrm{deform}}
```

when valid `plant_twin` state exists. Because `plant_twin` fits only the current deformation (it has no forward model), a horizon-summed $J_{\mathrm{deform}}$ needs an explicit approximation, e.g. quasi-static re-fits with the predicted fingertip as the contact constraint at a few horizon knots.

No V-JEPA visibility/identity prediction is allowed in this track's baseline.

---

## 6. Goal cost

Let desired grasp pose be $`(p_g,R_g)`$.

```math
J_{\mathrm{goal}}
=
\sum_{k=1}^{H}
\|p^e_k-p_g\|_Q^2
+
\lambda_R
d^2_{SO(3)}(R^e_k,R_g).
```

If final orientation is underconstrained, use task-specific orientation weighting rather than forcing all axes equally.

---

## 7. Collision cost and constraints

Semantic Scene provides

```math
\phi_{\mathrm{hard}}(p).
```

For collision primitive $j$,

```math
d_{j,k}
=
\phi_{\mathrm{hard}}(p_j(x_k))-r_j.
```

Require or heavily penalize:

```math
d_{j,k}\ge d_{\mathrm{safe}}.
```

Leaf soft cost:

```math
J_{\mathrm{leaf}}
=
\sum_{k,j}
\psi(
\phi_{\mathrm{leaf}}(p_j(x_k))
).
```

Unknown/stale geometry should invoke conservative behavior rather than free-space assumption.

---

## 8. Manipulability

For arm Jacobian $J_a(q)$,

```math
m(q)
=
\sqrt{
\det(J_aJ_a^T)
}.
```

Use

```math
J_{\mathrm{manip}}
=
\sum_k
\frac{1}{m(q_k)+\epsilon}.
```

Alternative numerically stable manipulability/singularity measures may be used if the determinant becomes unstable.

---

## 9. Base-motion regularization

Use

```math
J_{\mathrm{base}}
=
\sum_k
(
v_k^2
+
\lambda_\omega\omega_k^2
).
```

This encodes a useful behavioral prior:

> prefer arm motion when the task is comfortably solvable; use the base when it materially improves reachability, clearance, or arm condition.

A separate base-displacement term may be added if needed.

---

## 10. Smoothness

```math
J_{\mathrm{smooth}}
=
\sum_k
\|u_k-u_{k-1}\|_R^2.
```

This is especially important for the base near plants.

---

## 11. Solver candidates

Do not decide the paper around a solver name before benchmarking.

Candidate A — batched MPPI / sampling-based MPC.  
Candidate B — gradient-based trajectory optimization.  
Candidate C — cuRobo/cuMotion-compatible formulation if non-holonomy can be represented cleanly.

Selection criteria:
- correct dynamics;
- semantic SDF integration;
- Orin latency;
- robustness;
- implementation complexity.

If cuRobo requires fake holonomic base joints, it should not be the final scientific formulation.

---

## 12. Safety projection

Nominal MPC action:

```math
u_{\mathrm{MPC}}.
```

Executed action:

```math
u_{\mathrm{safe}}
=
\arg\min_u
\|u-u_{\mathrm{MPC}}\|_2^2
```

subject to:
- hard semantic clearance;
- velocity limits;
- acceleration limits;
- joint limits;
- watchdog/freshness;
- force/contact where relevant (requires a force source: no wrist F/T sensor is on the platform yet; see [`../README.md`](../README.md)).

This safety layer is shared conceptually with Piper-JEPA Stage C.

---

## 13. Control hierarchy

### Far / navigation-scale motion

Nav2 moves Scout to the local manipulation region.

### Manipulation-scale whole-body motion

Whole-Body MPC coordinates Scout + Piper.

### Near-contact motion

Local servo / MPPI takes authority.

The whole-body controller should not attempt to replace the local contact controller.

---

## 14. Experimental scene classes

R1 — target comfortably reachable by arm.  
R2 — target near arm workspace boundary.  
R3 — target unreachable without base repositioning.  
R4 — several geometrically feasible base/arm paths in clutter that differ in target visibility.

R4 later becomes the bridge to Piper-JEPA visibility-sensitive experiments.

---

## 15. Baselines

W0 — arm-only planner/controller.  
W1 — sequential base reposition then arm planning.  
W2 — holistic/reactive QP baseline.  
W3 — unified whole-body MPC with binary geometry.  
W4 — unified whole-body MPC with semantic class-aware geometry.

Later Piper-JEPA becomes W5, but W5 does not belong to this deterministic research track.

---

## 16. Experiments

Use experiment IDs WE1–WE7 (methods keep W0–W5).

WE1 — dynamics/kinematics validation.  
WE2 — arm-only vs sequential vs unified reachability.  
WE3 — manipulability/base-motion behavior.  
WE4 — semantic geometry ablation.  
WE5 — clutter/reactive replanning.  
WE6 — near-contact handoff compatibility.  
WE7 — Orin timing and sustained operation.

Full protocol: `WHOLE_BODY_MPC_EXPERIMENTS.md`.

---

## 17. Primary metrics

- target reachability;
- grasp/approach success;
- time to target;
- Scout travel distance;
- EE path length;
- minimum hard clearance;
- leaf soft-contact metric;
- manipulability;
- number of replans;
- command smoothness;
- MPC solve latency;
- missed deadlines.

---

## 18. Key ablations

### A1 — non-holonomic model

Correct unicycle (skid-steer-identified) dynamics versus fake planar holonomic joints.

### A2 — base penalty

```math
w_b=0
```

versus tuned $`w_b>0`$.

### A3 — manipulability

With versus without $J_{\mathrm{manip}}$.

### A4 — semantic geometry

Binary-hard geometry versus class-aware hard/soft geometry.

### A5 — safety projection

Nominal MPC versus safety-projected execution in controlled validation.

---

## 19. Relationship to Semantic Scene

Whole-Body MPC can begin before Semantic Scene is complete by using:
- analytic obstacles;
- simple occupancy maps;
- synthetic SDFs.

However, the **final W4 deterministic controller depends on Semantic Scene** for the manipulation-specific geometry claim.

This allows parallel development.

---

## 20. Relationship to Piper-JEPA

Piper-JEPA Stage C should reuse the same:
- dynamics;
- solver;
- goal cost;
- collision cost;
- leaf cost;
- manipulability term;
- base regularization;
- safety projection.

Then add only:

```math
w_vJ_{\mathrm{visibility}}
+
w_iJ_{\mathrm{identity}}.
```

This creates a clean scientific comparison.

---

## 21. Relationship to plant_twin

Optional.

If useful:

```math
J_{\mathrm{geo}}
\rightarrow
J_{\mathrm{geo}}+w_dJ_{\mathrm{deform}}.
```

But the deterministic whole-body paper must stand without it.

---

## 22. Kill criteria

### K1

If unified MPC cannot outperform sequential control on R3/R4, do not claim coordination advantage.

### K2

If semantic geometry adds no benefit over binary geometry, retain simpler geometry in the controller paper.

### K3

If Orin timing is inadequate, reduce horizon/sample count or use asynchronous/split-rate architecture instead of overstating real-time performance.

---

## 23. Implementation mapping

Package layout (geometry-only baseline implemented 2026-10-06; `ROADMAP.md` §7 and the Piper-JEPA plan point here). Piper-JEPA adds its visibility/identity terms from `scout_piper_jepa/predictive_cost.py` through the `extra` hook in `costs/terms.py` without changing these modules.

```text
Codes/src/scout_piper_whole_body_mpc/
  scout_piper_whole_body_mpc/
    dynamics/
      scout.py          # unicycle + skid-steer k_v, k_ω; fit_slip (WE1)
      piper.py          # FK / Jacobian / manipulability from the URDF
      whole_body.py     # rollout, frames, collision spheres
    costs/
      terms.py          # goal, collision, unknown space, leaf, manipulability, joint margin, base, smoothness
    safety/
      projection.py     # limits, one-step clearance, watchdog
    solvers/
      mppi.py           # incl. arm_only mode (W0), elitism, knot-space gradient refinement (candidate B)
    baselines/
      sequential.py     # W1: IK base pose, base phase, arm-only phase
    scene_adapter.py    # Semantic Scene distance query → cost/safety
    sim.py
    controller_node.py
  config/
  launch/
  benchmarks/
    reachability.py
  test/
```

Candidate B is implemented as a gradient refinement of the MPPI result (P3A.6), not as a standalone solver; W2 (QP) is not implemented.

---

## 24. Immediate tasks

1. Implement Scout unicycle rollout and identify skid-steer slip / effective track width (WE1) (rollout + `fit_slip` done 2026-10-06; real-floor identification open, P3A.7).
2. Integrate Piper forward kinematics/Jacobian (done 2026-10-06, P3A.2).
3. Create synthetic SDF benchmark environment (done 2026-10-06: `sim.py`, analytic sphere fields, `benchmarks/reachability.py`).
4. Implement goal, smoothness, base, and manipulability costs (done 2026-10-06, P3A.3).
5. Compare MPPI versus one gradient/QP baseline offline (MPPI ± gradient refinement compared on O1, 2026-10-07; QP baseline W2 open).
6. Integrate Semantic Scene distance-query API when stable (adapter done 2026-10-06, `scene_adapter.py`; not yet run on real geometry).
7. Run R1-R3 before R4 (offline synthetic R1-R3 done 2026-10-06; hardware open).
8. Freeze W4 before integrating Piper-JEPA costs.

---

## 25. Publication positioning

Strong standalone framing:

> **A non-holonomic whole-body MPC that coordinates a skid-steer base and lightweight arm under class-aware semantic geometry for thin-structure plant manipulation.**

If novelty over existing whole-body MPC is insufficient, this track should become the deterministic method/baseline section of the Piper-JEPA paper rather than being forced into a separate publication.
