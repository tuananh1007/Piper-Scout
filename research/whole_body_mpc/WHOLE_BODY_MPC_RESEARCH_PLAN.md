# Whole-Body MPC Research Plan

**Working title:** *Non-Holonomic Whole-Body MPC for Mobile Manipulation in Thin-Structure Plant Environments*  
**Track:** deterministic Scout + Piper control  
**Role:** strongest geometry-only controller and baseline for Piper-JEPA Stage C  
**Platform:** AgileX Scout 2.0 + AgileX Piper 6-DoF arm

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

Scout planar configuration:

\[
x_b=[x_b,y_b,\theta_b]^T.
\]

Arm configuration:

\[
q\in\mathbb R^6.
\]

Combined state:

\[
x=
[x_b,y_b,\theta_b,q^T]^T.
\]

Control:

\[
u=
[v,\omega,\dot q^T]^T
\in\mathbb R^8.
\]

Scout dynamics:

\[
x_{b,t+1}
=
x_{b,t}
+
\Delta t\,v_t\cos\theta_t,
\]

\[
y_{b,t+1}
=
y_{b,t}
+
\Delta t\,v_t\sin\theta_t,
\]

\[
\theta_{t+1}
=
\theta_t
+
\Delta t\,\omega_t.
\]

Arm:

\[
q_{t+1}
=
q_t+\Delta t\,\dot q_t.
\]

The final controller must preserve this non-holonomic structure rather than treating base \(x,y\) as independently actuated joints.

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

\[
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
\]

Optional later term:

\[
+w_dJ_{\mathrm{deform}}
\]

when valid \`plant_twin\` state exists.

No V-JEPA visibility/identity prediction is allowed in this track's baseline.

---

## 6. Goal cost

Let desired grasp pose be \((p_g,R_g)\).

\[
J_{\mathrm{goal}}
=
\sum_{k=1}^{H}
\|p^e_k-p_g\|_Q^2
+
\lambda_R
d^2_{SO(3)}(R^e_k,R_g).
\]

If final orientation is underconstrained, use task-specific orientation weighting rather than forcing all axes equally.

---

## 7. Collision cost and constraints

Semantic Scene provides

\[
\phi_{\mathrm{hard}}(p).
\]

For collision primitive \(j\),

\[
d_{j,k}
=
\phi_{\mathrm{hard}}(p_j(x_k))-r_j.
\]

Require or heavily penalize:

\[
d_{j,k}\ge d_{\mathrm{safe}}.
\]

Leaf soft cost:

\[
J_{\mathrm{leaf}}
=
\sum_{k,j}
\psi(
\phi_{\mathrm{leaf}}(p_j(x_k))
).
\]

Unknown/stale geometry should invoke conservative behavior rather than free-space assumption.

---

## 8. Manipulability

For arm Jacobian \(J_a(q)\),

\[
m(q)
=
\sqrt{
\det(J_aJ_a^T)
}.
\]

Use

\[
J_{\mathrm{manip}}
=
\sum_k
\frac{1}{m(q_k)+\epsilon}.
\]

Alternative numerically stable manipulability/singularity measures may be used if the determinant becomes unstable.

---

## 9. Base-motion regularization

Use

\[
J_{\mathrm{base}}
=
\sum_k
(
v_k^2
+
\lambda_\omega\omega_k^2
).
\]

This encodes a useful behavioral prior:

> prefer arm motion when the task is comfortably solvable; use the base when it materially improves reachability, clearance, or arm condition.

A separate base-displacement term may be added if needed.

---

## 10. Smoothness

\[
J_{\mathrm{smooth}}
=
\sum_k
\|u_k-u_{k-1}\|_R^2.
\]

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

\[
u_{\mathrm{MPC}}.
\]

Executed action:

\[
u_{\mathrm{safe}}
=
\arg\min_u
\|u-u_{\mathrm{MPC}}\|_2^2
\]

subject to:
- hard semantic clearance;
- velocity limits;
- acceleration limits;
- joint limits;
- watchdog/freshness;
- force/contact where relevant.

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
R4 — several feasible base/arm paths in clutter.

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

Use experiment IDs W1-W7.

W1 — dynamics/kinematics validation.  
W2 — arm-only vs sequential vs unified reachability.  
W3 — manipulability/base-motion behavior.  
W4 — semantic geometry ablation.  
W5 — clutter/reactive replanning.  
W6 — near-contact handoff compatibility.  
W7 — Orin timing and sustained operation.

Full protocol: \`WHOLE_BODY_MPC_EXPERIMENTS.md\`.

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

Correct differential-drive dynamics versus fake planar holonomic joints.

### A2 — base penalty

\[
w_b=0
\]

versus tuned \(w_b>0\).

### A3 — manipulability

With versus without \(J_{\mathrm{manip}}\).

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

\[
w_vJ_{\mathrm{visibility}}
+
w_iJ_{\mathrm{identity}}.
\]

This creates a clean scientific comparison.

---

## 21. Relationship to plant_twin

Optional.

If useful:

\[
J_{\mathrm{geo}}
\rightarrow
J_{\mathrm{geo}}+w_dJ_{\mathrm{deform}}.
\]

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

Proposed package:

\`\`\`text
Codes/src/scout_piper_whole_body_mpc/
  dynamics/
    scout_diff_drive.py
    piper_kinematics.py
    rollout.py
  costs/
    goal.py
    collision.py
    leaf.py
    manipulability.py
    base_motion.py
    smoothness.py
  safety/
    projection.py
    watchdog.py
  solvers/
    mppi.py
    gradient.py
  controller_node.py
  config/
  launch/
  benchmarks/
\`\`\`

---

## 24. Immediate tasks

1. Implement Scout differential-drive rollout.
2. Integrate Piper forward kinematics/Jacobian.
3. Create synthetic SDF benchmark environment.
4. Implement goal, smoothness, base, and manipulability costs.
5. Compare MPPI versus one gradient/QP baseline offline.
6. Integrate Semantic Scene distance-query API when stable.
7. Run R1-R3 before R4.
8. Freeze W4 before integrating Piper-JEPA costs.

---

## 25. Publication positioning

Strong standalone framing:

> **A non-holonomic whole-body MPC that coordinates a differential-drive base and lightweight arm under class-aware semantic geometry for thin-structure plant manipulation.**

If novelty over existing whole-body MPC is insufficient, this track should become the deterministic method/baseline section of the Piper-JEPA paper rather than being forced into a separate publication.
