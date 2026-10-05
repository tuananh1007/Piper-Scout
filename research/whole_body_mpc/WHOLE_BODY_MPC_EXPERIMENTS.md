# Whole-Body MPC Experiments

**Companion to:** WHOLE_BODY_MPC_RESEARCH_PLAN.md  
**Purpose:** deterministic whole-body-control evaluation protocol

---

## 1. Questions

W1. Is the dynamics/kinematics rollout correct?  
W2. Does unified control improve reachability over arm-only/sequential control?  
W3. Does manipulability/base regularization produce better coordination?  
W4. Does semantic class-aware geometry improve feasibility safely?  
W5. Can the controller react to changing local geometry?  
W6. Does handoff to local servo remain stable?  
W7. Can the stack run sustainably on Orin?

---

## 2. Method IDs

W0 — arm-only.  
W1 — sequential base then arm.  
W2 — holistic/reactive QP.  
W3 — unified MPC + binary geometry.  
W4 — unified MPC + semantic geometry.

Reserve W5 for Piper-JEPA later.

---

## 3. Scene classes

R1 — comfortably arm reachable.  
R2 — arm workspace boundary.  
R3 — base motion required.  
R4 — multiple feasible whole-body paths in clutter.

Use the same R labels as Piper-JEPA experiments.

---

## 4. W1 — dynamics/kinematics validation

### Base

Execute commanded \((v,\omega)\) sequences and compare predicted versus observed pose.

Translation error:

\[
E_p(k)
=
\|\hat p_k-p_k\|_2.
\]

Yaw error:

\[
E_\theta(k)
=
|\operatorname{wrap}(\hat\theta_k-\theta_k)|.
\]

### Arm

Validate predicted EE pose under joint-velocity rollout.

This experiment prevents controller results from being confounded by a wrong robot model.

---

## 5. W2 — reachability benchmark

### Methods

W0-W4.

### Recommended minimum

20 scenes per reachability class × 5 methods × 2 repeats.

Preferred physical campaign can be reduced after simulation/offline screening.

### Primary endpoint

Task/approach success.

### Key comparison

R3:

\[
W4 \text{ vs } W0/W1.
\]

### Secondary

- time;
- base distance;
- EE path;
- min clearance;
- replans.

---

## 6. W3 — coordination behavior

Construct R1/R2 scenes where base motion is possible but not necessary.

Test whether the base regularizer avoids gratuitous driving.

Metrics:
- base distance;
- task time;
- minimum manipulability;
- arm joint excursion.

Then construct R3 scenes where base motion is necessary.

Desired behavior:
- little/no base movement in R1;
- increasing base contribution in R2/R3.

---

## 7. W4 — semantic geometry ablation

Compare:
- W3: binary hard geometry;
- W4: semantic class-aware geometry.

Use scenes with leaves blocking an otherwise safe path while hard stems/branches remain nearby.

Metrics:
- success;
- no-plan rate;
- leaf-contact/soft-cost;
- hard-contact violations;
- path length;
- clearance.

Critical requirement: improved feasibility must not come from unsafe stem/branch contact.

---

## 8. W5 — reactive replanning

Introduce controlled changes:
- leaf moved into path;
- target pose update;
- small Scout localization correction;
- partial scene update.

Measure:
- replan latency;
- recovery success;
- command discontinuity;
- minimum clearance.

This is deterministic reactivity, not learned prediction.

---

## 9. W6 — local-servo handoff

Run approach until handoff threshold, then switch to local controller.

Measure:
- pose discontinuity;
- command discontinuity;
- handoff failure;
- final approach success;
- force.

A handoff is acceptable only if it does not introduce a transient toward obstacles.

---

## 10. W7 — Orin benchmark

Measure:
- rollout latency;
- cost evaluation latency;
- solve time median/p95/p99;
- missed deadlines;
- GPU memory;
- power;
- thermal behavior.

Sweep:
- horizon;
- trajectory/sample count;
- collision primitive count;
- SDF query batch size.

Do not preselect “100 Hz” or “200 Hz” as a claim.

---

## 11. Randomization

For physical comparisons:
- block by scene;
- randomize method order;
- reset base/arm start state;
- record plant changes;
- exclude only predeclared hardware/logging failures.

---

## 12. Statistics

Binary success:
- mixed-effects logistic regression with method + reachability class;
- random scene/plant effect.

Continuous:
- paired mixed model or paired bootstrap.

Main planned contrasts:
- W4 vs W0;
- W4 vs W1;
- W4 vs W3.

---

## 13. Pass criteria

### Reachability

W4 should clearly improve R3 success over arm-only.

### Coordination

W4 should not regress R1 behavior by driving unnecessarily.

### Semantic geometry

W4 should reduce false infeasibility relative to W3 without increasing hard collisions.

### Deployment

Timing must support a stable receding-horizon loop with documented latency.

---

## 14. Tables

### Table W-A

| Method | R1 success | R2 | R3 | R4 | Time ↓ |
|---|---:|---:|---:|---:|---:|

### Table W-B

| Method | Base dist ↓ | EE path ↓ | Min manip ↑ | Min clearance ↑ |
|---|---:|---:|---:|---:|

### Table W-C

| Horizon | Samples | Solve ms | p95 | Memory |
|---|---:|---:|---:|---:|

---

## 15. Immediate tasks

1. Build kinematic/dynamic validation scripts.
2. Create 10 synthetic R1-R3 scenes.
3. Implement W0/W1 first.
4. Implement W3 with synthetic SDF.
5. Connect Semantic Scene for W4.
6. Freeze W4 before Piper-JEPA integration.
