# Whole-Body MPC Paper Outline

**Working title:** *Non-Holonomic Whole-Body MPC for Mobile Manipulation in Thin-Structure Plant Environments*

---

## 1. Thesis

A mobile manipulator should not choose a base pose first and then hope the arm can solve the task. A unified non-holonomic controller can coordinate Scout motion and Piper motion continuously while respecting semantic clearance and arm condition.

---

## 2. Abstract structure

1. Sequential base/arm planning is inefficient or infeasible near workspace/clutter boundaries.
2. Introduce unified non-holonomic Scout+Piper MPC.
3. Integrate semantic hard/soft SDF geometry and manipulability/base regularization.
4. Compare against arm-only, sequential, and reactive/QP baselines.
5. Report reachability, efficiency, clearance, and Orin timing.

---

## 3. Introduction

### P1 — mobile manipulation failure
Arm-only reach is limited; sequential base repositioning can choose poor local configurations.

### P2 — plant-specific challenge
Thin obstacles and leaves make base/arm coordination contact-sensitive.

### P3 — method
Unified differential-drive + arm MPC.

### P4 — contributions
- explicit non-holonomic whole-body formulation;
- manipulation-specific semantic geometry integration;
- arm/base coordination objective;
- embedded physical evaluation.

---

## 4. Related work

### 4.1 Mobile manipulation
### 4.2 Whole-body QP/MPC
### 4.3 GPU motion optimization / MPPI / cuRobo
### 4.4 Agricultural manipulation

Position against sequential planning and fake holonomic-base formulations.

---

## 5. Method

### 5.1 State/control

```math
x=[x_b,y_b,\theta_b,q]^T
```

```math
u=[v,\omega,\dot q]^T.
```

### 5.2 Differential-drive dynamics

Provide equations explicitly.

### 5.3 Whole-body objective

```math
J_{\mathrm{geo}}
=
w_gJ_g
+w_cJ_c
+w_lJ_l
+w_mJ_m
+w_bJ_b
+w_sJ_s.
```

### 5.4 Semantic collision integration

Use hard stem/branch and soft leaf fields.

### 5.5 Manipulability/base tradeoff

Explain why base penalty and manipulability jointly control behavior.

### 5.6 Solver

Describe actual solver selected after benchmark.

### 5.7 Safety projection and local handoff

Keep deterministic safety separate from optimization reward.

---

## 6. Experiments

W1 — model validation.  
W2 — reachability.  
W3 — coordination.  
W4 — semantic geometry.  
W5 — reactive replanning.  
W6 — servo handoff.  
W7 — Orin.

---

## 7. Results

### R1 — unified control expands reachability
Headline R3 result.

### R2 — arm/base coordination
Show base movement by reach class.

### R3 — semantic geometry improves feasible motion
W3 vs W4.

### R4 — reactive performance
Changing local scene.

### R5 — embedded timing
Horizon/sample trade-off.

---

## 8. Main figures

Figure 1 — Scout+Piper model and objective.  
Figure 2 — sequential vs unified trajectories.  
Figure 3 — R1/R2/R3 coordination behavior.  
Figure 4 — semantic hard/soft geometry trajectory example.  
Figure 5 — success/efficiency results.  
Figure 6 — Orin timing.

---

## 9. Main tables

Table 1 — method capabilities.  
Table 2 — success/time/path/clearance.  
Table 3 — ablations.  
Table 4 — runtime.

---

## 10. Required ablations

- non-holonomic vs fake holonomic base;
- with/without manipulability;
- with/without base penalty;
- binary vs semantic geometry;
- with/without safety projection.

---

## 11. Discussion

Explain:
- why unified coordination matters;
- why semantic geometry changes feasible motion;
- relationship to Piper-JEPA;
- limits of deterministic geometry for future target visibility.

That last point naturally motivates the Piper-JEPA paper:

> Whole-Body MPC solves geometric reachability, but not future visual target identity.

---

## 12. Limitations

- localization drift;
- geometry update latency;
- static/short-horizon plant assumptions;
- tuning sensitivity;
- solver compute;
- local-contact behavior delegated to another controller.

---

## 13. Publication decision

Submit standalone only if W2-W4 show clear gains over strong existing whole-body baselines.

Otherwise use this as the deterministic control backbone and primary baseline in Piper-JEPA.
