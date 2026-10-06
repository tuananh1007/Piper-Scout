# Piper-JEPA Paper Outline

**Working title:** *Piper-JEPA: Task-Conditioned Dense World Models for Safe Whole-Body Plant Manipulation*  
**Primary story:** predict whether the exact language-selected plant target will remain visible and identifiable after candidate mobile-manipulator motion, while explicit geometry and a separate safety controller govern execution  
**Companions:** PIPER_JEPA_RESEARCH_PLAN.md and PIPER_JEPA_EXPERIMENTS.md

---

## 1. Paper thesis

The paper should make one coherent argument:

> Geometry-aware whole-body planning can make a target reachable, but it does not guarantee that the exact language-selected target remains visually identifiable during the approach. Piper-JEPA adds a target-conditioned dense predictive video state to forecast future visibility and identity under candidate Scout + Piper actions, and uses that prediction as a soft objective within explicit semantic RGB-D whole-body MPC.

The paper should not read as a collection of independent modules.

---

## 2. Core novelty

The novelty is **not**:
- using V-JEPA 2.1;
- using V-JEPA for robot control;
- combining an action predictor with a video encoder;
- adding an LLM/VLM to a robot.

The novelty should be framed as:

1. persistent exact-instance target memory initialized from language grounding;
2. target-weighted dense action prediction for small deformable structures;
3. future visibility/identity costs inside non-holonomic whole-body MPC;
4. learned prediction separated from metric geometry and hard safety.

---

## 3. Abstract structure

Target 170-220 words.

### Sentence 1 — problem

Mobile manipulators can find collision-free motions to visually selected objects while still losing the exact target through self-occlusion, foliage occlusion, or instance ambiguity.

### Sentence 2 — gap

Existing geometry-based control models reachability and clearance, while latent world-model planning typically optimizes broader visual goal similarity rather than persistent identity of a language-selected small structure.

### Sentence 3 — method

Introduce Piper-JEPA: language-grounded target initialization + V-JEPA 2.1 dense target memory + target-weighted action-conditioned prediction + semantic RGB-D whole-body MPC.

### Sentence 4 — safety architecture

Learned prediction contributes visibility/identity costs; semantic SDF constraints, force limits, and the local servo remain authoritative.

### Sentences 5-6 — evidence

Report:
- target-ID persistence gain;
- future prediction gain;
- R4 visibility-sensitive control gain;
- full grasp gain;
- no safety degradation.

Do not write numerical claims until final experiments are frozen.

---

## 4. Introduction

### Paragraph 1 — application and failure mode

Open with a concrete command:

> “Grasp the third flower from the left.”

Explain why the difficult part is not only recognition or reachability. The robot must preserve the identity of “third flower” while its base and eye-in-hand camera move through foliage.

### Paragraph 2 — limitations of geometry-only mobile manipulation

Semantic SDF / MPC can model:
- clearance;
- reachability;
- non-holonomic base motion;
- manipulability.

But geometry does not predict visual identity loss.

### Paragraph 3 — limitations of framewise grounding

Framewise segmentation can:
- flicker;
- merge instances;
- switch targets;
- fail during occlusion.

A language-grounded target needs temporal persistence.

### Paragraph 4 — V-JEPA opportunity and novelty boundary

Explain:
- V-JEPA 2 established predictive representations and action-conditioned planning;
- V-JEPA 2.1 strengthens dense temporally consistent features;
- therefore “use V-JEPA for planning” is already insufficient as novelty.

State the new problem: exact target persistence under coordinated mobile manipulation.

### Paragraph 5 — Piper-JEPA

Describe the architecture in one paragraph.

### Paragraph 6 — contributions

Recommended contribution bullets:

1. **Target-conditioned dense memory.** A language-initialized V-JEPA 2.1 target state that maintains exact-instance identity and calibrated uncertainty through camera/base motion and temporary occlusion.
2. **Target-weighted predictive world model.** An action-conditioned dense predictor emphasizing the small target/plant regions and forecasting future target visibility and identity.
3. **Predictive non-holonomic whole-body control.** A Scout + Piper MPC that uses future visual-state costs while keeping semantic SDF collision constraints and safety projection explicit.
4. **Physical evaluation.** A benchmark and robot study covering similar targets, occlusion, sway, base motion, whole-body reachability, and grasp execution.

Only retain contribution 4 wording if the dataset/evaluation is released or sufficiently comprehensive.

---

## 5. Related work

Keep this section tightly connected to the gap.

### 5.1 Agricultural and deformable-object manipulation

Cover:
- harvesting/pollination;
- peduncle/stem perception;
- visual servoing in foliage;
- deformable plant interaction.

Conclude: these systems generally do not predict exact target identity under whole-body future actions.

### 5.2 Mobile manipulation and whole-body control

Cover:
- sequential base-arm planning;
- holistic QP;
- reactive whole-body control;
- GPU motion optimization / MPPI / cuRobo.

Conclude: strong geometry and dynamics, but no dense target-persistence world model.

### 5.3 Vision-language grounding for robotics

Cover:
- PaLM-E;
- RT-2;
- OK-Robot;
- MoMa-LLM;
- open-vocabulary grounding/segmentation.

Conclude: language determines what target is intended; it does not by itself maintain the same physical instance through motion.

### 5.4 Video world models and V-JEPA

Cover:
- JEPA family;
- V-JEPA 2;
- V-JEPA 2-AC;
- V-JEPA 2.1.

Be explicit that action-conditioned planning is prior work.

### 5.5 Positioning paragraph

End related work with a precise contrast:

> Piper-JEPA does not use a video world model as a replacement for geometry or control. It uses dense action-conditioned prediction to estimate whether the exact selected target remains visible and identifiable under candidate whole-body motions.

---

## 6. Method overview

This should correspond to Figure 1.

### 6.1 Inputs

- RGB-D stream;
- robot state;
- natural-language command.

### 6.2 Outputs

- persistent target state;
- future target-state predictions;
- safe base + arm command.

### 6.3 Responsibility separation

State explicitly:

```math
\boxed{
\text{language selects}
\rightarrow
\text{JEPA predicts}
\rightarrow
\text{geometry constrains}
\rightarrow
\text{MPC decides}
\rightarrow
\text{safety executes}
}
```

---

## 7. Method Section A — Language-grounded target initialization

Define:

```math
q_\ell=(c,k,r,\rho,a).
```

Explain candidate masks and ordinal/spatial selection.

Keep this concise; it is enabling machinery, not the main contribution.

### Required implementation detail

After initialization, the physical instance ID is frozen. Later framewise grounding cannot silently substitute another instance.

---

## 8. Method Section B — Dense V-JEPA 2.1 target memory

### 8.1 Dense encoder

```math
F_t=E_\theta(I_{t-L+1:t}).
```

### 8.2 Target descriptor

```math
r_t=
\frac{
\sum_p\widetilde M_t^*(p)F_t(p)
}{
\sum_p\widetilde M_t^*(p)+\epsilon
}.
```

### 8.3 Similarity distribution

```math
P_t(p)
=
\operatorname{softmax}
(
\cos(F_t(p),r_t)/\tau
).
```

### 8.4 State estimate

```math
\hat u_t=\sum_pP_t(p)p.
```

### 8.5 Uncertainty

```math
H_t=-\sum_pP_t(p)\log P_t(p).
```

Explain:
- why dense rather than pooled features;
- how target state survives segmentation dropout;
- how target loss/re-ground is triggered;
- how depth is fused only when valid.

### Evidence pointer

E1 + E2.

---

## 9. Method Section C — Action-conditioned target prediction

### 9.1 Robot action

```math
u_t=[v_b,\omega_b,\dot q_{1:6}].
```

### 9.2 Action embedding

```math
a_t=
[
\Delta s_b,
\Delta\theta_b,
\Delta p_e,
\Delta r_e,
\Delta g
].
```

Explain why this is preferable to raw motor commands.

### 9.3 Predictor

```math
\hat Z_{t+1:t+H}
=
P_\phi(
Z_t,
a_{t:t+H-1},
s_t
).
```

### 9.4 Target-weighted loss

```math
w_t(p)
=
1
+
\lambda_T M_t^{target}(p)
+
\lambda_P M_t^{plant}(p).
```

```math
\mathcal L_{AC}
=
\mathcal L_{TF}
+
\lambda_R\mathcal L_{roll}.
```

The text should emphasize that small target regions would otherwise contribute little to a global dense prediction loss.

### Evidence pointer

E3.

---

## 10. Method Section D — Predictive target visibility and identity

From predicted dense state:

```math
\hat P_{t+k}(p)
=
\operatorname{softmax}
(
\cos(\hat Z_{t+k}(p),r_t)/\tau
).
```

Then derive:
- predicted target position;
- entropy;
- identity consistency.

Define:

```math
J_{vis}
=
\sum_k
[
\|\hat u_{t+k}-u_{des}\|^2
+\lambda_H\hat H_{t+k}
+B_{FoV}
].
```

```math
J_{id}
=
\sum_k
[
1-\cos(\hat r_{t+k},r_t)
].
```

This section should contain the conceptual novelty figure: two geometrically feasible trajectories with different predicted visual outcomes.

---

## 11. Method Section E — Semantic geometry and plant deformation

### 11.1 Semantic SDF

Describe:
- non-target stem/branch hard obstacles;
- leaf soft cost;
- selected target special handling during gated grasp mode.

### 11.2 plant_twin

Describe only as much as needed:
- explicit deformable leaf/stem fit;
- optional deformation cost;
- confidence/freshness gating.

Do not let plant_twin dominate the main paper unless E8 is exceptionally strong.

---

## 12. Method Section F — Non-holonomic whole-body MPC

Scout state:

```math
[x_b,y_b,\theta_b].
```

Control:

```math
[v_b,\omega_b].
```

Combined control:

```math
u=
[v_b,\omega_b,\dot q_{1:6}]
\in\mathbb R^8.
```

Give the unicycle dynamics explicitly and report the identified skid-steer slip / effective track width.

Full objective:

```math
J
=
w_gJ_{goal}
+w_vJ_{vis}
+w_iJ_{id}
+w_cJ_{collision}
+w_lJ_{leaf}
+w_mJ_{manip}
+w_bJ_{base}
+w_sJ_{smooth}
+w_dJ_{deform}.
```

Explain:
- geometry-only baseline removes $J_{vis}$ and $J_{id}$;
- Piper-JEPA adds them;
- the comparison is therefore interpretable.

---

## 13. Method Section G — Safety and local handoff

### Safety projection

```math
u_{safe}
=
\arg\min_u
\|u-u_{MPC}\|_2^2
```

subject to:
- hard semantic clearance;
- force;
- joint/base velocity;
- acceleration;
- watchdog/freshness.

### Handoff

Switch to local servo when:
- distance below threshold;
- target entropy below threshold;
- confidence above threshold;
- semantic clearance valid.

State clearly:

> Learned prediction never owns final contact authority.

Evidence pointer: E7.

---

## 14. Experimental setup

### 14.1 Hardware

Report:
- Piper arm;
- Scout 2.0;
- RealSense eye-in-hand camera (state the deployed model: D435 or D405);
- Jetson AGX Orin 64 GB;
- F/T sensor if used.

### 14.2 Software

Report:
- ROS 2;
- V-JEPA checkpoint;
- nvblox;
- predictor;
- MPC implementation;
- software commit.

### 14.3 Plant scenes

Report:
- species;
- number of physical plants;
- number of scene configurations;
- target types;
- foliage conditions.

### 14.4 Evaluation factors

Use the E1-E9 definitions from PIPER_JEPA_EXPERIMENTS.md.

---

## 15. Results Section 1 — V-JEPA 2.1 improves exact target persistence

Evidence:
- E1;
- E2.

Main result:
- ID retention versus occlusion;
- recovery after temporary full occlusion;
- false target switching.

### Figure 2

Qualitative sequence.

### Table 1

Tracking/target-memory metrics.

### Required narrative

Do not merely say V-JEPA 2.1 has better dense features. Show why those features matter for the exact physical target.

---

## 16. Results Section 2 — Target-weighted action prediction improves future target estimates

Evidence:
- E3.

### Figure 3

Predicted target maps for alternative candidate actions.

### Figure 4

Prediction error versus:
- horizon;
- data budget.

### Table 2

P0-P3 metrics.

### Required narrative

Highlight whether target weighting improves target-specific prediction even if global latent prediction changes little.

---

## 17. Results Section 3 — Whole-body geometry baseline solves reachability

Evidence:
- E4.

Purpose:
- prove the system already has a strong non-JEPA controller;
- isolate the later JEPA benefit from simple mobility.

Show:
- R1-R4 success;
- Scout travel;
- time;
- clearance.

This section can be brief if space is limited.

---

## 18. Results Section 4 — Piper-JEPA preserves targets in visibility-sensitive motion

Evidence:
- E5.

This is the headline result.

### Figure 5

One R4 scene with:
- geometry-only chosen trajectory;
- Piper-JEPA trajectory;
- predicted future target heatmaps;
- actual visibility outcome.

### Figure 6

Target retained and first-attempt grasp success by method.

### Required comparison

Piper-JEPA versus geometry-only MPC on R4.

### Required safety check

Show no significant deterioration in:
- minimum clearance;
- max force.

---

## 19. Results Section 5 — Full language-to-grasp system

Evidence:
- E6.

Report:
- correct-instance grasp;
- first-attempt success;
- final success;
- target-loss rate;
- completion time.

Break down by:
- reachability class;
- occlusion;
- number of similar targets.

This demonstrates system relevance beyond the curated R4 experiment.

---

## 20. Results Section 6 — Safety, deformation, and embedded performance

This can be a compact main-paper section with details in supplement.

### Safety

E7:
- force;
- clearance;
- intervention.

### plant_twin

E8:
- only promote to main text if it materially improves control or interpretation.

### Orin

E9:
- encoder latency;
- predictor latency;
- MPC latency;
- end-to-end command latency;
- memory/power.

---

## 21. Discussion

### 21.1 What Piper-JEPA adds

It adds predicted visual target persistence to a geometry-based controller.

### 21.2 What it does not add

It does not replace:
- metric geometry;
- semantic collision checking;
- force limits;
- local servo;
- language grounding.

### 21.3 Why plant manipulation is a strong test

Plants combine:
- thin structures;
- deformability;
- self-occlusion;
- repeated similar instances;
- depth failure;
- moving eye-in-hand camera.

### 21.4 Generalization beyond agriculture

Potential domains:
- cable manipulation;
- surgical tool targeting;
- warehouse picking among similar objects;
- cluttered household manipulation.

Keep this speculative and clearly separate from measured evidence.

---

## 22. Limitations

Include explicitly:
- V-JEPA 2.1 was not designed specifically for plant instances;
- RealSense depth is weak on thin structures;
- action predictor requires embodiment-specific robot data;
- long-horizon rollouts may degrade;
- plant_twin is an approximate fitted model, not ground truth;
- system may be species/domain dependent;
- hard safety depends on quality/freshness of geometric state;
- embedded compute may limit predictor frequency.

A strong limitations section increases credibility.

---

## 23. Failure analysis

Use the experiment taxonomy:

F1 wrong initial grounding.  
F2 target switch.  
F3 lost target.  
F4 geometry/depth failure.  
F5 planner infeasible.  
F6 safety rejection.  
F7 servo failure.  
F8 grasp mechanics failure.  
F9 excessive deformation.  
F10 hardware/system fault.

Include a bar chart or table in supplementary material.

---

## 24. Conclusion

Three sentences are enough:

1. Piper-JEPA adds exact-target predictive visual state to whole-body mobile manipulation.
2. The learned model improves visibility-sensitive target retention while geometry and local control preserve safety.
3. The result supports a broader design principle: learned world models are most useful when they predict task-relevant uncertainty that explicit geometry does not represent.

---

## 25. Figure plan

### Figure 1 — Piper-JEPA architecture

Panels:
A. language command and selected target;
B. dense target memory;
C. candidate action-conditioned visual rollouts;
D. semantic whole-body MPC and safety handoff.

### Figure 2 — Target persistence

A. image sequence;
B. similarity maps;
C. temporary occlusion;
D. recovery;
E. ID retention plot.

### Figure 3 — Action prediction

A. candidate action sequences;
B. predicted future target distributions;
C. actual frames;
D. horizon-error plot.

### Figure 4 — Whole-body planning

A. R4 scene;
B. geometry-only trajectory;
C. Piper-JEPA trajectory;
D. target entropy over time.

### Figure 5 — Main quantitative robot result

A. target retained;
B. first-attempt grasp success;
C. target-loss count;
D. completion time.

### Figure 6 — Safety/deployment

A. minimum clearance;
B. max force;
C. Orin latency stack;
D. qualitative full-system grasp.

If the venue strongly limits figures, move Figure 6 to supplement.

---

## 26. Table plan

### Table 1 — Related method capabilities

| Method family | Language target | Temporal identity | Action prediction | Metric geometry | Whole-body | Hard safety |
|---|---:|---:|---:|---:|---:|---:|

### Table 2 — Representation/prediction ablation

| Method | ID retention | Recovery | Error @ horizon 4 | Error @ horizon 8 | Visibility F1 |
|---|---:|---:|---:|---:|---:|

### Table 3 — Robot results

| Method | Target retained | First grasp | Final success | Clearance | Force |
|---|---:|---:|---:|---:|---:|

### Table 4 — Embedded performance

| Model | Encoder | Predictor | MPC | End-to-end | Memory |
|---|---:|---:|---:|---:|---:|

Table 4 can move to supplement.

---

## 27. Claim-to-evidence matrix

| Manuscript claim | Evidence |
|---|---|
| exact-instance target memory is more robust | E1, E2 |
| action prediction forecasts target future | E3 |
| target weighting matters | E3 P3 vs P2 |
| whole-body mobility improves reachability | E4 |
| learned visibility prediction improves trajectory choice | E5 |
| full system improves correct-target grasping | E6 |
| safety not degraded | E7 + E5/E6 |
| plant_twin adds deformation value | E8 |
| system is deployable on Orin | E9 |

Before submission, remove any claim whose evidence cell is incomplete.

---

## 28. Main ablation set

The minimum reviewer-proof ablation should include:

A — current/legacy pipeline.  
B — geometry-only whole-body MPC.  
C — V-JEPA 2 action-conditioned.  
D — V-JEPA 2.1 dense + generic predictor.  
E — Piper-JEPA target-weighted predictor without local safety handoff.  
F — full Piper-JEPA.

Interpretation:

- C → D: V-JEPA 2.1 dense representation.
- D → E: target-conditioned/weighted predictive method.
- B → F: learned prediction beyond geometry.
- E → F: safety/local handoff.

plant_twin should be a separate ablation unless it becomes essential to the main result.

---

## 29. Supplementary material outline

### Supplement S1 — Hardware and calibration details
### Supplement S2 — Full network/predictor architecture
### Supplement S3 — ROS graph and timing
### Supplement S4 — Dataset annotation protocol
### Supplement S5 — Additional E1/E2 breakdowns
### Supplement S6 — Horizon/data-budget prediction results
### Supplement S7 — Whole-body controller parameters
### Supplement S8 — Safety filter details
### Supplement S9 — plant_twin model and ablation
### Supplement S10 — Orin profiling
### Supplement S11 — Full failure taxonomy
### Supplement S12 — Additional videos/qualitative cases

---

## 30. Video supplement plan

A robotics paper in this area should have a strong video.

Sequence:

1. Show command: “grasp the third flower from the left.”
2. Overlay grounded target ID.
3. Move arm/base while target is partly occluded.
4. Show target-memory heatmap persisting.
5. Show two candidate future rollouts.
6. Show geometry-only path losing target.
7. Show Piper-JEPA path preserving it.
8. Handoff to local servo.
9. Grasp.
10. Show failure examples, not only successes.

---

## 31. Reviewer objections to anticipate

### Objection 1

“V-JEPA 2-AC already plans robot actions.”

Response must come from E3/E5:
- exact small-instance target persistence;
- target-weighted dense prediction;
- non-holonomic base + arm;
- explicit semantic geometry;
- visibility-sensitive manipulation.

### Objection 2

“Why not use a tracker plus MPC?”

E1/E5 must compare against a strong tracker and geometry-only MPC.

### Objection 3

“Why is a learned model needed if RGB-D geometry exists?”

R4 scenes must show geometry-equivalent trajectories with different visual outcomes.

### Objection 4

“Is the gain just from moving the base?”

E4 isolates mobility. E5 compares against geometry-only whole-body MPC.

### Objection 5

“Does the learned model compromise safety?”

E7 and safety metrics in E5/E6 answer this.

### Objection 6

“Is plant_twin doing the work?”

Ablate it separately.

### Objection 7

“Can this run on the robot?”

E9.

---

## 32. Writing priorities

### Introduction

Be problem-first, not model-first.

Bad opening:
> V-JEPA 2.1 is a powerful foundation model...

Better:
> A mobile manipulator may reach the correct region of a plant while losing the exact flower requested by a user.

### Methods

Keep equations tightly tied to the target-persistence problem.

### Results

Lead with exact target identity and visibility-sensitive control, not generic benchmark scores.

### Discussion

Emphasize complementarity of learned prediction and explicit geometry.

---

## 33. Paper title alternatives

Primary:

**Piper-JEPA: Task-Conditioned Dense World Models for Safe Whole-Body Plant Manipulation**

Alternative 1:

**Predict What You Can Still Grasp: Dense Latent World Models for Visibility-Aware Mobile Manipulation**

Alternative 2:

**Preserving the Target: Predictive Dense World Models for Language-Grounded Whole-Body Manipulation**

Alternative 3:

**Language-Grounded Predictive Whole-Body Manipulation of Thin Plant Structures with V-JEPA 2.1**

Use the primary title unless the final paper broadens beyond plants.

---

## 34. One-line contribution statement

> Piper-JEPA predicts whether the exact object a human selected will remain identifiable after each candidate whole-body motion, and uses that prediction to complement—not replace—metric geometry and safety-bounded control.

---

## 35. Submission-readiness checklist

Before writing the final abstract:

- E1 target-persistence result complete.
- E3 prediction result complete.
- E4 strong geometry baseline complete.
- E5 R4 visibility-sensitive result complete.
- E6 full-system result complete.
- E7 safety comparison complete.
- E9 Orin timing complete.
- all main plots regenerated from frozen scripts;
- trial counts and exclusions finalized;
- statistical model finalized;
- no unsupported frequency or safety claim;
- no unverified “V-JEPA 2.1-AC” checkpoint claim;
- all contribution bullets mapped to evidence.

If E1, E3, or E5 fails, revise the paper thesis before submission rather than compensating with more peripheral experiments.
