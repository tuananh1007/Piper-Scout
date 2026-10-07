# Piper-JEPA Research Plan

**Working title:** *Piper-JEPA: Task-Conditioned Dense World Models for Safe Whole-Body Plant Manipulation*  
**Platform:** AgileX Piper 6-DoF arm + AgileX Scout 2.0 skid-steer base + eye-in-hand Intel RealSense (D435 or D405, to be confirmed) + NVIDIA Jetson AGX Orin 64 GB; shared facts in [`../README.md`](../README.md)  
**Application:** language-grounded grasping of thin, deformable plant structures (flowers, peduncles, branches)  
**Status:** scientific plan aligned with `ROADMAP.md` as of 2026-10-05  
**Role of this document:** authoritative research/paper plan. `ROADMAP.md` remains the engineering sequence.

---

## 1. Executive summary

Piper-JEPA studies a specific failure mode in mobile manipulation that is poorly addressed by geometry-only planning: the robot may have several collision-free trajectories to a language-selected target, but some motions make the requested flower or peduncle disappear behind foliage, induce target ambiguity, or cause the system to switch to a visually similar neighboring instance.

The central idea is to combine:

1. **language/VLM grounding** to select the intended physical instance;
2. **V-JEPA 2.1 dense temporal features** to maintain target identity through motion and temporary occlusion;
3. **an action-conditioned predictor** to forecast the visual consequence of candidate Scout + Piper motions;
4. **explicit RGB-D semantic geometry** for metric collision and clearance constraints;
5. **optional explicit plant deformation state from `plant_twin`**;
6. **non-holonomic whole-body MPC** for coordinated base + arm motion;
7. **a separate safety-bounded local visual servo** for the final centimeters.

The core design principle is:

> **JEPA predicts; geometry constrains; MPC decides; the safety layer executes.**

The research objective is not to replace robotics with a learned world model. The objective is to determine whether a dense predictive video representation can add useful information that explicit geometry does not contain: **will the exact target requested by the human remain identifiable and visible after this candidate robot motion?**

---

## 2. Research gap

### 2.1 Geometry alone is insufficient

A metric map can answer:
- Is a trajectory collision-free?
- How far is the robot from a stem or branch?
- Is the target kinematically reachable?
- Is the arm near a singularity?

It does not directly answer:
- Will the selected flower still be visible after a base turn?
- Will a leaf occlude the peduncle?
- Will target identity become ambiguous among several similar flowers?
- Will the target reappear after temporary self-occlusion?
- Will a small visually selected structure remain trackable when the eye-in-hand camera moves?

These are predictive visual-state questions.

### 2.2 Framewise segmentation is not persistent identity

The current Piper-Scout pipeline has strong single-frame perception components, but target state is still largely constructed from framewise masks, centroids, depth, and cached skeleton heuristics. This is vulnerable to:
- segmentation flicker;
- mask merging/splitting;
- camera motion;
- plant sway;
- depth dropout on thin structures;
- temporary occlusion;
- target switching between visually similar instances.

### 2.3 Existing action-conditioned JEPA work changes the novelty boundary

V-JEPA 2-AC already demonstrates action-conditioned latent prediction and image-goal robot planning. Therefore the following claim is **not sufficient novelty**:

> “We use V-JEPA for robot planning.”

V-JEPA 2.1 further improves dense, temporally consistent visual representations. Therefore even:

> “We combine V-JEPA 2.1 with an action predictor”

is not strong enough by itself.

Piper-JEPA must instead establish a task-specific methodological contribution:

> **target-conditioned dense predictive state for exact-instance visibility and identity preservation under coordinated non-holonomic mobile-manipulator motion, fused with explicit semantic geometry and safety-bounded control.**

---

## 3. Central research question

> **Can a target-conditioned dense action-predictive video world model improve whole-body manipulation of thin, deformable plant targets by forecasting whether the exact language-selected instance will remain visible and identifiable after candidate base + arm motions?**

This breaks into five testable hypotheses.

### H1 — Dense temporal representation

V-JEPA 2.1 dense features will provide higher exact-target identity retention and lower temporal jitter than framewise segmentation, conventional tracking, and V-JEPA 2 features under camera motion, plant sway, and temporary occlusion.

### H2 — Action-conditioned prediction

A Piper-Scout action-conditioned predictor can forecast future target location, visibility, and identity confidence over short horizons better than action-free temporal extrapolation.

### H3 — Target-weighted learning

Target-weighted predictive training will outperform a global latent prediction objective because the target peduncle/flower occupies only a small fraction of the image/token grid.

### H4 — Predictive control benefit

Adding predicted target visibility/identity costs to a strong geometry-only whole-body MPC will improve target retention and grasp success specifically in visibility-sensitive scenes, without reducing geometric safety.

### H5 — Explicit deformation can complement, not replace, JEPA

When `plant_twin` has a valid fit, its explicit deformation state will provide interpretable auxiliary information that improves analysis and may improve control near contact. However, the main JEPA contribution should remain measurable even without `plant_twin`.

---

## 4. Proposed contributions

The paper should claim no more than the experiments support. The intended contribution set is:

### C1 — Task-conditioned dense target memory

A V-JEPA 2.1-based target-memory mechanism that binds language-selected target initialization to a persistent dense video representation and outputs:
- target identity confidence;
- image-space target distribution;
- target uncertainty;
- occlusion state;
- 3-D target estimate when valid depth is available.

### C2 — Target-weighted action-conditioned world model

An embodiment-aware action-conditioned predictor that forecasts dense future visual state under coordinated Scout + Piper motion, with explicit weighting of target and plant regions.

### C3 — JEPA-aware non-holonomic whole-body MPC

A controller that augments geometry-based whole-body planning with predicted:
- target visibility;
- target identity consistency;
- target uncertainty;

while retaining explicit semantic SDF constraints and non-holonomic Scout dynamics.

### C4 — Hierarchical safety architecture

A separation between:
- learned predictive costs;
- deterministic semantic geometry;
- hard safety projection;
- local near-contact controller.

### C5 — Plant-manipulation benchmark

A benchmark covering:
- similar-instance target selection;
- eye-in-hand motion;
- mobile-base motion;
- temporary occlusion;
- plant sway;
- thin-structure depth failure;
- whole-body reachability;
- final grasp outcome.

---

## 5. System architecture

```text
Language instruction
      |
      v
VLM / grounding
(class, ordinal, relation, attribute, action)
      |
      v
Initial target mask / instance ID
      |
      +------------------------+
      |                        |
      v                        v
V-JEPA 2.1 dense state     RGB-D geometry
target memory              semantic nvblox SDF
      |                        |
      v                        v
persistent target          collision / clearance
identity + uncertainty     metric target geometry
      |                        |
      +-----------+------------+
                  |
                  v
      action-conditioned JEPA
      future dense visual state
                  |
                  +-------------------+
                  |                   |
                  v                   v
      visibility / identity       plant_twin
      future costs                deformation state
                  |                   |
                  +---------+---------+
                            |
                            v
                  whole-body MPC
      Scout [v, omega] + Piper [qdot_1:6]
                            |
                            v
                    safety projection
                            |
                            v
              near-contact switching rule
                            |
                            v
              bounded MPPI / visual servo
                            |
                            v
                           grasp
```

---

## 6. Formal task definition

At time $t$, define the mobile-manipulator state

```math
x_t =
[x_{b,t},y_{b,t},\theta_{b,t},q_t]^T
```

where $q_t\in\mathbb R^6$ is the Piper joint configuration.

The Scout is a skid-steer base modelled as a unicycle (differential drive; slip identified in whole-body MPC experiment WE1), so the control is

```math
u_t =
[v_t,\omega_t,\dot q_t]^T
\in\mathbb R^8.
```

The base evolves as

```math
x_{b,t+1}
=
x_{b,t}
+
\Delta t\,v_t\cos\theta_{b,t},
```

```math
y_{b,t+1}
=
y_{b,t}
+
\Delta t\,v_t\sin\theta_{b,t},
```

```math
\theta_{b,t+1}
=
\theta_{b,t}
+
\Delta t\,\omega_t.
```

The arm evolves under velocity control:

```math
q_{t+1}
=
q_t+\Delta t\,\dot q_t.
```

Let:
- $I_t$: RGB frame;
- $D_t$: depth frame;
- $S_t$: semantic geometry;
- $z_t$: learned target-conditioned visual state;
- $\xi_t$: optional `plant_twin` deformation state;
- $\ell$: language instruction.

The control objective is to reach and grasp the target specified by $\ell$, while:
1. preserving target identity;
2. preserving sufficient target visibility;
3. satisfying collision and dynamics constraints;
4. limiting undesirable plant interaction;
5. maintaining local safety near contact.

---

## 7. Language-grounded target initialization

Map the language instruction to a structured query

```math
q_\ell=(c,k,r,\rho,a)
```

where:
- $c$: semantic class;
- $k$: ordinal;
- $r$: spatial relation;
- $\rho$: optional attribute;
- $a$: requested action.

Example:

> “grasp the third flower from the left”

becomes

```math
q_\ell =
(\text{flower},3,\text{left-to-right},\varnothing,\text{grasp}).
```

Suppose candidate masks are

```math
\mathcal M_t=\{M_t^1,\ldots,M_t^N\}.
```

For candidate $i$, compute centroid

```math
u_t^i
=
\frac{1}{|M_t^i|}
\sum_{p\in M_t^i}p.
```

For a left-to-right ordinal command,

```math
\pi
=
\operatorname{argsort}(u_{t,x}^{1:N})
```

and select

```math
i^*=\pi_k.
```

The grounding module initializes identity. It should **not** be responsible for persistence after the robot starts moving.

---

## 8. Dense V-JEPA 2.1 target memory

### 8.1 Dense visual state

For a video context window

```math
\mathcal I_t
=
\{I_{t-L+1},\ldots,I_t\},
```

the frozen or lightly adapted encoder produces

```math
F_t
=
E_\theta(\mathcal I_t)
\in
\mathbb R^{H'\times W'\times d}.
```

The method retains the spatial token grid rather than globally pooling it.

### 8.2 Target descriptor

Given the initial selected target mask $`M_t^*`$, resize it to the feature grid and compute

```math
r_t
=
\frac{
\sum_p
\widetilde M_t^*(p)F_t(p)
}{
\sum_p
\widetilde M_t^*(p)+\epsilon
}.
```

A more robust implementation may keep a target prototype bank

```math
\mathcal R_t
=
\{r^{(1)},\ldots,r^{(m)}\}
```

rather than a single descriptor.

### 8.3 Target probability map

For the next frame,

```math
C_{t+1}(p)
=
\cos(F_{t+1}(p),r_t).
```

Convert to a spatial probability map:

```math
P_{t+1}(p)
=
\frac{
\exp(C_{t+1}(p)/\tau)
}{
\sum_q\exp(C_{t+1}(q)/\tau)
}.
```

Then

```math
\hat u_{t+1}
=
\sum_p
P_{t+1}(p)p.
```

### 8.4 Uncertainty

Use entropy

```math
H_{t+1}
=
-\sum_p
P_{t+1}(p)
\log P_{t+1}(p)
```

and optionally covariance

```math
\Sigma_{u,t+1}
=
\sum_p
P_{t+1}(p)
(p-\hat u_{t+1})
(p-\hat u_{t+1})^T.
```

### 8.5 3-D target state

For valid target depth $d_t$,

```math
p_c
=
d_tK^{-1}
[u,v,1]^T
```

and

```math
p_w
=
T_c^w p_c.
```

Depth is not forced when invalid. In thin-peduncle cases the system may retain an image-space target with high depth uncertainty rather than hallucinating metric precision.

---

## 9. Action representation for Piper-Scout prediction

The optimizer chooses

```math
u_t=
[v_b,\omega_b,\dot q_1,\ldots,\dot q_6].
```

Map that action into approximate task-space consequences:

```math
a_t
=
\Gamma(x_t,u_t)
```

with

```math
a_t=
[
\Delta s_b,
\Delta\theta_b,
\Delta p_e,
\Delta r_e,
\Delta g
]^T.
```

Components:
- $\Delta s_b$: forward base displacement;
- $\Delta\theta_b$: base yaw;
- $\Delta p_e\in\mathbb R^3$: EE translation;
- $\Delta r_e\in\mathbb R^3$: EE orientation increment;
- $\Delta g$: gripper command/state.

This preserves the non-holonomic structure of Scout motion while making the predictor less dependent on Piper joint numbering.

---

## 10. Action-conditioned Piper-JEPA predictor

Let $Z_t$ denote the dense learned visual state.

The one-step predictor is

```math
\hat Z_{t+1}
=
P_\phi(
Z_{t-K+1:t},
a_t,
s_t
),
```

where $s_t$ is proprioceptive state.

For planning horizon $H$,

```math
\hat Z_{t+1:t+H}
=
P_\phi(
Z_t,
a_{t:t+H-1},
s_t
).
```

The predictor should be evaluated separately from the controller. A controller improvement without an accurate prediction benchmark would make causal interpretation weak.

---

## 11. Target-weighted predictive learning

### 11.1 Motivation

A peduncle may occupy only a handful of feature patches. A global latent error can be numerically dominated by background, leaves, pot, greenhouse structure, and robot body.

### 11.2 Spatial weighting

Define

```math
w_t(p)
=
1
+
\lambda_T M_t^{target}(p)
+
\lambda_P M_t^{plant}(p).
```

Typically

```math
\lambda_T>\lambda_P>0.
```

### 11.3 One-step term

```math
\mathcal L_{TF}
=
\sum_p
w_t(p)
\left\|
\hat Z_{t+1}(p)
-
\operatorname{sg}
Z_{t+1}(p)
\right\|_1.
```

### 11.4 Rollout term

```math
\mathcal L_{roll}
=
\sum_{k=1}^H
\gamma^{k-1}
\sum_p
w_{t+k}(p)
\left\|
\hat Z_{t+k}(p)
-
\operatorname{sg}
Z_{t+k}(p)
\right\|_1.
```

### 11.5 Total objective

```math
\boxed{
\mathcal L_{AC}
=
\mathcal L_{TF}
+
\lambda_R\mathcal L_{roll}
}
```

Potential auxiliary terms, only if required by experiments:
- target-center consistency;
- visibility classification;
- target-ID contrastive consistency;
- uncertainty calibration.

---

## 12. Predictive target visibility and identity

From predicted dense features:

```math
\hat C_{t+k}(p)
=
\cos(
\hat Z_{t+k}(p),
r_t
)
```

and

```math
\hat P_{t+k}(p)
=
\operatorname{softmax}
(
\hat C_{t+k}(p)/\tau
).
```

Predicted image location:

```math
\hat u_{t+k}
=
\sum_p
\hat P_{t+k}(p)p.
```

Predicted entropy:

```math
\hat H_{t+k}
=
-\sum_p
\hat P_{t+k}(p)
\log\hat P_{t+k}(p).
```

Let $\hat r_{t+k}$ be the predicted target-region descriptor. Identity consistency is

```math
c_{id,t+k}
=
\cos(
\hat r_{t+k},
r_t
).
```

This allows the planner to ask:

> “If the Scout moves forward 10 cm and turns 8 degrees while the wrist rotates, will the selected flower remain uniquely identifiable?”

---

## 13. Explicit semantic geometry

Maintain semantic signed-distance functions

```math
\phi_c(x)
```

for:
- non-target stem;
- branch;
- leaf;
- target.

Recommended policy:
- non-target stem/branch: hard collision;
- leaf: soft interaction cost;
- selected target: excluded from non-target hard collision only in an explicitly gated grasp phase.

For robot collision primitive $c_j(x)$, require

```math
\phi_{hard}(c_j(x))
\ge
d_{safe}.
```

The learned visual predictor must never be allowed to override this hard condition.

---

## 14. Integration of `plant_twin`

### 14.1 Role

`plant_twin` is an explicit deformable model, not a learned visual world model.

It provides interpretable quantities such as:
- stem centerline (Catmull-Rom control points) and length;
- leaf rigid pose and bending field (RBF height lattice);
- leaf edge stretch relative to the rest mesh;
- contact-conditioned deformation (fingertip contact residual);
- per-frame fit cost.

Current implementation limits (`Codes/src/plant_twin/`): one leaf and its stem per fitter instance; ≈28 ms/frame on a 4-core x86 dev CPU (Orin not yet measured); untested on hardware; the node publishes markers and the petiole point but not yet a fit-confidence/timestamp message.

### 14.2 Optional control term

When fit confidence is valid. Note that `plant_twin` fits the *current* deformation and has no forward model, so $E(k)$ for future horizon steps must come from an explicit approximation, for example quasi-static re-fits with the candidate's predicted fingertip position as the contact constraint at a few horizon knots. The paper must state which approximation is used:

```math
J_{deform}
=
\sum_k
[
\alpha_sE_{stem}(k)
+
\alpha_lE_{stretch}(k)
+
\alpha_bE_{bend}(k)
].
```

### 14.3 Scientific use

More importantly, `plant_twin` enables the analysis question:

> Do changes in JEPA latent state correlate with physically interpretable deformation modes?

Possible analyses:
- canonical correlation between latent change and bending parameters;
- regression from frozen latent deltas to explicit deformation state;
- stratification of prediction error by low/high fitted deformation.

### 14.4 Validity gating

Every `plant_twin` state used by the controller must have:
- timestamp;
- fit residual/confidence;
- freshness threshold.

Invalid or stale deformation state is omitted from the objective.

---

## 15. Geometry-only whole-body MPC baseline

Before JEPA-aware control, build a strong deterministic baseline.

```math
J_{geo}
=
w_gJ_{goal}
+
w_cJ_{collision}
+
w_lJ_{leaf}
+
w_mJ_{manip}
+
w_bJ_{base}
+
w_sJ_{smooth}
+
w_dJ_{deform}.
```

### Goal term

```math
J_{goal}
=
\sum_{k=1}^H
\|p^e_{t+k}-p_g\|_Q^2
+
\lambda_R
d_{SO(3)}^2
(
R^e_{t+k},R_g
).
```

### Manipulability

```math
m(q)
=
\sqrt{
\det(
J_a(q)J_a(q)^T
)
}
```

with

```math
J_{manip}
=
\sum_k
\frac{1}{
m(q_{t+k})+\epsilon
}.
```

### Base-motion regularization

```math
J_{base}
=
\sum_k
(
v_{b,k}^2
+
\lambda_\omega
\omega_{b,k}^2
).
```

---

## 16. JEPA-aware whole-body MPC

Optimize

```math
U^*
=
\arg\min_U
J(U)
```

with

```math
\boxed{
J
=
w_gJ_{goal}
+
w_vJ_{vis}
+
w_iJ_{id}
+
w_cJ_{collision}
+
w_lJ_{leaf}
+
w_mJ_{manip}
+
w_bJ_{base}
+
w_sJ_{smooth}
+
w_dJ_{deform}
}
```

where the novel predictive terms are $J_{vis}$ and $J_{id}$.

### Visibility cost

```math
J_{vis}
=
\sum_k
\left[
\|
\hat u_{t+k}
-
u_{des}
\|^2
+
\lambda_H
\hat H_{t+k}
+
B_{FoV}
(
\hat u_{t+k}
)
\right].
```

### Identity cost

```math
J_{id}
=
\sum_k
[
1-
\cos(
\hat r_{t+k},
r_t
)
].
```

The key experiment is whether these predictive terms improve performance over $J_{geo}$ in scenes where geometry is valid but visibility is trajectory-dependent.

---

## 17. Safety projection

The planner outputs nominal control $u_{MPC}$.

Execution uses

```math
u_{safe}
=
\arg\min_u
\|
u-u_{MPC}
\|_2^2
```

subject to:
- semantic hard-clearance constraints;
- Piper joint limits;
- base velocity/acceleration limits;
- arm velocity/acceleration limits;
- force/contact threshold (needs a force source; the platform has no wrist F/T sensor yet, see [`../README.md`](../README.md));
- state freshness;
- watchdog constraints.

A CBF-like formulation can use

```math
h_j(x)
=
\phi_{hard}(c_j(x))
-
d_{safe}
```

with

```math
\dot h_j
+
\alpha h_j
\ge
0.
```

### Formal-guarantee caution

Do not claim formal safety guarantees unless the final implementation explicitly states assumptions such as:
- obstacle-field regularity;
- bounded sensing delay;
- bounded state-estimation error;
- feasible safety QP;
- control-rate bounds.

Until then, describe this as a **safety projection/filter**.

---

## 18. MPC-to-servo handoff

Switch from whole-body MPC to local control when

```math
d(p_e,p_g)
<
d_{switch},
```

```math
H_t<H_{max},
```

```math
c_t>c_{min},
```

and semantic clearance is valid.

With the eye-in-hand camera, $d_{switch}$ must account for the sensor's minimum depth: below it the local controller runs on the image-space target distribution and the last valid metric estimate (D435), or keeps live depth longer (D405).

After switching:
- freeze the Scout or heavily penalize base motion;
- run the local controller at the highest sustainable rate;
- enforce force/contact bounds;
- retract/abort on loss of confidence or hard-clearance violation.

---

## 19. Data-collection plan

### 19.1 Episode contents

Each rosbag2 episode should include:
- RGB;
- depth aligned to colour (`/camera/aligned_depth_to_color/image_raw`);
- CameraInfo;
- TF;
- Piper joint state;
- Scout odometry;
- segmentation masks;
- selected target identity;
- target prompt/query;
- executed base and arm commands;
- target state;
- semantic SDF state or reconstructible inputs;
- force/contact (when a force source exists; otherwise gripper state);
- `plant_twin` parameters + fit confidence when valid;
- task outcome.

### 19.2 Scene factors

Systematically vary:
- number of similar flowers: 1 / 3 / 5 / 7;
- foliage density;
- target thickness/size;
- partial occlusion;
- temporary full occlusion;
- plant sway amplitude;
- arm-only camera motion;
- base-only camera motion;
- combined whole-body motion;
- illumination/background;
- base start pose;
- target reachability category.

### 19.3 Train/validation/test split

Split by **plant instance and scene arrangement**, not random frame.

### 19.4 Robot-data budgets

Evaluate approximately:
- 1 hour;
- 2 hours;
- 5 hours;
- 10 hours;

or the closest feasible balanced subsets.

---

## 20. Experiment 1 — Dense target persistence

### Goal

Test H1 independently of control.

### Conditions
- stationary robot / stationary plant;
- plant sway;
- arm eye-in-hand motion;
- Scout motion;
- combined motion;
- temporary occlusion;
- segmentation dropout;
- depth dropout;
- lighting change.

### Baselines
1. framewise segmentation;
2. optical flow / conventional tracker;
3. strong dense self-supervised feature tracker;
4. V-JEPA 2;
5. V-JEPA 2.1.

### Metrics

```math
E_{2D}
=
\|
\hat u-u^*
\|_2.
```

```math
E_{3D}
=
\|
\hat p-p^*
\|_2.
```

```math
A_{ID}
=
\frac{
\#\text{frames with correct physical instance}
}{
\#\text{evaluated frames}
}.
```

Also report:
- false target switches;
- occlusion recovery;
- temporal jitter;
- mask IoU where applicable;
- confidence calibration.

### Success condition

V-JEPA 2.1 must beat the strongest practical baseline on at least target-ID retention or occlusion recovery with a meaningful effect size.

---

## 21. Experiment 2 — Language-selected instance persistence

Scenes contain

```math
N\in\{3,5,7\}
```

similar flowers.

Commands include:
- “third flower from the left”;
- “rightmost flower”;
- “flower above the large leaf”;
- “longest visible peduncle”;
- “the same flower as before.”

Initial grounding:

```math
A_{ground}
=
\frac{
N_{correct}
}{
N_{commands}
}.
```

Persistence:

```math
A_{persist}(\tau)
=
P(
\hat i_{t+\tau}=i_t^*
).
```

Separate initial grounding errors from persistence errors.

---

## 22. Experiment 3 — Action-conditioned prediction

### Trajectory classes
- arm-only;
- base-only;
- simultaneous base+arm;
- approach;
- retreat;
- viewpoint change;
- occlusion/reappearance;
- mild contact-induced plant motion.

### Methods
1. action-free temporal predictor;
2. V-JEPA 2-AC-style baseline;
3. V-JEPA 2.1 + generic dense predictor;
4. V-JEPA 2.1 + target-weighted Piper-JEPA predictor.

### Horizons

```math
H\in\{1,2,4,8\}.
```

### Metrics

```math
E_{target}(H)
=
\frac1H
\sum_{k=1}^H
\|
\hat u_{t+k}
-
u_{t+k}
\|_2.
```

Also report:
- visibility F1/AUROC;
- future target-ID accuracy;
- entropy calibration;
- rollout degradation;
- dense latent error;
- latency vs horizon.

### Key result

Target-weighted prediction should improve target-specific metrics even when global latent error changes little.

---

## 23. Experiment 4 — Geometry-only whole-body planning

### Methods
1. arm-only;
2. sequential base → arm;
3. holistic/reactive QP;
4. geometry-only whole-body MPC.

### Reach categories
1. comfortably arm reachable;
2. near workspace boundary;
3. requires base motion;
4. multiple collision-free paths with visibility differences.

### Metrics
- planning success;
- execution success;
- time;
- Scout travel;
- EE path length;
- minimum clearance;
- replans;
- target-loss rate.

This experiment must be complete before claiming JEPA-aware control benefit.

---

## 24. Experiment 5 — JEPA-aware predictive whole-body control

| ID | Method | Dense 2.1 | Action predictor | Target weighting | Semantic SDF | Whole-body |
|---|---|---:|---:|---:|---:|---:|
| B | Geometry-only MPC | no | no | no | yes | yes |
| C | V-JEPA 2 action-conditioned | no | yes | no | yes | yes |
| D | V-JEPA 2.1 generic predictor | yes | yes | no | yes | yes |
| E | Piper-JEPA | yes | yes | yes | yes | yes |

### Critical visibility-sensitive scenes

Design scenes where:
- all candidate trajectories satisfy geometry constraints;
- at least one path causes target occlusion or identity ambiguity;
- another path preserves target visibility.

### Primary metrics
- target retained until handoff;
- first-attempt grasp success;
- final grasp success;
- target-loss events;
- execution time;
- safety metrics.

### Required interpretation

A gain only in reachability is not enough; geometry-only whole-body control can already explain that. The JEPA contribution should be strongest in visibility-sensitive cases.

---

## 25. Experiment 6 — Full closed-loop grasping

### Suggested scope

Approximately 40–50 scene configurations balanced across:
- target ordinal;
- foliage density;
- target size;
- reach category;
- plant motion;
- base start pose.

### Primary endpoint

**First-attempt grasp success.**

### Secondary endpoints
- final success after recovery;
- correct-instance grasp;
- target-loss rate;
- approach duration;
- recovery count;
- Scout motion;
- max contact force;
- min clearance;
- safety-filter interventions.

---

## 26. Safety ablation

Compare:

1. predictive planner only;
2. + semantic SDF;
3. + safety projection;
4. + local near-contact servo.

Report:
- collision/intervention rate;
- max contact force;
- min clearance;
- grasp success.

The desired result is **higher success without increased force or reduced clearance**.

---

## 27. `plant_twin` ablation

Evaluate:
- Piper-JEPA without `plant_twin`;
- Piper-JEPA with deformation-state cost;
- Piper-JEPA with deformation state used only for analysis.

Questions:
1. Does explicit deformation improve final-contact behavior?
2. Does it improve prediction of post-contact target motion?
3. Does it add enough value to justify real-time compute?
4. Does JEPA already encode useful deformation cues without explicit fitting?

If no significant benefit is found, `plant_twin` remains an interpretation/evaluation tool rather than part of the final controller.

---

## 28. Embedded deployment study

Profile on Jetson AGX Orin.

### Encoder
- ViT-B/16 384;
- ViT-L/16 384.

Measure:
- clip latency;
- effective target-state FPS;
- peak unified memory;
- GPU utilization;
- power;
- thermal throttling.

### Predictor
Measure versus $H$:
- rollout latency;
- memory;
- prediction quality.

### MPC
Measure:
- median solve time;
- p95/p99 solve time;
- deadline misses;
- stale-prediction rate.

Do not state a control frequency in the paper unless measured on the deployed stack.

---

## 29. Statistical analysis

### Binary grasp outcome

```math
\operatorname{logit}
P(Y=1)
=
\beta_0
+
\beta_1Method
+
\beta_2Occlusion
+
\beta_3Motion
+
\beta_4Reachability
+
b_{plant}.
```

Possible random effects:
- plant instance;
- day/session;
- operator if relevant.

### Continuous outcomes

For time, target error, clearance, contact force, and path length, use linear mixed models when assumptions are acceptable.

Otherwise use:
- bootstrap confidence intervals;
- permutation tests;
- rank-based comparisons.

Always report:
- effect size;
- 95% CI;
- physical-trial count;
- number of unique plant/scene instances.

---

## 30. Falsifiable outcomes and kill criteria

### Kill criterion A — representation

If V-JEPA 2.1 does not materially outperform simpler tracking in Experiment 1, do not make it the central method.

### Kill criterion B — prediction

If target-specific predictive error is not better than action-free or generic prediction, do not claim useful action-conditioned target forecasting.

### Kill criterion C — control

If geometry-only MPC matches Piper-JEPA in visibility-sensitive scenes, learned prediction does not justify the added control complexity.

### Kill criterion D — embedded feasibility

If Orin latency makes prediction too stale for control, reposition JEPA as a slower active-perception / planning prior rather than a real-time MPC term.

These criteria protect the project from confirmation bias.

---

## 31. Implementation mapping

### Existing packages

`stem_grasp`
- segmentation;
- point-cloud filtering;
- skeleton/candidate logic;
- current adaptive servo baseline.

`scout_piper_scene_repr`
- nvblox integration (RealSense → nvblox validated);
- class demux + semantic class policy YAML;
- CPU `SemanticVoxelMap` + `SemanticDistanceQuery` planner distance query (`python/scout_piper_scene_repr_py/`, v0 backend, 2026-10-06);
- MoveIt semantic collision plugin (FCL + the CPU distance field, P1.7.7; software-tested only).

`plant_twin`
- explicit leaf/stem deformation fitting (one leaf + stem, analytic-Jacobian LM).

`scout_piper_bringup`
- integrated system launch.

### `scout_piper_jepa` (Stage A implemented 2026-10-06; not yet in `full_system.launch.py`)

```text
Codes/src/scout_piper_jepa/scout_piper_jepa/
  encoder.py            # implemented: DenseEncoder, VJepaEncoder (torch.hub), ColorPatchEncoder (numpy reference)
  target_memory.py      # implemented: Stage A dense target memory
  metrics.py            # implemented: E1 tracking metrics
  episode.py            # implemented: rosbag2 → .npz export + offline E1 evaluation (covers rosbag_dataset / eval_tracking)
  image_codec.py        # implemented: sensor_msgs/Image ↔ numpy without cv_bridge
  action.py             # implemented: Stage B action embedding Γ(x,u)
  target_state_node.py  # implemented: ROS 2 node
  predictor.py          # planned (Stage B)
  predictive_cost.py    # planned (Stage C; plugs into WholeBodyCost.extra)
  uncertainty.py        # planned
  train_predictor.py    # planned
  eval_prediction.py    # planned
```

### `scout_piper_whole_body_mpc` (geometry-only baseline implemented 2026-10-06; not yet in `full_system.launch.py`)

Layout is owned by the whole-body MPC track ([`../whole_body_mpc/WHOLE_BODY_MPC_RESEARCH_PLAN.md`](../whole_body_mpc/WHOLE_BODY_MPC_RESEARCH_PLAN.md) §23). Piper-JEPA does not add modules there: the visibility and identity terms live in `scout_piper_jepa/predictive_cost.py` and are passed to the MPC through the `extra` hook of `costs/terms.py`, so the geometry-only baseline (`WB:W4` / `C2`) and Piper-JEPA (`C3`) share identical dynamics, solver, geometry costs and safety layer.

---

## 32. Recommended ROS interfaces

### Target state

`/piper_jepa/target_state`

Fields:
- timestamp;
- target ID;
- image mean;
- image covariance;
- 3-D mean;
- 3-D covariance;
- confidence;
- entropy;
- visible flag;
- valid-depth flag.

Implemented in Stage A (2026-10-06) by `scout_piper_jepa/target_state_node.py` as `std_msgs/String` JSON (`stamp`, `max_age`, `status`, `visible`, `confidence`, `entropy`, `u_mean`, `u_cov`, `p_world`, `p_cov`; `p_world` is null without valid depth), plus `/piper_jepa/target_point` (`PointStamped`, valid depth only) and `/piper_jepa/target_visible` (`Bool`). Re-grounding goes through `/piper_jepa/init_mask`. No target-ID field is published yet.

### Predictor output

`/piper_jepa/prediction_status`

Avoid streaming entire dense rollouts over ordinary ROS messages in the normal control path. Keep tensor-heavy prediction and MPC cost evaluation in-process or use zero-copy where practical.

### Planner output

Implemented in `scout_piper_whole_body_mpc/controller_node.py` (2026-10-06) without a combined command topic. After the safety filter, base $v,\omega$ go to `/cmd_vel` for `scout_ros2` and arm joint velocities go to `moveit_servo` as `control_msgs/JointJog` on `/servo_node/delta_joint_cmds`. With `execute: false` (the default) both go to `/whole_body_mpc/preview/*` instead. `/whole_body_mpc/status` (`std_msgs/String` JSON) carries controller mode, TCP error, safety reason, solve time and state/geometry ages; `/whole_body_mpc/plan` (`nav_msgs/Path`) carries the predicted TCP path.

---

## 33. Expected paper figures

### Figure 1 — Problem and architecture

Show:
1. language command;
2. selected flower among similar instances;
3. candidate whole-body trajectories;
4. one path preserves target visibility while another causes occlusion.

### Figure 2 — Dense target memory

Show:
- RGB sequence;
- target-mask initialization;
- V-JEPA 2.1 similarity map;
- occlusion;
- successful re-identification.

### Figure 3 — Action-conditioned prediction

For candidate actions:
- predicted future target map;
- actual future frame;
- target-position error;
- predicted entropy.

### Figure 4 — Piper-JEPA MPC

Show:
- action samples;
- robot rollout;
- semantic SDF;
- future visual rollout;
- visibility + identity cost;
- selected trajectory.

### Figure 5 — Representation/prediction results

Plots:
- target-ID retention;
- occlusion recovery;
- prediction error vs horizon;
- target-weighted vs generic prediction.

### Figure 6 — Robot results

Grouped by reachability/visibility category:
- grasp success;
- target loss;
- completion time;
- safety metrics.

### Extended data
- Orin profiling;
- `plant_twin` ablation;
- additional horizons;
- ViT-B vs ViT-L;
- failure cases.

---

## 34. Recommended paper tables

### Table 1 — Method comparison

Columns:
- language target;
- dense temporal state;
- action-conditioned prediction;
- metric geometry;
- whole-body control;
- hard safety;
- deformable-state modeling.

### Table 2 — Main ablation

| Method | ID retention | Prediction error | Target loss | Grasp success | Min clearance |
|---|---:|---:|---:|---:|---:|

### Table 3 — Embedded performance

| Model | Encoder latency | Predictor latency | MPC latency | Memory | Power |
|---|---:|---:|---:|---:|---:|

---

## 35. Reviewer-facing novelty statement

> **We introduce Piper-JEPA, a task-conditioned predictive-control framework for language-grounded mobile manipulation of thin, deformable plant structures. Unlike prior latent world-model planning that primarily optimizes global visual goal similarity for stationary manipulators, Piper-JEPA maintains a dense persistent representation of the exact language-selected target, predicts how coordinated skid-steer base and arm motions alter that target’s future visibility and identity, and incorporates those predictions as soft objectives inside explicit semantic RGB-D whole-body MPC. Learned prediction augments rather than replaces geometry: semantic signed-distance constraints, force limits, and a separate high-rate local controller retain authority over collision avoidance and final contact.**

This wording should be weakened if experiments do not directly establish all clauses.

---

## 36. What not to claim

Do not claim:
- V-JEPA 2.1 itself is novel;
- action-conditioned JEPA planning is novel;
- a specific official “V-JEPA 2.1-AC” packaged checkpoint exists unless explicitly verified at submission time;
- formal safety guarantees without assumptions and proof;
- 100/200 Hz performance before Orin measurement;
- generalized plant physics understanding from one species or a small dataset;
- `plant_twin` is ground truth unless externally validated.

---

## 37. Publication strategy

### Primary target

The strongest single paper is the combined Piper-JEPA story:
- dense target persistence;
- action-conditioned target prediction;
- geometry-aware whole-body MPC;
- full robot validation.

Potential venues depending maturity/deadline:
- ICRA;
- IROS;
- RSS;
- RA-L.

### Extended journal

A T-RO extension becomes justified if it adds:
- larger cross-species benchmark;
- more rigorous safety analysis;
- `plant_twin` deformation integration;
- active perception;
- extensive embedded deployment study.

### Avoid fragmentation

Do not publish every phase as a separate small paper if doing so weakens the main result. Phase 2A, 2B and 3A should primarily serve as controlled baselines/building blocks for Phase 3B unless independently strong.

---

## 38. Reproducibility checklist

Before submission:
- freeze dataset split;
- version model checkpoints;
- record all ROS parameters;
- record camera calibration;
- record hand-eye calibration;
- record Orin power mode;
- record CUDA / TensorRT / PyTorch versions;
- fix seeds where meaningful;
- release exact evaluation scripts;
- include unsuccessful physical trials;
- log safety interventions;
- define target identity annotation protocol;
- document exclusion criteria before analysis.

---

## 39. Immediate next implementation tasks

1. Build `scout_piper_jepa` package skeleton (done 2026-10-06, P2A.1–P2A.4).
2. Add rosbag2 dataset recorder for RGB-D + TF + robot state + target ID.
3. Run V-JEPA 2.1 ViT-B inference offline on recorded Piper-Scout video.
4. Implement mask-initialized dense target descriptor and similarity map (done 2026-10-06 with the numpy reference encoder; V-JEPA not yet run, item 3).
5. Compare against the current framewise target and one strong conventional tracker.
6. Measure target-ID retention under arm motion and temporary occlusion.
7. Only after H1 is supported, begin embodiment-specific action-conditioned post-training.
8. In parallel, finish Phase 3A geometry-only whole-body MPC so Piper-JEPA has a strong deterministic comparator (offline baseline in place 2026-10-06, P3A.1–P3A.5; P3A.6–P3A.8 open).

---

## 40. References

### V-JEPA
- Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985, 2025.
- Mur-Labadia et al., *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning*, arXiv:2603.14482, 2026.
- Official implementation: https://github.com/facebookresearch/vjepa2

### Mobile manipulation / planning
- Haviland & Corke, holistic mobile manipulation / NEO.
- cuRobo / NVIDIA Isaac ROS cuMotion.
- MPPI-VS.
- EHC-MM.
- RMMI.

### Language-conditioned robotics
- PaLM-E.
- RT-2.
- OK-Robot.
- MoMa-LLM.
- PhysVLM.

### Active perception / geometry
- nvblox.
- VGGT.
- Next Best Sense.
- ActiveSplat.

---

## 41. One-sentence project definition

> **Piper-JEPA predicts whether the exact object a human asked for will remain identifiable and manipulable after each candidate whole-body robot motion, then lets explicit geometry and a safety-bounded controller decide what is safe to execute.**
