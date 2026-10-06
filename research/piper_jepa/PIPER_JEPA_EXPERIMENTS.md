# Piper-JEPA Experiments and Evaluation Protocol

**Companion to:** PIPER_JEPA_RESEARCH_PLAN.md  
**Purpose:** executable experimental specification for dataset collection, ablations, robot trials, statistics, and paper tables  
**Status:** initial protocol; freeze a version before final data collection

---

## 1. Experimental objective

The experiments must answer four separate questions without conflating them:

1. **Representation:** does V-JEPA 2.1 improve exact-instance target persistence?
2. **Prediction:** does action conditioning improve future target visibility/identity prediction?
3. **Control:** does predictive target state improve whole-body manipulation beyond strong geometry-only MPC?
4. **Safety/deployment:** does the learned component improve task performance without degrading clearance, force, latency, or system reliability?

The strongest result is not simply higher grasp success. The paper should demonstrate a causal chain:

```math
\text{better target state}
\rightarrow
\text{better future target prediction}
\rightarrow
\text{better trajectory choice}
\rightarrow
\text{better grasp outcome}
```

while explicit geometry and the safety layer remain authoritative.

---

## 2. Experimental hierarchy

Use the following IDs consistently in code, logs, plots, and the paper:

| ID | Name | Main hypothesis |
|---|---|---|
| E1 | Dense target persistence | H1 |
| E2 | Language-selected instance persistence | H1 |
| E3 | Action-conditioned future prediction | H2, H3 |
| E4 | Geometry-only whole-body control | deterministic baseline |
| E5 | JEPA-aware predictive whole-body control | H4 |
| E6 | Full closed-loop grasping | H4 |
| E7 | Safety hierarchy ablation | safety claim |
| E8 | plant_twin deformation ablation | H5 |
| E9 | Embedded deployment benchmark | feasibility |

E1-E4 should be completed before the final E5/E6 robot campaign.

---

## 3. Episode and dataset schema

Every recorded episode should have a globally unique episode_id.

Recommended naming:

~~~
YYYYMMDD_sessionXX_sceneYY_trialZZ_methodNAME
~~~

Each episode should record enough information to replay perception and analyze control offline.

### 3.1 Required ROS data

Record:
- RGB image;
- aligned depth image;
- CameraInfo;
- TF tree;
- Piper joint states;
- Piper command velocities/trajectories;
- Scout odometry;
- Scout commanded v and omega;
- segmentation masks;
- target initialization mask;
- selected target ID;
- target prompt/query;
- V-JEPA target state;
- target confidence and entropy;
- nvblox / semantic scene inputs or exported state sufficient to reconstruct clearance;
- force/wrench;
- safety-filter state;
- controller mode;
- final task outcome.

### 3.2 Optional but recommended data

When available:
- plant_twin parameter vector;
- plant_twin fit cost/confidence;
- plant_twin contact state;
- gripper state;
- battery/power mode;
- GPU utilization;
- GPU memory;
- power draw;
- system temperature.

### 3.3 Metadata file per episode

Store a compact YAML/JSON record containing:

~~~
episode_id
date
session_id
plant_id
scene_id
method_id
random_seed
target_instance_id
language_prompt
target_class
target_ordinal
occlusion_level
foliage_density
motion_condition
reachability_class
lighting_condition
camera_calibration_id
hand_eye_calibration_id
software_commit
vjepa_checkpoint
predictor_checkpoint
orin_power_mode
trial_valid
exclusion_reason
outcome
~~~

---

## 4. Annotation protocol

### 4.1 Physical target identity

Every candidate flower/peduncle in a scene receives a stable physical instance ID before the trial.

Example:

~~~
flower_01
flower_02
flower_03
...
~~~

The selected target identity is frozen at command time.

A later prediction is considered correct only if it corresponds to the same physical instance, not merely the same semantic class.

### 4.2 Image-space annotation

For evaluation frames annotate:
- target visible: yes/no;
- target center;
- target mask where feasible;
- peduncle centerline where feasible;
- occlusion fraction category.

Recommended occlusion bins:
- O0: 0-10%;
- O1: 10-35%;
- O2: 35-65%;
- O3: 65-90%;
- O4: >90% / effectively invisible.

### 4.3 3-D annotation

Where depth or multi-view reconstruction permits, store:
- 3-D target point;
- local peduncle direction;
- uncertainty/validity flag.

Do not treat unreliable RealSense depth on thin peduncles as ground truth.

### 4.4 Trial exclusion

Predefine valid exclusion reasons:
- hardware emergency stop unrelated to tested method;
- sensor recording failure;
- calibration failure detected before analysis;
- human interference;
- target physically breaks before method execution;
- corrupted rosbag.

Do not exclude ordinary method failures.

---

## 5. Scene-factor design

The benchmark should span independent difficulty axes.

### 5.1 Number of similar instances

```math
N\in\{1,3,5,7\}
```

### 5.2 Reachability class

R1 — comfortably reachable by arm only.  
R2 — near arm workspace boundary.  
R3 — requires Scout repositioning.  
R4 — multiple geometrically feasible paths, but visibility differs substantially.

R4 is the most important class for Piper-JEPA.

### 5.3 Motion condition

M0 — static.  
M1 — natural low-amplitude sway.  
M2 — induced moderate sway.  
M3 — eye-in-hand arm motion.  
M4 — Scout motion.  
M5 — combined base + arm motion.

### 5.4 Occlusion condition

O0-O4 from Section 4.2.

### 5.5 Foliage density

F1 — sparse.  
F2 — moderate.  
F3 — dense.

### 5.6 Lighting/background

At minimum:
- stable indoor;
- brighter side illumination;
- lower contrast / cluttered background.

Do not over-expand the factor space before the core hypotheses are validated.

---

## 6. Dataset split rules

### 6.1 No frame leakage

Never split neighboring frames from the same episode across train and test.

### 6.2 No plant-instance leakage

Primary evaluation split should be by plant instance and scene arrangement.

Recommended initial split:

- 60% train;
- 20% validation;
- 20% held-out test;

with all episodes from a plant/arrangement assigned to one split.

### 6.3 Predictor data-efficiency subsets

Construct nested subsets:

```math
D_1 \subset D_2 \subset D_5 \subset D_{10}
```

corresponding approximately to 1 h, 2 h, 5 h, and 10 h of robot interaction.

Use the same validation/test set for all data-budget comparisons.

---

## 7. Method IDs

Use fixed method names throughout the project.

### Tracking / representation methods

T0 — framewise segmentation.  
T1 — conventional tracker / optical-flow baseline.  
T2 — strong dense self-supervised feature tracker.  
T3 — V-JEPA 2 target memory.  
T4 — V-JEPA 2.1 target memory.

### Prediction methods

P0 — action-free temporal extrapolation.  
P1 — V-JEPA 2 action-conditioned baseline.  
P2 — V-JEPA 2.1 generic dense predictor.  
P3 — Piper-JEPA target-weighted predictor.

### Control methods

C0 — current/legacy arm-only pipeline.  
C1 — sequential Scout then arm.  
C2 — geometry-only whole-body MPC.  
C3 — JEPA-aware whole-body MPC.  
C4 — Piper-JEPA + safety projection + local servo.

Do not rename methods after seeing results.

---

## 8. E1 — Dense target persistence

### 8.1 Objective

Test whether V-JEPA 2.1 improves exact physical target persistence independent of action prediction and MPC.

### 8.2 Recommended dataset size

**Minimum:** 240 clips.  
**Preferred:** 360-480 clips.

A practical preferred design:

- 12 distinct plant/scene arrangements;
- 6 motion/occlusion conditions;
- 5 repeats;

```math
12\times6\times5=360
```

clips.

Clip duration: approximately 8-15 s.

### 8.3 Conditions

Include:
- static;
- arm motion;
- base motion;
- combined motion;
- partial occlusion;
- temporary full occlusion and reappearance.

Depth dropout and lighting shifts can be layered onto a subset.

### 8.4 Compared methods

T0-T4.

### 8.5 Primary endpoint

Exact target-ID retention:

```math
A_{ID}
=
\frac{
\#\text{correct-instance evaluated frames}
}{
\#\text{evaluated frames}
}.
```

### 8.6 Secondary endpoints

2-D center error:

```math
E_{2D}
=
\|\hat u-u^*\|_2.
```

3-D error where valid:

```math
E_{3D}
=
\|\hat p-p^*\|_2.
```

Also:
- false switch rate;
- occlusion recovery rate;
- temporal jitter;
- mask IoU;
- entropy calibration.

### 8.7 Required plot

Target-ID retention versus occlusion level for T0-T4.

### 8.8 Pass criterion

T4 should show a meaningful improvement over the strongest non-JEPA practical baseline in ID retention and/or occlusion recovery. If not, H1 is not supported.

---

## 9. E2 — Language-selected instance persistence

### 9.1 Objective

Separate language grounding quality from temporal persistence.

### 9.2 Recommended size

**Minimum:** 200 commands.  
**Preferred:** 300-400 commands.

Use at least 30 distinct scenes with 3, 5, or 7 visually similar targets.

### 9.3 Prompt families

- ordinal: “third flower from the left”;
- extrema: “rightmost flower”;
- relational: “flower above the large leaf”;
- attribute: “longest visible peduncle”;
- continuity: “the same flower as before.”

### 9.4 Metrics

Initial grounding:

```math
A_{ground}
=
\frac{N_{correct}}{N_{commands}}.
```

Persistence after robot motion:

```math
A_{persist}(\tau)
=
P(
\hat i_{t+\tau}
=
i_t^*
).
```

Report both separately.

### 9.5 Error taxonomy

Classify failure as:
- G1: incorrect initial grounding;
- G2: correct grounding, later target switch;
- G3: target lost, no re-acquisition;
- G4: target re-acquired incorrectly;
- G5: geometry/depth failure despite correct visual identity.

---

## 10. E3 — Action-conditioned future prediction

### 10.1 Objective

Test H2 and H3 without controller confounds.

### 10.2 Data

Collect trajectories spanning:
- arm-only motion;
- base-only motion;
- combined motion;
- approach/retreat;
- viewpoint change;
- occlusion/reappearance;
- mild plant contact.

### 10.3 Training budgets

Evaluate P1-P3 at approximately:
- 1 h;
- 2 h;
- 5 h;
- 10 h.

### 10.4 Prediction horizons

```math
H\in\{1,2,4,8\}.
```

The physical time represented by H must be reported explicitly because it depends on sampling rate.

### 10.5 Primary target metric

```math
E_{target}(H)
=
\frac1H
\sum_{k=1}^{H}
\|
\hat u_{t+k}-u_{t+k}
\|_2.
```

### 10.6 Secondary metrics

- future target-ID accuracy;
- visibility F1/AUROC;
- entropy calibration;
- rollout degradation;
- target-region latent error;
- global latent error;
- latency versus H.

### 10.7 Critical ablation

Compare P2 versus P3.

If target weighting is useful, P3 should improve target-specific error even when global latent error changes little.

### 10.8 Recommended result plots

1. Target error versus horizon.
2. Target-ID accuracy versus horizon.
3. Target error versus robot-data budget.
4. Target-region error versus global error for P2/P3.

---

## 11. E4 — Geometry-only whole-body MPC

### 11.1 Objective

Build the deterministic comparator that Piper-JEPA must beat.

### 11.2 Methods

C0, C1, C2.

### 11.3 Recommended physical trial size

**Minimum:** 20 scene configurations × 3 methods × 2 repeats = 120 trials.  
**Preferred:** 30 scenes × 3 methods × 3 repeats = 270 trials.

Balance across R1-R4.

### 11.4 Primary endpoints

- task success;
- target reached;
- grasp success;
- time to grasp.

### 11.5 Secondary endpoints

- Scout travel;
- EE path length;
- minimum semantic clearance;
- replans;
- target-loss rate;
- contact force.

### 11.6 Required interpretation

C2 should establish that coordinated base+arm motion solves reachability failures. This prevents later Piper-JEPA gains from being incorrectly attributed merely to having a mobile base.

---

## 12. E5 — JEPA-aware predictive whole-body control

### 12.1 Objective

Test the central control hypothesis H4.

### 12.2 Methods

B — geometry-only whole-body MPC.  
C — V-JEPA 2 action-conditioned control.  
D — V-JEPA 2.1 generic dense predictive control.  
E — Piper-JEPA target-weighted predictive control.

### 12.3 Critical scene design

Use R4 visibility-sensitive scenes where:
- multiple trajectories are collision-free;
- one path is likely to hide or confuse the target;
- another path preserves visibility.

If all paths are equivalent in visibility, E5 cannot test the paper's main novelty.

### 12.4 Recommended size

**Minimum:** 20 R4 scenes × 4 methods × 3 repeats = 240 trials.  
**Preferred:** 30 R4 scenes × 4 methods × 3 repeats = 360 trials.

Use randomized method order within scene.

### 12.5 Primary endpoint

Target retained until MPC-to-servo handoff.

### 12.6 Co-primary robot endpoint

First-attempt correct-target grasp success.

### 12.7 Secondary endpoints

- target-loss events;
- false target switches;
- final success after recovery;
- execution time;
- Scout travel;
- EE path length;
- minimum clearance;
- maximum force;
- safety interventions.

### 12.8 Required comparison

The most important statistical comparison is:

```math
E\; \text{vs}\; B
```

on R4 scenes.

---

## 13. E6 — Full closed-loop grasping

### 13.1 Objective

Evaluate the complete language-to-grasp system over a broader scene distribution.

### 13.2 Recommended scene set

40-50 unique configurations spanning:
- R1-R4;
- F1-F3;
- N = 1, 3, 5, 7;
- several occlusion and motion conditions.

### 13.3 Methods

Use only the 3-4 most informative methods after E1-E5 to keep physical trial count feasible.

Recommended:
- C0 current pipeline;
- C2 geometry-only whole-body MPC;
- D V-JEPA 2.1 generic predictor;
- C4 full Piper-JEPA.

### 13.4 Trial count

Target approximately 250-400 physical executions total.

E5 trials may be included in E6 only if the protocol was prospectively compatible. Do not silently double-count outcomes as independent evidence.

### 13.5 Primary endpoint

First-attempt correct-target grasp success.

### 13.6 Secondary endpoints

- final success after recovery;
- correct target selected;
- target retained;
- time;
- recovery count;
- force;
- clearance;
- base travel;
- safety-filter activations.

---

## 14. E7 — Safety hierarchy ablation

Compare:

S0 — predictive planner only.  
S1 — + semantic SDF.  
S2 — + safety projection.  
S3 — + local near-contact servo.

Primary safety metrics:
- collision/contact violation count;
- minimum non-target clearance;
- maximum contact force;
- emergency stop count.

Task metric:
- grasp success.

Desired result:

```math
\text{success increases}
\quad \text{without} \quad
F_{max}\uparrow
\quad \text{or} \quad
d_{min}\downarrow.
```

Do not intentionally run unsafe variants outside a controlled, low-energy validation regime.

---

## 15. E8 — plant_twin deformation ablation

### 15.1 Variants

D0 — no plant_twin in controller.  
D1 — plant_twin used only for logging/analysis.  
D2 — valid plant_twin deformation cost included in local/whole-body objective.

### 15.2 Questions

1. Does explicit deformation reduce excessive leaf/stem displacement?
2. Does it reduce contact force?
3. Does it improve post-contact target prediction?
4. Does it add meaningful compute cost?
5. Does JEPA latent state already correlate with deformation?

### 15.3 Metrics

- fitted stem displacement;
- leaf bend/stretch energy proxy;
- max contact force;
- grasp success;
- fit confidence;
- fit latency.

### 15.4 Interpretation

If D2 does not improve control, retain plant_twin as an interpretability and evaluation tool. Do not force it into the final method.

---

## 16. E9 — Embedded deployment benchmark

All measurements must be performed on the deployed Jetson AGX Orin configuration.

### 16.1 Encoder

Compare:
- V-JEPA 2.1 ViT-B/16 384;
- ViT-L/16 384 if feasible.

Measure:
- clip inference latency;
- effective target-state rate;
- unified-memory peak;
- GPU utilization;
- power;
- thermals.

### 16.2 Predictor

For each H:
- latency;
- p95 latency;
- memory;
- target-prediction error.

### 16.3 MPC

Measure:
- median solve time;
- p95;
- p99;
- missed control deadlines;
- prediction age at command issue.

### 16.4 End-to-end timing

Record timestamps for:
1. camera frame arrival;
2. encoder completion;
3. predictor completion;
4. MPC solution;
5. safety projection;
6. command publication.

Report actual perception-to-command latency.

---

## 17. Randomization and blocking

For physical trials:
- randomize method order within each scene;
- block by plant/scene;
- distribute methods across sessions/days;
- avoid always running the strongest method last;
- record plant condition changes across repeated manipulation.

If the plant visibly changes after a trial, either:
- reset/replace the plant; or
- treat order as a covariate/blocking factor.

---

## 18. Blinding

Full operator blinding is usually impossible because controller behavior is visible.

However:
- target annotations can be scored without method labels;
- success/failure adjudication should follow pre-written rules;
- statistical scripts should consume method IDs rather than manually curated result groups.

---

## 19. Statistical plan

### 19.1 Binary outcomes

For grasp success / target retained:

```math
\operatorname{logit}P(Y=1)
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
b_{plant}
+
b_{scene}.
```

Use mixed-effects logistic regression where supported by the final dataset.

### 19.2 Continuous outcomes

Examples:
- target error;
- completion time;
- force;
- clearance;
- travel distance.

Use linear mixed models if residual assumptions are acceptable.

Otherwise use:
- bootstrap confidence intervals;
- permutation tests;
- rank-based alternatives.

### 19.3 Pairwise comparisons

Pre-register a small number of planned comparisons:
- T4 vs strongest non-JEPA tracker;
- P3 vs P2;
- E vs B;
- full Piper-JEPA vs geometry-only MPC.

Use Holm correction for families of related pairwise tests.

### 19.4 Reporting

For every main comparison report:
- effect size;
- 95% CI;
- exact number of trials;
- number of unique plant/scene instances;
- failure counts.

Do not report p-values alone.

---

## 20. Minimum evidence required for each paper claim

| Claim | Minimum evidence |
|---|---|
| V-JEPA 2.1 improves persistence | E1 + E2 |
| action conditioning predicts future visibility | E3 |
| target weighting matters | P3 vs P2 in E3 |
| JEPA helps whole-body control beyond geometry | E5, especially R4 |
| full system improves grasping | E6 |
| safety is not degraded | E7 + force/clearance in E5/E6 |
| plant_twin adds value | E8 |
| deployable on Orin | E9 |

No claim should appear in the abstract unless its supporting experiment is complete.

---

## 21. Recommended result-table templates

### Table A — Target persistence

| Method | ID retention ↑ | Switch rate ↓ | Recovery ↑ | 2-D error ↓ | 3-D error ↓ |
|---|---:|---:|---:|---:|---:|
| T0 | | | | | |
| T1 | | | | | |
| T2 | | | | | |
| T3 | | | | | |
| T4 | | | | | |

### Table B — Prediction

| Method | H=1 px ↓ | H=4 px ↓ | H=8 px ↓ | ID@H8 ↑ | Visibility F1 ↑ |
|---|---:|---:|---:|---:|---:|
| P0 | | | | | |
| P1 | | | | | |
| P2 | | | | | |
| P3 | | | | | |

### Table C — Robot control

| Method | Target retained ↑ | First-grasp ↑ | Final success ↑ | Force ↓ | Clearance ↑ |
|---|---:|---:|---:|---:|---:|
| C0 | | | | | |
| C2 | | | | | |
| D | | | | | |
| C4 | | | | | |

---

## 22. Recommended figure set

Figure E1 — target memory sequence under occlusion.  
Figure E2 — ID retention versus occlusion.  
Figure E3 — future prediction examples for several candidate actions.  
Figure E4 — prediction error versus horizon and data budget.  
Figure E5 — visibility-sensitive whole-body scene with alternative trajectories.  
Figure E6 — grasp success / target loss by reachability category.  
Figure E7 — safety metrics.  
Extended — Orin latency, plant_twin, ViT-B vs ViT-L, failure taxonomy.

---

## 23. Pilot-to-final protocol

### Pilot stage

Use small experiments to validate:
- logging;
- synchronization;
- annotation;
- target IDs;
- safety;
- metric computation;
- model interfaces.

Pilot data may be used for debugging/training, but should not silently become final test data.

### Freeze point

Before final robot campaign:
1. freeze method definitions;
2. freeze test scenes or scene-generation protocol;
3. freeze exclusion rules;
4. freeze primary endpoints;
5. freeze statistics script structure.

Then collect final held-out trials.

---

## 24. Failure taxonomy

Record every failure using one primary category:

F1 — wrong initial language grounding.  
F2 — target tracking switch.  
F3 — target lost and not recovered.  
F4 — depth/geometry failure.  
F5 — planner infeasible.  
F6 — unsafe path rejected by safety layer.  
F7 — local servo failure.  
F8 — grasp mechanics failure despite correct approach.  
F9 — excessive plant deformation/contact.  
F10 — hardware/system fault.

This taxonomy should appear in the final paper or supplement.

---

## 25. Stop conditions

Stop a physical trial immediately on:
- emergency stop;
- force above the hardware safety threshold;
- uncontrolled base motion;
- invalid/stale state beyond watchdog limits;
- collision-risk state;
- operator-declared unsafe condition.

Safety stop is counted as a method outcome unless caused by unrelated hardware malfunction.

---

## 26. Immediate experiment implementation tasks

1. Define the episode metadata schema in the repository.
2. Add a rosbag2 recorder launch file.
3. Add stable physical target IDs to the perception pipeline.
4. Write the E1 annotation/evaluation script before collecting the full E1 dataset.
5. Record the first 20-30 pilot clips.
6. Validate synchronization and target-ID scoring.
7. Benchmark T0/T1/T4 offline.
8. Freeze E1 before scaling data collection.
9. Start E3 data collection only after action logging is verified.
10. Do not begin E5 final trials until C2 geometry-only whole-body MPC is stable.

---

## 27. Experimental success criteria for the project

The Piper-JEPA headline is supported only if all of the following hold:

1. V-JEPA 2.1 materially improves exact target persistence or recovery.
2. Piper-JEPA predicts target-specific future state better than generic/action-free alternatives.
3. Piper-JEPA improves R4 visibility-sensitive control relative to geometry-only MPC.
4. The gain does not come with worse force/clearance safety metrics.
5. The deployed system runs with acceptable prediction age and control latency on the Jetson AGX Orin.

If one item fails, narrow the paper claim rather than hiding the negative result.
