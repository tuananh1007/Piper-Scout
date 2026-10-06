# Piper + Scout Research & Development Roadmap

**Platform:** AgileX Piper 6-DoF arm + RealSense camera mounted on AgileX Scout 2.0 UGV  
**Compute:** NVIDIA Jetson AGX Orin 64 GB (on-robot) + operator laptop (GUI)  
**Application domain:** Autonomous plant manipulation — peduncle/branch grasping for pollination and selective harvesting  
**Last updated:** 2026-10-05

---

## 0. End-state vision

A non-expert operator stands next to the Scout and types or speaks:

> “Go to the second row of plants on your left, find the flower with the longest peduncle, and grasp it just below the bud.”

The system:

1. **Parses and grounds the instruction** into a structured query: semantic class, ordinal/spatial relation, optional attribute, and requested action.
2. **Navigates the Scout** to the work area with Nav2.
3. **Initializes the requested target** with open-vocabulary segmentation, then hands target identity to a V-JEPA 2.1 dense temporal representation so the same flower/peduncle persists through camera motion, robot motion, plant sway, and temporary occlusion.
4. **Maintains explicit metric geometry** with RealSense + semantic nvblox SDFs. V-JEPA does not replace depth or collision geometry.
5. **Maintains explicit deformation state where useful** using the existing `plant_twin` leaf/stem fitter during contact and pull interactions.
6. **Approaches the plant** with geometry-only whole-body MPC coordinating the Scout differential-drive base and Piper arm.
7. **Adds predictive whole-body planning** with a Piper-Scout action-conditioned JEPA model that forecasts whether candidate base+arm motions preserve target identity, visibility, and future manipulability.
8. **Hands off near contact** to a high-rate safety-bounded local MPPI / visual-servo controller with force, velocity, confidence, and semantic-clearance gates.
9. **Reports back** in plain language.

The architectural rule is:

> **JEPA predicts; geometry constrains; MPC decides; the safety layer executes.**

The headline research claim is therefore not “V-JEPA applied to agriculture,” but **task-conditioned dense prediction for visibility-aware whole-body manipulation of thin, deformable plant structures**.

---

## 1. Architecture overview

```text
                         OPERATOR LAPTOP
                  text / voice + confirmation
                            |
                            v
                  Language / VLM grounding
             q = (class, ordinal, relation,
                   attribute, action)
                            |
                            v
 RealSense RGB ---> Grounded-SAM / SAM-2 -----------+
       |               target initialization         |
       |                                             |
       v                                             v
 V-JEPA 2.1 dense video encoder               RealSense depth
 temporal target memory                       + camera geometry
       |                                             |
       |                                   semantic nvblox SDF
       |                                stem / branch / leaf / target
       |                                             |
       +--------------------+------------------------+
                            |
                            v
                  persistent task state
        target ID + 2-D distribution + 3-D estimate
              + visibility / uncertainty
                            |
                +-----------+------------+
                |                        |
                v                        v
      action-conditioned JEPA       plant_twin
       future dense rollouts    deformation/contact state
                |                        |
                +-----------+------------+
                            |
                    explicit robot
                 dynamics + kinematics
                            |
                            v
                whole-body predictive MPC
           controls [v_base, omega_base, qdot_1:6]
       reach + target visibility + identity + SDF + limits
                            |
                            v
                      safety projection
             hard clearance / velocity / force / watchdog
                            |
                    near-target switch
                            |
                            v
               bounded MPPI / visual servo
                            |
                            v
                          GRASP
```

### Separation of responsibility

| Layer | Responsible for | Must not be responsible for |
|---|---|---|
| Language / VLM | Interpret operator intent and initialize the requested target | Direct motor actuation |
| V-JEPA 2.1 | Dense temporal representation, target persistence, predictive visual state | Metric collision distance or final safety authority |
| RealSense + nvblox | Metric 3-D geometry and semantic clearance | Long-horizon semantic target identity |
| `plant_twin` | Explicit fitted leaf/stem deformation and contact-conditioned geometry | Replacing perception when fitting is invalid |
| Whole-body MPC | Optimize feasible Scout + Piper motion | Override hard safety constraints |
| Local MPPI / servo | Final centimeter-scale closed-loop approach | Open-ended semantic reasoning |
| Safety layer | Enforce hard clearance, velocity, force, freshness and abort rules | Optimize task reward |

---

## 2. Current baseline and gaps

| Component | Current repository state | Remaining gap |
|---|---|---|
| ROS 2 | Humble workspace and bringup scaffolded | Hardware regression and final wiring |
| Semantic perception | YOLO / Grounded-SAM paths | Robust open-vocabulary target initialization |
| Geometry | RealSense → nvblox path validated | Per-class SDF completion and planner query path |
| Target persistence | Framewise mask centroid + cached 3-D skeleton | Identity can jump under motion, sway, occlusion, or segmentation dropout |
| Servo | `FullAdaptiveServoController` | No explicit horizon or semantic constraints |
| Whole-body control | Planned | No coordinated non-holonomic base + arm MPC yet |
| Prediction | None | Planner cannot forecast whether motion preserves target visibility |
| Deformation | `plant_twin` exists | Not yet coupled to planning/control |
| Language interface | Planned | No persistent linkage between grounded language target and execution target |

---

## 3. Phased plan

The revised sequence separates **representation**, **local control**, **geometry-only whole-body control**, and **learned predictive whole-body control** so every learned claim has a deterministic comparator.

### Phase 0 — ROS 2 Humble migration + Scout integration (Months 1–2)

**Goal:** complete and validate the ROS 2 hardware baseline.

**Current status:** ROS 2 workspace, Piper+Scout description, segmentation path, masked point cloud, skeleton/candidate logic, bringup scaffolding, and live RealSense→nvblox smoke path are present.

**Remaining work**
- moveit_servo wiring;
- Nav2 hardware validation;
- Piper + Scout + RealSense synchronized hardware bringup;
- ROS 1 → ROS 2 regression;
- E-stop / stop-and-zero validation.

**Exit criteria**
- Arm + base + camera + TF operate together on hardware.
- Servo commands reach the arm through moveit_servo.
- Scout executes a basic Nav2 goal.
- Candidate poses reproduce the legacy stack within documented tolerance.
- Safety stop path is validated before any learned controller is allowed to command motion.

**Publication role:** enabling/platform work.

---

### Phase 1 — Semantic RGB-D scene representation + deformable plant state (Months 3–4)

**Goal:** establish explicit metric geometry and deformation baselines before learned prediction.

#### 1A — Semantic nvblox

- RealSense depth → nvblox TSDF/ESDF.
- Semantic classes: stem, branch, leaf, target.
- Non-target stem/branch: hard collision policy.
- Leaf: soft cost with configurable clearance/contact budget.
- Selected target: grasp attractor and removable from the non-target hard field only in final grasp mode.
- Complete semantic-mask integration.
- Finish collision-plugin ESDF query path.
- Measure update latency, GPU memory and sustained rate on the actual AGX Orin.

#### 1B — `plant_twin` integration

The repository now contains a deformable leaf + stem digital twin with:
- leaf mesh + bending state;
- stem centerline + length/smoothness constraints;
- temporal priors;
- gripper-contact residuals;
- warm-started LM / Gauss-Newton fitting.

Use it for:
1. deformation-state analysis during contact;
2. optional local MPC deformation costs;
3. simulation-style perturbation experiments;
4. testing whether JEPA latent changes correlate with explicit deformation state.

`plant_twin` is **not** a hard dependency for tracking or safety. Invalid/stale fits are ignored.

**Exit criteria**
- Semantic clearance queries available to controllers.
- Thin structures measurably better represented than the legacy Octomap baseline.
- `plant_twin` synchronized with the manipulation pipeline and exposing fit confidence/freshness.
- Orin timing measured rather than assumed.

---

### Phase 2A — V-JEPA 2.1 dense temporal target state (Months 5–6) ★ NEW

**Goal:** replace framewise target-persistence heuristics with a temporally consistent representation of the exact flower/peduncle selected by the operator.

**Research question**

Can dense V-JEPA 2.1 features preserve small-instance identity through eye-in-hand motion, Scout motion, plant sway, temporary occlusion and segmentation dropout better than practical trackers and V-JEPA 2?

**Initial deployment**
- V-JEPA 2.1 ViT-B/16, 384 px (80M) first.
- Frozen encoder for the first benchmark.
- ViT-L/16 only after the ViT-B end-to-end path works.
- JEPA consumes RGB; RealSense remains the metric geometry source.

Given a video window:

```math
F_t = E_\theta(I_{t-L+1:t})
```

where $F_t$ is a dense token grid.

Grounding initializes target mask $`M_t^*`$. Define target descriptor

```math
r_t =
\frac{\sum_p M_t^*(p)F_t(p)}
{\sum_p M_t^*(p)+\epsilon}.
```

Dense similarity:

```math
C_{t+1}(p)=\cos(F_{t+1}(p),r_t)
```

and spatial target distribution:

```math
P_{t+1}(p)=\mathrm{softmax}(C_{t+1}(p)/\tau).
```

Target image position:

```math
\hat u_{t+1}=\sum_p P_{t+1}(p)p
```

and uncertainty:

```math
H_{t+1}=-\sum_p P_{t+1}(p)\log P_{t+1}(p).
```

Back-project valid depth to obtain a 3-D target estimate and uncertainty.

**Implementation rule**

Keep dense tensors inside one GPU process. Publish compact ROS state:
- target image mean/covariance;
- target 3-D mean/covariance;
- target-ID confidence;
- entropy / visibility;
- occluded/lost flag;
- timestamp + maximum-valid-age.

**Suggested package**

```text
Codes/src/scout_piper_jepa/
  scout_piper_jepa/
    encoder.py
    target_memory.py
    target_state_node.py
    rosbag_dataset.py
    eval_tracking.py
  config/vjepa2_1.yaml
```

**Suggested topics**

```text
/piper_jepa/target_state
/piper_jepa/target_uncertainty
/piper_jepa/target_visible
/piper_jepa/debug_similarity
```

**Dataset v0**

Record synchronized rosbag2 episodes containing:
- RGB, depth, CameraInfo;
- TF;
- Piper joint state;
- Scout odometry;
- target masks and selected target identity;
- executed base/arm commands;
- force/contact;
- `plant_twin` state when valid;
- final outcome.

Factors:
- static target;
- natural / induced sway;
- partial occlusion;
- temporary full occlusion;
- arm-only camera motion;
- base-only motion;
- combined base+arm motion;
- lighting variation;
- thin-structure depth dropout.

**Baselines**
1. Framewise Grounded-SAM / SAM.
2. Conventional optical / feature tracking.
3. Strong dense self-supervised visual-feature tracking.
4. V-JEPA 2.
5. V-JEPA 2.1.

**Metrics**
- target-ID retention;
- false target-switch rate;
- 2-D center error;
- 3-D target error;
- mask IoU where visible;
- occlusion recovery;
- temporal jitter;
- latency / FPS / memory / power on Orin.

**Go/no-go gate**

Do not advance the JEPA paper claim solely because V-JEPA 2.1 is newer. Continue only if it gives a meaningful target-persistence or occlusion-recovery gain over the strongest practical baseline.

---

### Phase 2B — Safety-bounded local MPPI visual servo (Months 6–8)

**Goal:** replace the legacy adaptive IBVS with a constrained final-approach controller while keeping the current controller as a fallback baseline.

**State**
- Piper joint state + EE pose;
- target distribution/uncertainty from Phase 2A;
- RealSense depth;
- semantic SDF clearance;
- force/contact;
- optional `plant_twin` deformation state.

**Control**
- Piper arm joint velocity;
- Scout frozen by default during near-contact mode.

**Costs**
- image target error;
- target uncertainty / visibility;
- joint limits;
- manipulability;
- semantic clearance;
- smoothness;
- force/contact penalty;
- optional deformation penalty.

**Hard gates**
- stop/re-ground after persistent target-confidence loss;
- stop/retract on force-limit violation;
- stop on hard semantic-clearance violation;
- enforce velocity/acceleration bounds independently of JEPA.

**Ablation**
- current `FullAdaptiveServoController`;
- constrained conventional servo;
- MPPI with framewise target state;
- MPPI with V-JEPA 2.1 target state;
- with/without `plant_twin` deformation cost.

---

### Phase 3A — Geometry-only whole-body GPU MPC (Months 8–11)

**Goal:** build the strongest deterministic mobile-manipulation baseline before learned prediction.

Scout configuration:

```math
x_b=[x,y,\theta]
```

with differential-drive controls

```math
u_b=[v,\omega].
```

Combined configuration:

```math
x=[x_b,y_b,\theta_b,q_1,\ldots,q_6]^T
```

and control

```math
u=[v,\omega,\dot q_1,\ldots,\dot q_6]^T\in\mathbb R^8.
```

Base dynamics:

```math
x_{t+1}=x_t+\Delta t\,v_t\cos\theta_t
```

```math
y_{t+1}=y_t+\Delta t\,v_t\sin\theta_t
```

```math
\theta_{t+1}=\theta_t+\Delta t\,\omega_t.
```

Arm:

```math
q_{t+1}=q_t+\Delta t\,\dot q_t.
```

Do **not** model the Scout as independently actuated Cartesian $x/y$ joints in the final formulation.

Geometry-only objective:

```math
J_{\rm geo}
=
w_gJ_{\rm goal}
+w_cJ_{\rm collision}
+w_lJ_{\rm leaf}
+w_mJ_{\rm manip}
+w_bJ_{\rm base}
+w_sJ_{\rm smooth}
+w_dJ_{\rm deform}.
```

$J_{\rm deform}$ is optional and comes from valid `plant_twin` state.

**Baselines**
- arm-only;
- sequential Scout → Piper;
- holistic/reactive QP;
- geometry-only whole-body MPC.

**Scene categories**
1. comfortably arm reachable;
2. near workspace boundary;
3. unreachable without base motion;
4. geometrically reachable but visibility-sensitive.

Do not commit to a 200 Hz claim until the controller is measured on Orin.

---

### Phase 3B — Piper-JEPA predictive whole-body MPC (Months 10–14) ★ HEADLINE

**Goal:** predict whether candidate whole-body motions preserve the identity and visibility of the selected thin plant target, and use that prediction as a soft planning signal inside explicit geometry-constrained MPC.

**Novelty boundary**

V-JEPA 2-AC already performs action-conditioned latent prediction for robot planning. V-JEPA 2.1 provides stronger dense representations. Therefore the contribution is **not** simply “use V-JEPA 2.1 with actions.”

The intended claim is:

> **A task-conditioned dense predictive world model improves whole-body manipulation of small deformable plant structures by forecasting target identity, visibility and uncertainty under coordinated mobile-base + arm motion, while explicit RGB-D semantic geometry and a separate safety layer remain authoritative for collision and contact constraints.**

#### Action representation

Optimizer action:

```math
u_t=[v_b,\omega_b,\dot q_{1:6}].
```

Map to an embodiment-normalized JEPA action:

```math
a_t=\Gamma(x_t,u_t)
```

with

```math
a_t=[
\Delta s_b,
\Delta\theta_b,
\Delta p_{ee}^{(3)},
\Delta r_{ee}^{(3)},
\Delta g].
```

#### Action-conditioned predictor

```math
\hat Z_{t+1}
=
P_\phi(Z_{t-K+1:t},a_t,s_t).
```

Rollout:

```math
\hat Z_{t+1:t+H}
=
P_\phi(Z_t,a_{t:t+H-1},s_t).
```

#### Target-weighted prediction loss

Thin peduncles occupy few patches, so global latent loss can ignore the task-relevant region.

```math
w_t(p)=
1+\lambda_T M_{\rm target}(p)
+\lambda_P M_{\rm plant}(p).
```

Teacher-forced loss:

```math
\mathcal L_{\rm TF}
=
\sum_p
w_t(p)
\|
\hat Z_{t+1}(p)-\mathrm{sg}[Z_{t+1}(p)]
\|_1.
```

Rollout loss:

```math
\mathcal L_{\rm roll}
=
\sum_{k=1}^H
\gamma^{k-1}
\sum_p
w_{t+k}(p)
\|
\hat Z_{t+k}(p)-\mathrm{sg}[Z_{t+k}(p)]
\|_1.
```

Total:

```math
\mathcal L_{\rm AC}
=
\mathcal L_{\rm TF}
+\lambda_R\mathcal L_{\rm roll}.
```

#### Predict future target state

```math
\hat C_{t+k}(p)
=
\cos(\hat Z_{t+k}(p),r_t)
```

```math
\hat P_{t+k}(p)
=
\mathrm{softmax}(\hat C_{t+k}(p)/\tau).
```

Then:

```math
\hat u_{t+k}
=
\sum_p \hat P_{t+k}(p)p
```

and

```math
\hat H_{t+k}
=
-\sum_p
\hat P_{t+k}(p)\log\hat P_{t+k}(p).
```

#### JEPA-aware whole-body objective

```math
J=
w_gJ_{\rm goal}
+w_vJ_{\rm visibility}
+w_iJ_{\rm identity}
+w_cJ_{\rm collision}
+w_lJ_{\rm leaf}
+w_mJ_{\rm manip}
+w_bJ_{\rm base}
+w_sJ_{\rm smooth}
+w_dJ_{\rm deform}.
```

Predictive visibility:

```math
J_{\rm visibility}
=
\sum_k
\left[
\|\hat u_{t+k}-u_{\rm des}\|^2
+\lambda_H\hat H_{t+k}
+B_{\rm FoV}(\hat u_{t+k})
\right].
```

Identity preservation:

```math
J_{\rm identity}
=
\sum_k
\left[
1-\cos(\hat r_{t+k},r_t)
\right].
```

Optional explicit deformation from `plant_twin`:

```math
J_{\rm deform}
=
\sum_k
[
\alpha_sE_{\rm stem\ strain}
+\alpha_lE_{\rm leaf\ stretch}
+\alpha_bE_{\rm leaf\ bend}
].
```

Roles remain distinct:
- JEPA predicts future visual target state.
- `plant_twin` provides explicit fitted deformation quantities.

#### Hard safety

For robot collision primitives $c_j$:

```math
\phi_{\rm hard}(c_j(x_{t+k}))\ge d_{\rm safe}.
```

The learned predictor never relaxes this condition.

Safety projection:

```math
u_{\rm safe}
=
\arg\min_u
\|u-u_{\rm MPC}\|_2^2
```

subject to:
- semantic hard clearance;
- joint/base velocity and acceleration limits;
- force limit;
- state freshness;
- watchdog / abort conditions.

#### Near-contact handoff

Switch to Phase 2B when:
- EE-target distance < $d_{\rm switch}$;
- target entropy < $H_{\max}$;
- confidence > $c_{\min}$;
- semantic clearance is valid.

Freeze or strongly penalize Scout motion after handoff.

#### Training-data study

Evaluate increasing Piper-Scout data budgets:

```math
1\,{\rm h}, 2\,{\rm h}, 5\,{\rm h}, 10\,{\rm h}
```

or the closest feasible balanced subsets. Report data efficiency.

#### Core ablations

1. V-JEPA 2 vs V-JEPA 2.1.
2. pooled/global vs dense state.
3. no target weighting vs target weighting.
4. no action conditioning vs action conditioning.
5. arm-only actions vs whole-body actions.
6. rollout $H=1,2,4,8$.
7. learned RGB state alone vs learned state + RGB-D geometry.
8. geometry-only MPC vs JEPA-aware MPC.
9. without vs with `plant_twin` deformation cost.
10. without vs with local safety-servo handoff.
11. ViT-B vs ViT-L accuracy/latency trade-off.

#### Primary robot metrics

- first-attempt grasp success;
- final grasp success;
- target-ID retention;
- target-loss events;
- completion time;
- Scout distance;
- EE path length;
- minimum non-target clearance;
- maximum contact force;
- safety-filter activations;
- predictor/MPC latency;
- GPU memory.

#### Critical experiment

Create scenes where several trajectories are geometrically feasible but some cause self-occlusion or foliage occlusion of the requested target. This is the experiment that must demonstrate value beyond Phase 3A geometry-only MPC.

**Success criterion:** improve task outcome or target retention without worsening safety metrics, and beat the geometry-only whole-body controller.

**Working title:** *Piper-JEPA: Task-Conditioned Dense World Models for Safe Whole-Body Plant Manipulation*

---

### Phase 4 — Uncertainty-driven active perception (Months 13–15)

**Goal:** acquire new viewpoints only when the persistent target state or metric geometry is uncertain.

Combine:
- JEPA target uncertainty;
- RGB-D / VGGT geometry uncertainty;
- `plant_twin` fit uncertainty where deformation matters.

```math
U_{\rm total}
=
\alpha U_{\rm JEPA}
+\beta U_{\rm geometry}
+\gamma U_{\rm twin}.
```

**Ablations**
- fixed ring;
- random reachable viewpoints;
- geometry-only NBV;
- JEPA-only uncertainty;
- fused task-aware NBV.

**Exit criterion:** fewer views or lower acquisition time at equal or better grasp performance.

---

### Phase 5 — Language/VLM + operator GUI for non-experts (Months 15–18) ★ APPLICATION CAPSTONE

**Goal:** expose the validated stack through natural-language interaction without making the VLM the motor controller.

Map instruction $\ell$ to:

```math
q_\ell=
({\rm class,ordinal,spatial\ relation,attribute,action}).
```

Example:

> “grasp the third flower from the left”

becomes:

```text
class = flower
ordinal = 3
relation = left-to-right
action = grasp
```

The VLM initializes the target; Phase 2A maintains its identity afterward.

**VLM responsibilities**
- parse instruction;
- ground candidates;
- ask for clarification when ambiguous;
- generate readable state/rejection explanations.

**VLM must not**
- bypass semantic SDF constraints;
- bypass reachability checks;
- bypass safety projection;
- send raw motor commands.

**GUI**
- RGB/RGB-D live view;
- selected target overlay + confidence;
- target-loss/occlusion state;
- grasp pose + planned path;
- force/clearance/safety state;
- `plant_twin` deformation overlay when valid;
- confirm/abort;
- text + local speech input.

---

## 4. Timeline at a glance

```text
Month   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18
P0     ██████
P1           ██████
P2A                ██████
P2B                   ████████
P3A                         ████████████
P3B                               ██████████████
P4                                           ████████
P5                                                 ████████████
```

| Phase | Relative months | Lead milestone | Publication role |
|---|---|---|---|
| 0 — ROS 2 + Scout | 1–2 | Hardware-integrated baseline | Platform / tech report |
| 1 — Semantic + deformable geometry | 3–4 | Valid clearance + `plant_twin` state | Workshop / baseline |
| 2A — V-JEPA 2.1 target state | 5–6 | Target persistence under motion/occlusion | Representation benchmark |
| 2B — Bounded local MPPI servo | 6–8 | Safer final approach | IROS / RA-L if independently strong |
| 3A — Geometry whole-body MPC | 8–11 | Non-holonomic base+arm control | Deterministic baseline / possible full paper |
| 3B — Piper-JEPA predictive MPC | 10–14 | Predict future target visibility/identity | **Headline ICRA / IROS / RSS / RA-L** |
| 4 — Active perception | 13–15 | Uncertainty-driven NBV | Follow-on if clear gain |
| 5 — Language + GUI | 15–18 | Non-expert language-grounded grasping | HRI / RO-MAN |

The objective is **not** to force one paper per phase. Phases 2A–3B should converge into one coherent Piper-JEPA paper if the ablations support the combined claim.

---

## 5. Compute and real-time budget on Jetson AGX Orin 64 GB

Do not treat projected desktop-GPU throughput as an Orin result.

| Component | Initial choice | Required measurement |
|---|---|---|
| RealSense + nvblox | Existing Phase 1 path | latency, memory, sustained rate |
| Segmentation / grounding | current YOLO/Grounded-SAM | invocation latency, duty cycle, memory |
| V-JEPA 2.1 encoder | ViT-B/16 384 first; ViT-L second | clip latency, FPS, memory, power |
| Target-memory matching | same process as encoder | incremental latency, jitter |
| Action predictor | compact predictor first | rollout latency vs horizon, accuracy |
| `plant_twin` | current CPU fitter first | fit rate, confidence, CPU/GPU cost |
| Geometry whole-body MPC | GPU-batched | solve-time distribution, missed deadlines |
| JEPA-aware MPC | asynchronous predictor + faster controller | prediction age at command time |
| Local MPPI / servo | highest-rate bounded loop | sustained rate, jitter, stop latency |
| Language / VLM | laptop default if needed | instruction latency, network dependency |

### Scheduling principle

- language grounding: event driven;
- segmentation refresh: event/uncertainty driven;
- V-JEPA target state: moderate rate from profiling;
- action-conditioned rollout: asynchronous if needed;
- whole-body control: faster loop using freshest valid prediction;
- `plant_twin`: only when deformation/contact state is useful;
- local safety/servo: highest-rate loop.

All learned/fitted states carry timestamps and maximum-valid-age watchdogs.

### Model scaling

1. Complete the system with V-JEPA 2.1 ViT-B.
2. Measure whether ViT-L improves target persistence/prediction enough to justify cost.
3. Do not target ViT-g/ViT-G for embedded deployment without measured benefit.

---

## 6. Cross-phase risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| V-JEPA 2.1 does not beat simpler tracking | Medium | Phase 2A go/no-go; retain strongest tracker |
| Dense features preserve class but not exact instance | Medium | target initialization + explicit target-switch metric |
| Predictor learns ego-motion but not plant deformation | Medium-high | collect sway/contact episodes; stratify with `plant_twin` state |
| `plant_twin` fit invalid under severe occlusion | Medium | confidence/freshness gating |
| Action predictor too slow on Orin | High | asynchronous rollout, shorter horizon, distillation |
| Learned prediction conflicts with safe geometry | Low by design | JEPA is soft cost only |
| Thin peduncle depth failure | High | temporal fusion + active perception |
| Scout localization drift | Medium | odom + IMU + local RGB-D alignment |
| Non-holonomic base awkward in optimizer | Medium | explicit differential-drive dynamics |
| GPU contention | High | asynchronous scheduling, ViT-B first |
| Target silently changes identity | Medium | persistent descriptor + abort/re-ground |
| Scope expands into too many papers | High | Phase 3B remains headline |

---

## 7. Open-source strategy

| Artifact | Phase | Proposed license | Notes |
|---|---|---|---|
| Piper-on-Scout URDF + bringup | 0 | Apache 2.0 | Platform baseline |
| Semantic SDF collision/query layer | 1 | Apache 2.0 | Useful beyond agriculture |
| `plant_twin` | 1 | repository license | Add confidence/planner interface |
| Rosbag benchmark schema | 1–3 | dataset-specific | annotations/splits where permitted |
| `scout_piper_jepa` target-memory package | 2A | Apache 2.0 | wrappers/evaluation code |
| Bounded MPPI servo | 2B | BSD / Apache 2.0 | reusable |
| Non-holonomic whole-body controller | 3A | Apache 2.0 | Scout/Piper reference |
| Piper-JEPA predictor + MPC integration | 3B | Apache 2.0 | main research release |
| Task-aware NBV | 4 | Apache 2.0 | optional |
| Language bridge + GUI | 5 | MIT / Apache 2.0 | no direct motor bypass |

Suggested layout:

```text
Codes/src/
  scout_piper_bringup/
  scout_piper_description/
  scout_piper_scene_repr/
  plant_twin/
  stem_grasp/
  scout_piper_jepa/
    scout_piper_jepa/
      encoder.py
      target_memory.py
      predictor.py
      target_state_node.py
      predictive_cost.py
    config/
    launch/
    test/
  scout_piper_whole_body_mpc/
    dynamics/
    costs/
    safety/
    launch/
```

---

## 8. Core experiment / ablation plan

| ID | Method | Dense 2.1 | Action predictor | Target weighting | Semantic SDF | plant_twin | Whole-body | Safety servo |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | Legacy/current pipeline | – | – | – | – | – | – | yes |
| B | Geometry whole-body MPC | – | – | – | yes | optional | yes | yes |
| C | V-JEPA 2 action-conditioned | no | yes | no | yes | no | yes | yes |
| D | V-JEPA 2.1 dense + generic predictor | yes | yes | no | yes | no | yes | yes |
| E | Piper-JEPA, no final handoff | yes | yes | yes | yes | optional | yes | no |
| F | **Piper-JEPA full** | **yes** | **yes** | **yes** | **yes** | **optional** | **yes** | **yes** |

Key comparisons:
- C → D: benefit of V-JEPA 2.1 dense representation.
- D → E: contribution of target-conditioned / target-weighted prediction.
- B → F: value beyond geometry-only whole-body MPC.
- E → F: value of bounded local-control handoff.
- F without vs with `plant_twin`: contribution of explicit deformation state.

### Full-system scene factors

- foliage density;
- target thickness;
- number of similar flowers;
- occlusion severity;
- induced plant motion;
- start pose;
- reachability class;
- illumination/background.

### Primary metrics

- first-attempt grasp success;
- final grasp success;
- target-ID retention;
- false target switches;
- target-loss events;
- 2-D / 3-D tracking error;
- occlusion recovery;
- completion time;
- base distance;
- EE path length;
- minimum non-target clearance;
- maximum contact force;
- safety-filter activations;
- encoder/predictor/MPC latency;
- memory and power.

---

## 9. Key references

### V-JEPA / predictive world models
- V-JEPA 2: *Self-Supervised Video Models Enable Understanding, Prediction and Planning* — arXiv:2506.09985
- V-JEPA 2.1: *Unlocking Dense Features in Video Self-Supervised Learning* — arXiv:2603.14482
- Official code: `facebookresearch/vjepa2`

### Motion planning / control
- cuRobo
- NVIDIA Isaac ROS cuMotion
- MPPI-VS
- Haviland & Corke holistic mobile manipulation
- NEO
- EHC-MM
- RMMI

### Scene representation / active perception
- isaac_ros_nvblox
- VGGT
- Next Best Sense
- ActiveSplat

### Agricultural manipulation
- autonomous selective harvesting literature
- peduncle collision-free grasping
- robotic pollination