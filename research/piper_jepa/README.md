# Piper-JEPA

**Scope:** target-conditioned dense video state, action-conditioned target prediction, and later predictive whole-body manipulation.

Piper-JEPA is one research component of the broader Piper-Scout system. It does **not** require the entire Piper-Scout stack to be complete before work begins.

The research is intentionally divided into three stages so the representation and prediction questions can be tested early, while geometry and whole-body control are developed in parallel.

## Documents

- `PIPER_JEPA_RESEARCH_PLAN.md` — scientific hypotheses, method, theory, and research positioning.
- `PIPER_JEPA_EXPERIMENTS.md` — dataset, trial, ablation, metric, and statistical protocol.
- `PIPER_JEPA_PAPER_OUTLINE.md` — manuscript structure and claim-to-evidence mapping.

## Piper-JEPA stages

### Stage A — Dense Target Memory

**Goal:** determine whether V-JEPA 2.1 can preserve the exact selected physical target through camera motion, plant sway, temporary occlusion, and segmentation dropout.

Minimum inputs:
- RGB/video stream;
- an initial target mask or physical target ID.

Core pipeline:

```text
RGB video
   +
initial target mask / target ID
   ↓
V-JEPA 2.1 dense features
   ↓
target-conditioned dense memory
   ↓
persistent target identity
+ image-space target distribution
+ confidence / uncertainty
```

Stage A does **not** require:
- semantic nvblox;
- whole-body MPC;
- plant_twin;
- active perception;
- language/VLM;
- the final GUI.

This stage supports the E1/E2 experiments in `PIPER_JEPA_EXPERIMENTS.md`.

---

### Stage B — Action-Conditioned Prediction

**Goal:** predict how the selected target's visibility, location, and identity will change after candidate robot actions.

Additional inputs:
- Piper state/action;
- Scout state/action for whole-body prediction.

Core pipeline:

```text
persistent target state
        +
robot state / action
        ↓
action-conditioned Piper-JEPA predictor
        ↓
future target distribution
future target visibility
future target identity confidence
future uncertainty
```

Stage B still does **not** require semantic geometry or a finished whole-body controller. Robot data can be collected using:
- scripted arm motion;
- manual arm jogging;
- Scout teleoperation;
- scripted base motion;
- simple combined base + arm trajectories.

This stage supports the E3 experiment.

---

### Stage C — Predictive Whole-Body Control

**Goal:** use Piper-JEPA predictions to improve trajectory selection for the coordinated Scout + Piper system.

This is the stage where Piper-JEPA integrates with the other research tracks.

Required integration:

```text
Piper-JEPA future target prediction
              +
semantic_scene metric geometry
              +
whole_body_mpc dynamics/control
              ↓
visibility- and identity-aware
whole-body predictive control
```

The geometry-only whole-body controller is the critical deterministic baseline. Piper-JEPA should augment it with predictive visual costs rather than replace it.

Conceptually:

```math
J_{\rm PiperJEPA}
=
J_{\rm geo}
+
w_v J_{\rm visibility}
+
w_i J_{\rm identity}.
```

This stage supports E4-E7 and the main control claim of the paper.

---

## Dependency map

| Research component | Stage A | Stage B | Stage C | Role |
|---|---:|---:|---:|---|
| RGB/video | Required | Required | Required | learned visual state |
| Initial target mask / ID | Required | Required | Required | binds the physical target |
| Piper state/actions | Optional | Required | Required | action-conditioned prediction/control |
| Scout state/actions | No | Required for whole-body predictor | Required | non-holonomic mobile manipulation |
| `semantic_scene` | No | No | **Required** | metric geometry and collision clearance |
| `whole_body_mpc` | No | No | **Required** | deterministic controller and baseline |
| `plant_twin` | No | No | Optional | deformation state / interpretation / optional cost |
| `active_perception` | No | No | Optional later extension | new viewpoints driven by uncertainty |
| language/VLM | No | No | Optional application layer | target initialization from human instruction |
| GUI | No | No | No | final system interaction layer |

## Relationship to other research tracks

### `../semantic_scene/`

Not a prerequisite for Stage A or Stage B.

It becomes a **hard integration dependency for Stage C**, because the final predictive controller must use explicit metric geometry for collision and clearance constraints.

Piper-JEPA must not replace semantic geometry.

### `../whole_body_mpc/`

Not a prerequisite for Stage A or Stage B.

It becomes a **hard integration dependency for Stage C** and should provide the strongest geometry-only comparator:

```text
geometry-only whole-body MPC
             vs.
same MPC + Piper-JEPA visibility/identity prediction
```

This comparison is necessary to show that any improvement comes from predictive target state rather than simply from adding mobile-base control.

### `../plant_twin/`

`plant_twin` is **not a Piper-JEPA dependency**.

It is an optional complementary signal that can provide:
- explicit leaf/stem deformation state;
- deformation-aware MPC costs;
- analysis of whether JEPA latent changes correlate with physical bending/deformation;
- additional post-contact evaluation.

Piper-JEPA should remain scientifically valid without `plant_twin`.

### `../active_perception/`

Active perception is a **later extension**, not a prerequisite.

In fact, Piper-JEPA uncertainty can help drive active perception:

```text
Piper-JEPA target uncertainty
            ↓
task-aware next-best-view
            ↓
new observation
            ↓
updated Piper-JEPA target state
```

The two tracks are therefore complementary rather than strictly upstream/downstream.

## Recommended development strategy

Run two workstreams in parallel:

```text
WORKSTREAM A — Piper-JEPA

Stage A: Dense Target Memory
          ↓
Stage B: Action-Conditioned Prediction
          ↓
          ──────────────────────┐
                                │
WORKSTREAM B — Robot Control    │
                                │
semantic_scene                  │
      ↓                         │
geometry-only whole_body_mpc    │
      └─────────────────────────┤
                                ↓
                    Stage C: Piper-JEPA
                    Predictive Whole-Body Control
```

This lets the project answer the most important early question before investing in the full control stack:

> **Can V-JEPA 2.1 actually preserve and predict the exact selected plant target better than strong simpler baselines?**

If Stage A or Stage B fails, the project can change direction early. If they succeed, the results feed directly into Stage C.

## Immediate starting point

The recommended first implementation task is **Stage A**, not full-system integration:

1. record or reuse Piper/RealSense RGB sequences;
2. assign a physical target ID and initial mask;
3. run V-JEPA 2.1 ViT-B dense inference;
4. build the target descriptor / similarity distribution;
5. measure exact-instance retention under arm motion and temporary occlusion;
6. compare against framewise segmentation and a strong conventional tracker.

Only after this evidence is positive should embodiment-specific action-conditioned predictor training become the main effort.

**Status:** Stage A code exists in [`Codes/src/scout_piper_jepa/`](../../Codes/src/scout_piper_jepa/) (P2A.1–P2A.4, 2026-10-06): encoder interface (lazy V-JEPA via `torch.hub` + a numpy reference encoder), target memory, E1 metrics, episode export/evaluation and a ROS 2 node. It is tested only on synthetic scenes with the reference encoder, is untested on hardware and is not in `full_system.launch.py`. V-JEPA inference has not been run, the E1 dataset and the T0–T4 comparison are open (P2A.5–P2A.6), and Stage B has only the action embedding (`action.py`).

Since 2026-10-07 Stage B and the Stage C cost exist on a synthetic dense-feature world (P3B.1–P3B.7): the learned predictor (P0/P2/P3, target-weighted loss), §12 read-outs, E3 metrics and benchmark, and `JepaVisibilityCost` in the whole-body MPC. The plain §12 read-out cannot tell the selected flower from an identical twin, so the cost anchors its read-out on the projected metric target position when depth gives one. On the synthetic E3 benchmark (one seed) the target-weighted P3 is the only predictor that beats persistence on every target metric; at 0.8 s it cuts target location error from 18.0 px (persistence) and 21.9 px (P2) to 13.0 px. The synthetic closed-loop comparison (P3B.7, 2026-10-10): with the visibility cost as a goal-constrained secondary term, C3-oracle keeps the goal accuracy of C2 and, from starts where geometry alone loses the flower, keeps it in view 2.5–3.6 times as long during the approach; it does not change what is seen at the end pose, so identity at the grasp needs the end pose chosen for the view. Robot episodes, V-JEPA features and GPU timing are open. Since 2026-10-10 the learned predictor also takes the joint angles as input, as §10 defines it (no clear difference on a reduced synthetic run), and can work on V-JEPA features projected onto their main directions (`--proj-dim`): raw ViT-L features would make 2.7 GB of predicted features per MPPI iteration at 256 samples.

---

**Summary:** Piper-JEPA can start independently. Only its final Stage C predictive-control contribution depends directly on `semantic_scene` and `whole_body_mpc`; `plant_twin` and `active_perception` remain optional/later complementary tracks.
