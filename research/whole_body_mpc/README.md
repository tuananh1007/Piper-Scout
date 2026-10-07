# Whole-Body MPC

**Scope:** deterministic coordinated control of the AgileX Scout 2.0 skid-steer base (modelled as differential drive) and Piper 6-DoF arm.

This track builds the strongest non-learned mobile-manipulation controller for Piper-Scout. It is both a potentially standalone research contribution and the deterministic baseline required for Piper-JEPA Stage C.

It can start before Semantic Scene is complete by using synthetic/simple obstacle fields, then later integrate the final semantic distance-query interface.

## Documents

- `WHOLE_BODY_MPC_RESEARCH_PLAN.md` — state/control formulation, objective, hypotheses, solver strategy, safety, and research positioning.
- `WHOLE_BODY_MPC_EXPERIMENTS.md` — dynamics validation, reachability, coordination, semantic-geometry ablations, handoff, and Orin protocol.
- `WHOLE_BODY_MPC_PAPER_OUTLINE.md` — manuscript structure and claim-to-evidence mapping.

## Core questions

1. Does unified base+arm control improve reachability over arm-only and sequential planning?
2. Does explicit non-holonomic (skid-steer-identified unicycle) modeling matter?
3. Can manipulability and base regularization produce sensible arm-first/base-when-needed behavior?
4. Does semantic class-aware geometry improve feasibility safely?
5. Can the controller run sustainably on Jetson AGX Orin?

## Development stages

### Stage A — Robot model + synthetic geometry

```text
Scout unicycle dynamics
          +
Piper kinematics/Jacobian
          +
synthetic/simple SDF
          ↓
whole-body rollout + costs
```

Implemented (P3A.1–P3A.3, offline synthetic); skid-steer slip identification still needs real-floor WE1 data.

### Stage B — Deterministic geometry-only MPC

```text
goal
+ collision
+ manipulability
+ base penalty
+ smoothness
      ↓
non-holonomic whole-body MPC
```

Main comparisons:
- arm-only;
- sequential Scout→Piper;
- holistic/reactive QP;
- unified MPC.

### Stage C — Semantic geometry integration

```text
Semantic Scene distance API
          +
whole-body MPC
          ↓
class-aware deterministic controller
```

This yields the final geometry-only baseline used by Piper-JEPA.

### Stage D — Piper-JEPA integration

Do not change the deterministic backbone unnecessarily.

Piper-JEPA Stage C should reuse:

```text
same dynamics
same solver
same goal cost
same collision cost
same manipulability
same base regularization
same safety layer
```

and add only predictive visual terms:

```math
J_{\rm PiperJEPA}
=
J_{\rm geo}
+
w_v J_{\rm visibility}
+
w_i J_{\rm identity}.
```

This makes the comparison scientifically clean.

## Relationship to Semantic Scene

Semantic Scene is not required to begin Stages A/B.

It becomes the preferred/final geometry source for Stage C.

Parallel development:

```text
Whole-Body MPC A/B ─────────────┐
                                │
Semantic Scene A/B/C ───────────┤
                                ↓
                    Whole-Body MPC C
                  semantic geometry baseline
```

## Relationship to Piper-JEPA

Whole-Body MPC is a hard dependency only for **Piper-JEPA Stage C**, not Piper-JEPA Stage A/B.

The critical final comparison is:

```text
geometry-only Whole-Body MPC
              vs.
same controller + Piper-JEPA
visibility/identity prediction
```

## Relationship to plant_twin

Optional.

A deformation term may later be added, but the deterministic controller must stand without `plant_twin`.

## Immediate starting point

1. Implement Scout unicycle rollout and identify skid-steer slip / track width (WE1).
2. Validate Piper FK/Jacobian.
3. Build synthetic SDF test scenes.
4. Implement goal, collision, smoothness, base, and manipulability costs.
5. Compare solver candidates offline.
6. Run R1–R3 reachability cases (scene classes defined in [`../README.md`](../README.md)).
7. Integrate Semantic Scene distance queries when stable.
8. Freeze the geometry-only baseline before adding Piper-JEPA predictive costs.

**Status:** implementation exists in [`Codes/src/scout_piper_whole_body_mpc/`](../../Codes/src/scout_piper_whole_body_mpc/) (P3A.1–P3A.5, 2026-10-06): unicycle rollout + `fit_slip`, Piper FK/Jacobian from the URDF, J_geo costs, MPPI with an arm-only W0 mode, safety filter, semantic-scene adapter and a dry-run ROS 2 node. Offline synthetic R1–R3 runs only; untested on hardware and not in `full_system.launch.py`. Since 2026-10-07 the MPPI converges near obstacles (P3A.6: elitism, knot-space gradient refinement, planner margin above the filter) and the W1 sequential baseline exists (P3A.9); on synthetic R3 the unified controller reaches the goal in about half W1's time with 0.2 m less base travel. Open: WE1 slip / TCP calibration on hardware (P3A.7), Orin timing (P3A.8), the W2 QP baseline (P3A.10).
