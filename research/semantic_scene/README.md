# Semantic Scene Representation

**Scope:** deterministic metric RGB-D geometry for thin-structure plant manipulation.

This track provides the explicit geometric representation and collision/clearance authority used by Whole-Body MPC and later Piper-JEPA Stage C.

It can be developed independently of Piper-JEPA and in parallel with Whole-Body MPC.

## Documents

- `SEMANTIC_SCENE_RESEARCH_PLAN.md` — hypotheses, geometry formulation, class policies, interfaces, and research positioning.
- `SEMANTIC_SCENE_EXPERIMENTS.md` — reconstruction, semantic integrity, distance-query, planning, temporal-robustness, and Orin protocols.
- `SEMANTIC_SCENE_PAPER_OUTLINE.md` — manuscript structure and evidence mapping.

## Core questions

1. Can thin stems/branches be preserved reliably enough for manipulation?
2. Can semantic classes support different hard/soft collision policies?
3. Can signed-distance queries be accurate and fresh enough for real-time control?
4. Does class-aware semantic geometry improve planning over binary occupancy?

## Development stages

### Stage A — Geometry baseline

```text
RealSense RGB-D
      ↓
nvblox TSDF / ESDF
      ↓
non-semantic metric geometry
```

The RealSense→nvblox smoke path runs in the dev container (P1.1.2); the Orin benchmark is open (P1.1.3).

### Stage B — Semantic class fields

```text
RGB-D
  +
semantic masks
  ↓
class-specific metric fields
stem / branch / leaf / target
```

Primary output:
- hard stem/branch geometry;
- soft leaf geometry;
- selected-target geometry.

### Stage C — Planner-facing semantic distance API

```text
semantic metric fields
        ↓
batched signed-distance queries
+ class
+ validity
+ freshness
        ↓
Whole-Body MPC
```

This is the integration point with `../whole_body_mpc/`.

## Relationship to Whole-Body MPC

Whole-Body MPC can begin with synthetic or simple SDFs.

The final class-aware deterministic controller requires Semantic Scene for:
- hard collision distance;
- leaf soft cost;
- target-aware geometry;
- freshness/validity.

Therefore the tracks should develop in parallel and converge at Whole-Body MPC's semantic-geometry stage.

## Relationship to Piper-JEPA

Piper-JEPA Stage A/B does not depend on Semantic Scene.

Piper-JEPA Stage C does.

Division of responsibility:

```text
Semantic Scene:
  Is this candidate robot state geometrically safe?

Piper-JEPA:
  Will the exact target remain visible / identifiable?
```

Piper-JEPA augments, rather than replaces, this deterministic geometry.

## Relationship to plant_twin

`plant_twin` is optional and complementary:
- Semantic Scene estimates current metric occupied/free structure.
- `plant_twin` estimates explicit deformation of selected plant structures.

Neither should block the other initially.

## Immediate starting point

1. Run the current RealSense→nvblox path on the actual Orin (it has been validated in the dev container; confirm which RealSense model is on the wrist).
2. Implement/validate the planner-facing distance query (a CPU v0 `SemanticDistanceQuery` exists and passes synthetic tests, P1.7; real-scene validation is open).
3. Build controlled thin-structure reference scenes.
4. Measure voxel size vs reconstruction coverage and latency.
5. Run semantic mask gating end to end (`class_demux_node` already produces per-class masks and depth).
6. Validate hard stem/branch and soft leaf policies.
7. Export the stable query interface to Whole-Body MPC.

**Status:** class policies (plus a default hard `other` class for non-plant depth), per-class nvblox config, `class_demux_node`, the RealSense→nvblox smoke path and a CPU v0 planner-facing query (`SemanticVoxelMap` + `SemanticDistanceQuery`, `scene_query_node`; P1.7.1–P1.7.5, synthetic tests only) exist; `scout_piper_whole_body_mpc` can embed the query in-process (`use_semantic_scene`, off by default). The MoveIt semantic collision plugin is still a scaffold that reports no collision; wiring it to the query (P1.7.7), real thin-structure runs and Orin timing (P1.7.6), and the nvblox-backed query path are the main open items.
