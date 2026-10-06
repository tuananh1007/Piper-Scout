# Semantic Scene Experiments

**Companion to:** SEMANTIC_SCENE_RESEARCH_PLAN.md  
**Purpose:** executable evaluation protocol for deterministic semantic geometry

---

## 1. Experimental questions

S1. Does the representation preserve thin plant structures?  
S2. Are semantic classes correctly separated in metric space?  
S3. Are planner distance queries accurate?  
S4. Do class-aware hard/soft policies improve planning?  
S5. Is the representation robust to intermittent observations?  
S6. Can it run sustainably on Jetson AGX Orin?

---

## 2. Method IDs

G0 — legacy Octomap/binary occupancy.  
G1 — non-semantic TSDF/ESDF.  
G2 — semantic fields, all plant classes hard.  
G3 — semantic class-aware hard/soft policy.  
G4 — G3 + freshness/confidence gating.

Keep method IDs fixed.

---

## 3. Reference geometry

Use two kinds of test scenes.

### Controlled geometry

Use rods/cylinders or fabricated plant-like structures with known:
- diameter;
- centerline;
- pose;
- spacing.

This gives reliable geometric ground truth.

### Real plants

Use manually annotated or higher-quality multi-view reference reconstructions for ecological validity.

Do not call RealSense itself ground truth.

---

## 4. S1 — Thin-structure reconstruction

### Factors

- diameter: e.g. 2, 3, 5, 8, 12 mm where feasible;
- camera distance;
- orientation relative to optical axis;
- voxel size;
- motion/static;
- lighting.

### Metrics

Centerline coverage:

```math
R_{\mathrm{cover}}
=
\frac{L_{\mathrm{matched}}}{L_{\mathrm{gt}}}.
```

Mean centerline error:

```math
E_c
=
\frac1N
\sum_i
\min_j
\|p_i^{gt}-p_j^{rec}\|_2.
```

Miss rate:

```math
R_{\mathrm{miss}}
=
1-R_{\mathrm{cover}}.
```

Also report false geometry volume around the structure.

### Recommended size

At least 5 physical structures × 5 poses × 3 repeats per key voxel configuration.

---

## 5. S2 — Semantic class integrity

Construct scenes containing adjacent:
- stem;
- branch;
- leaf;
- selected target.

Measure:
- semantic precision/recall in projected 3-D;
- field cross-contamination;
- target/non-target leakage;
- class-switch rate over time.

A useful metric is voxel semantic purity:

```math
P_{\mathrm{sem}}
=
\frac{
\#\text{correctly labeled observed voxels}
}{
\#\text{labeled observed voxels}
}.
```

---

## 6. S3 — Distance-query accuracy

Place robot collision-query points at known offsets from controlled geometry.

For query point $x_i$:

```math
e_i
=
|\hat d_i-d_i^*|.
```

Report:
- mean absolute error;
- RMSE;
- p95 error;
- conservative error rate:

```math
P(\hat d_i>d_i^*+\epsilon),
```

because overestimating clearance is safety-relevant.

### Query classes

Test:
- stem;
- branch;
- leaf;
- hard-min field.

---

## 7. S4 — Planning feasibility

Use a fixed set of cluttered plant scenes.

Compare G0-G4 with the same planner/controller.

### Primary endpoint

Planning/execution success.

### Secondary endpoints

- no-plan rate;
- minimum true clearance;
- path length;
- planning time;
- number of leaf contacts;
- number of stem/branch contacts;
- unnecessary detours.

### Critical comparison

G2 versus G3 tests whether semantic hard/soft behavior adds value beyond semantics alone.

---

## 8. S5 — Temporal robustness

Introduce:
- depth dropout;
- segmentation dropout;
- partial occlusion;
- short camera interruption.

Measure:
- geometry persistence;
- stale geometry duration;
- false-free events;
- false-obstacle events;
- recovery time.

Evaluate G3 versus G4 to test freshness/confidence gating.

---

## 9. S6 — Orin deployment

Measure on Jetson AGX Orin:
- TSDF/ESDF update latency;
- semantic update latency;
- query latency per point and batch;
- memory;
- GPU utilization;
- power;
- sustained rate;
- thermal behavior.

Benchmark realistic batch sizes from the whole-body controller.

---

## 10. Scene set

Recommended minimum:
- 20 controlled thin-geometry scenes;
- 20 real plant scenes;
- 10 dense clutter scenes for planning;
- 10 dropout/temporal sequences.

Reuse scene IDs across methods to enable paired comparisons.

---

## 11. Statistics

For paired continuous metrics:
- paired bootstrap CI;
- paired t-test if assumptions hold;
- Wilcoxon signed-rank otherwise.

For success/failure:
- mixed-effects logistic regression or paired categorical analysis by scene.

Report:
- effect size;
- 95% CI;
- exact scene count.

---

## 12. Pass criteria

### Thin geometry

G3/G4 must improve thin-structure coverage or reduce miss rate versus G0/G1.

### Semantic policy

G3 should improve planning feasibility over G2 without increasing unsafe hard-structure contacts.

### Query reliability

Clearance overestimation must be characterized and bounded with an appropriate safety margin.

### Deployment

Planner query/update latency must be compatible with the selected whole-body controller architecture.

---

## 13. Result tables

### Table S-A — geometry

| Method | Coverage ↑ | Centerline error ↓ | Miss rate ↓ | False geometry ↓ |
|---|---:|---:|---:|---:|

### Table S-B — planning

| Method | Success ↑ | No-plan ↓ | Min clearance ↑ | Leaf contact | Hard contact ↓ |
|---|---:|---:|---:|---:|---:|

### Table S-C — deployment

| Voxel size | Update ms ↓ | Query ms ↓ | Memory | Coverage ↑ |
|---|---:|---:|---:|---:|

---

## 14. Immediate experimental tasks

1. Build controlled thin-cylinder reference scene.
2. Record RGB-D at multiple distances/orientations.
3. Evaluate current nvblox geometry with 2-3 voxel sizes.
4. Add semantic mask gating.
5. Implement query test harness.
6. Freeze G0-G4 definitions before S4.
