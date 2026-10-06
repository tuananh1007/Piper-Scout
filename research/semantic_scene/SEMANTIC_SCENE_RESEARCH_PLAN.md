# Semantic Scene Research Plan

**Working title:** *Semantic Metric Fields for Thin-Structure Plant Manipulation*  
**Track:** deterministic semantic geometry  
**Role:** provide metric collision/clearance authority for whole-body MPC and Piper-JEPA Stage C  
**Implementation base:** `Codes/src/scout_piper_scene_repr/`, RealSense D435, nvblox  
**Status:** aligned with `ROADMAP.md` as of 2026-10-05

---

## 1. Research objective

The semantic-scene track asks:

> **Can an explicit class-aware RGB-D signed-distance representation preserve thin plant structures well enough to improve manipulation planning over conventional occupancy/Octomap-style geometry, while remaining fast and reliable on Jetson AGX Orin?**

The purpose is not to build a learned world model. It is to construct the deterministic geometric layer that answers:

- where obstacles are;
- how far the robot is from them;
- which plant structures are hard obstacles;
- which structures may be brushed softly;
- where the selected target lies;
- whether the geometry is fresh and trustworthy.

This track is the **metric authority** for Piper-Scout.

---

## 2. Scientific gap

Conventional occupancy maps are poorly matched to plant manipulation because:
- thin stems and peduncles can disappear under voxelization;
- leaves should not necessarily be treated like rigid walls;
- the selected target must be distinguished from non-target plant material;
- class semantics matter for contact policy;
- depth from thin reflective/low-texture structures is intermittent;
- planner-facing distance queries require both accuracy and low latency.

The core research question is therefore not merely semantic segmentation, but **semantic metric geometry for contact-sensitive manipulation**.

---

## 3. Hypotheses

### H1 — Thin-structure preservation

Class-aware TSDF/ESDF integration with appropriate voxel resolution and temporal fusion will preserve stems/branches better than the current coarse occupancy baseline.

### H2 — Semantic collision policy

Class-dependent geometry policies will reduce false infeasibility relative to treating all plant material as hard obstacles.

### H3 — Reliable planner queries

Semantic signed-distance queries can provide sufficiently accurate and fresh clearance information for real-time whole-body control on Orin.

### H4 — Geometry confidence matters

Freshness/confidence gating will reduce unsafe or unstable planner behavior caused by stale or poorly observed thin geometry.

---

## 4. Proposed contributions

### C1 — Multi-class metric field

Maintain class-specific fields

```math
\phi_c(x)
```

for classes such as:
- non-target stem;
- branch;
- leaf;
- selected target.

### C2 — Manipulation-specific semantic policy

Map semantic class to planner behavior:

| Class | Policy |
|---|---|
| non-target stem | hard collision |
| branch | hard collision |
| leaf | soft interaction / clearance penalty |
| selected target | task attractor; gated collision exclusion near grasp |

### C3 — Thin-structure-aware fusion

Tune voxelization/fusion for narrow stems and branches, including:
- higher resolution for hard thin structures;
- temporal integration;
- minimum-observation/freshness rules;
- confidence-aware fallback when depth is missing.

### C4 — Planner-facing semantic distance API

Expose deterministic queries:
- signed distance;
- nearest obstacle class;
- local gradient/normal where supported;
- timestamp/freshness;
- validity/confidence.

---

## 5. Representation

For semantic class $c$, maintain a signed-distance field

```math
\phi_c:\mathbb R^3\rightarrow\mathbb R.
```

Interpretation:

```math
\phi_c(x)>0
```

outside the surface,

```math
\phi_c(x)=0
```

on the surface, and

```math
\phi_c(x)<0
```

inside occupied geometry where the backend supports signed distance.

For hard classes define

```math
\phi_{\mathrm{hard}}(x)
=
\min_{c\in\mathcal C_{\mathrm{hard}}}\phi_c(x).
```

For a robot collision primitive with center $`p_j(x_r)`$ and radius $r_j$,

```math
d_j
=
\phi_{\mathrm{hard}}(p_j)-r_j.
```

A hard-clearance constraint is

```math
d_j \ge d_{\mathrm{safe}}.
```

---

## 6. Semantic fusion

Let a depth observation produce a geometric update at voxel $v$, while semantic perception supplies class posterior $P(c\mid I_t,p)$.

A generic confidence-weighted class update can be written

```math
w_{c,t}(v)
=
w_{c,t-1}(v)
+
\alpha_t(v)P(c\mid I_t,p),
```

with accumulated evidence used to determine the class-specific field update.

The first implementation may remain simpler—mask-gated parallel TSDFs—provided the paper accurately describes the actual implementation.

### v0 implementation

Use four parallel fields:
- stem;
- branch;
- leaf;
- target.

This is preferable to inventing a complicated unified representation before validating the core need.

---

## 7. Thin-structure preservation

Key parameters to characterize experimentally:
- voxel size;
- truncation distance;
- minimum valid depth count;
- temporal integration horizon;
- semantic mask erosion/dilation;
- class inflation.

For a thin cylindrical structure of physical diameter $d_s$, define reconstruction coverage

```math
R_{\mathrm{cover}}
=
\frac{
\text{length of ground-truth centerline within }\epsilon\text{ of reconstructed geometry}
}{
\text{ground-truth centerline length}
}.
```

Also measure radial geometry error relative to a reference centerline or high-quality scan.

---

## 8. Leaf soft-cost model

Leaves should not necessarily produce hard infeasibility.

Define

```math
J_{\mathrm{leaf}}
=
\sum_{k,j}
\psi(
\phi_{\mathrm{leaf}}(p_j(x_k))
),
```

where $\psi$ penalizes penetration/proximity smoothly.

Example:

```math
\psi(d)
=
\begin{cases}
(d-d_{\mathrm{soft}})^2, & d<d_{\mathrm{soft}},\\
0, & d\ge d_{\mathrm{soft}}.
\end{cases}
```

Hard stem/branch constraints remain independent.

---

## 9. Target treatment

The selected target must not be accidentally removed from the world model globally.

Use explicit modes:

### Observation / approach mode

Target remains represented geometrically.

### Final grasp mode

Only the selected target region may be excluded from the non-target hard obstacle field, under:
- valid target ID;
- valid geometry;
- near-contact controller mode;
- bounded distance to grasp target.

Neighboring stems/branches remain hard obstacles.

---

## 10. Geometry freshness and confidence

Every planner query should expose timestamp $t_g$.

Define geometry age

```math
\Delta t_g
=
t_{\mathrm{now}}-t_g.
```

Reject or downweight geometry when

```math
\Delta t_g>\Delta t_{\max}.
```

For regions with insufficient observations, expose an unknown/invalid state rather than returning falsely confident free space.

This distinction is important for safety.

---

## 11. Interfaces to Whole-Body MPC

The semantic scene track should expose a stable interface independent of controller implementation.

Suggested API:

```text
query_distance(points, class_policy)
  -> distance[]
  -> gradient[] optional
  -> semantic_class[]
  -> valid[]
  -> timestamp
```

ROS/debug topics may publish visualization, but the high-rate planner should use in-process or low-overhead query paths where possible.

---

## 12. Relationship to Piper-JEPA

Piper-JEPA Stage A/B does not require this track.

Piper-JEPA Stage C does.

The division of responsibility is:

```text
Piper-JEPA:
  Will the target remain visible / identifiable?

Semantic Scene:
  Is the candidate robot state geometrically safe?
```

The two signals should remain separable in experiments.

---

## 13. Relationship to plant_twin

`plant_twin` models explicit deformation of selected plant structures.

Semantic Scene models current metric occupancy/clearance.

They may later be fused, but neither should initially depend on the other.

---

## 14. Experimental program

Use experiment IDs S1-S6.

### S1 — Thin-structure reconstruction accuracy
Measure stems/branches over different voxel sizes, distances, orientations, and depth quality.

### S2 — Semantic class integrity
Measure confusion/leakage between stem, branch, leaf, and target fields.

### S3 — Distance-query accuracy
Compare queried signed distance against controlled reference geometry.

### S4 — Planning feasibility
Compare Octomap/binary geometry versus semantic hard/soft policy.

### S5 — Temporal robustness
Measure behavior under intermittent depth/mask dropout.

### S6 — Orin deployment
Measure update latency, memory, query latency, and sustained operation.

Full protocol: `SEMANTIC_SCENE_EXPERIMENTS.md`.

---

## 15. Main baselines

G0 — legacy Octomap / binary occupancy.  
G1 — non-semantic TSDF/ESDF.  
G2 — semantic geometry with all plant classes hard.  
G3 — semantic class-aware hard/soft policy.  
G4 — G3 + freshness/confidence gating.

The important comparison is not only G0 versus G4. G2 versus G3 isolates the value of class-dependent manipulation semantics.

---

## 16. Main metrics

- centerline coverage;
- geometric surface error;
- missed thin-structure rate;
- false-obstacle rate;
- distance-query error;
- semantic field leakage;
- planner success rate;
- unsafe clearance violations;
- planning time;
- nvblox update latency;
- query latency;
- memory use.

---

## 17. Kill criteria

### K1

If the semantic representation does not preserve thin structures better than a simpler geometry baseline, do not claim thin-structure advantage.

### K2

If leaf soft-cost policy does not improve feasibility without unsafe interactions, keep leaves hard.

### K3

If query latency or freshness is inadequate for MPC, use a lower-rate geometry snapshot plus conservative safety margin.

---

## 18. Implementation mapping

Current implementation:
- `Codes/src/scout_piper_scene_repr/`;
- RealSense→nvblox smoke test;
- semantic collision plugin scaffold.

Recommended additions:

```text
scout_piper_scene_repr/
  semantic_integrator.py
  class_policy.py
  distance_query.py
  freshness.py
  benchmarks/
    reconstruction_eval.py
    distance_eval.py
    latency_eval.py
```

---

## 19. Immediate tasks

1. Bring up the current RealSense→nvblox path on the actual Orin.
2. Freeze semantic classes and class-policy YAML.
3. Implement mask-gated parallel semantic fields.
4. Finish planner-facing distance query.
5. Build S1 reference scenes with thin cylinders/stems of known geometry.
6. Measure voxel-size versus preservation/latency.
7. Validate hard stem/branch and soft leaf behavior.
8. Export a stable interface to Whole-Body MPC.

---

## 20. Publication positioning

A standalone paper is justified only if the work demonstrates a meaningful geometry/manipulation contribution beyond ordinary semantic mapping.

Strongest framing:

> **Semantic metric fields for thin, contact-sensitive plant manipulation, with class-dependent collision semantics and real-time planner queries.**

If the independent novelty is modest, this track should remain an enabling deterministic component and be presented as a strong baseline inside the Whole-Body MPC or Piper-JEPA paper.
