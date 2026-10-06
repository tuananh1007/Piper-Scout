# Semantic Scene Paper Outline

**Working title:** *Semantic Metric Fields for Thin-Structure Plant Manipulation*

---

## 1. Thesis

Manipulation in plants requires more than generic occupancy: thin stems must remain geometrically visible, leaves and rigid branches should not share the same collision policy, and real-time control needs reliable metric distance queries.

---

## 2. Abstract structure

1. Thin/deformable plant geometry challenges occupancy maps.
2. Introduce class-aware metric fields with manipulation-specific collision semantics.
3. Describe thin-structure fusion + planner-facing distance queries.
4. Report reconstruction, planning, and Orin deployment results.
5. State whether planning feasibility improves without compromising hard-structure clearance.

---

## 3. Introduction

### P1 — failure mode
Thin stems vanish; leaves overconstrain planning when treated as rigid.

### P2 — gap
Semantic segmentation alone is not enough; manipulation needs metric, queryable geometry.

### P3 — approach
Class-specific TSDF/ESDF + hard/soft policy + freshness.

### P4 — contributions
- thin-structure semantic metric field;
- class-aware manipulation policy;
- planner-facing query layer;
- physical benchmark.

---

## 4. Related work

### 4.1 RGB-D mapping / TSDF / ESDF
### 4.2 Semantic mapping
### 4.3 Agricultural scene reconstruction
### 4.4 Collision representations for manipulation

End by positioning the work around **manipulation-specific semantics and thin-structure metric reliability**.

---

## 5. Method

### 5.1 Class-specific fields

```math
\phi_c(x)
```

### 5.2 Hard field

```math
\phi_{\mathrm{hard}}(x)
=
\min_{c\in\mathcal C_{\mathrm{hard}}}
\phi_c(x).
```

### 5.3 Leaf soft cost

```math
J_{\mathrm{leaf}}
=
\sum\psi(\phi_{\mathrm{leaf}}).
```

### 5.4 Target mode handling

Explain approach versus final grasp mode.

### 5.5 Freshness/unknown geometry

Explain timestamp and validity gating.

### 5.6 Planner query API

Show batched query interface.

---

## 6. Experiments

Map directly to:
- S1 thin reconstruction;
- S2 semantic integrity;
- S3 query accuracy;
- S4 planning;
- S5 temporal dropout;
- S6 Orin.

---

## 7. Results

### R1 — thin-structure preservation
Primary quantitative geometry result.

### R2 — semantic separation
Show target/non-target and leaf/stem field integrity.

### R3 — planner query accuracy
Emphasize conservative error.

### R4 — class-aware planning
Headline manipulation result.

### R5 — deployment
Latency/memory trade-off.

---

## 8. Figures

Figure 1 — representation architecture.  
Figure 2 — thin stem reconstruction versus baseline.  
Figure 3 — semantic class fields.  
Figure 4 — planner trajectories under binary vs class-aware geometry.  
Figure 5 — accuracy/latency trade-off.

---

## 9. Tables

Table 1 — geometry reconstruction.  
Table 2 — planning success/safety.  
Table 3 — Orin runtime.

---

## 10. Discussion

Explain:
- where semantics change the control problem;
- where depth sensing remains limiting;
- why unknown geometry should not equal free space;
- how the representation feeds Whole-Body MPC and Piper-JEPA.

---

## 11. Limitations

- RealSense thin-structure failure;
- semantic mask errors;
- dynamic plant motion;
- parallel-field memory cost;
- no full deformation prediction.

---

## 12. Publication decision

Submit standalone only if S1-S4 show a clear methodological gain.

Otherwise integrate this track as the deterministic geometry section/baseline in the Whole-Body MPC or Piper-JEPA paper.
