# Active Perception

**Scope:** task-aware viewpoint selection for thin, occluded plant structures.

Planned research questions:
- uncertainty-driven next-best-view selection;
- fusion of target uncertainty and metric geometry uncertainty;
- VGGT / multi-view reconstruction for local plant geometry;
- minimizing view count while preserving grasp-relevant reconstruction quality;
- using whole-body control to reach informative viewpoints.

This track should build on stable semantic geometry and whole-body control rather than becoming a prerequisite for the first Piper-JEPA result.

**Status:** first implementation, simulation and fake robot (2026-10-12):
[`Codes/src/scout_piper_active_perception/`](../../Codes/src/scout_piper_active_perception/).

What exists and what it showed:

- **Task-aware information.** The grasp needs the axis of the stem at the
  grasp point; its uncertainty is a cylinder fit's covariance, with each
  view's camera pose error fused per view (FisherRF's Fisher information,
  applied to the joint's five parameters instead of a radiance field). In
  simulation one clear view at 0.2-0.4 m fixes the axis to about 0.5-1 mm;
  what takes views is seeing past the leaves into the approach corridor that
  the MPC's no-entry rule needs known.
- **Fusion.** U_JEPA, U_geometry (joint, corridor) and U_twin are each mapped
  to 0.5 at their requirement. The view score weights each term's gain by how
  far that term is from being met (weighting by U itself kept spending views
  on a joint already known); a view is needed while a required term is unmet.
- **Reachable views.** Candidates are whole-body poses the MPC reaches: inside
  its reach margin and with a penalty on approach-axis turns (on the fake
  robot it stalled short of stretched-arm views). In simulation the fused
  policy is the fastest to a ready scene (14.5 s, 4.5 views) but not more
  often ready than a fixed ring of views (17 vs 19 of 30); with views at any
  arm pose the ring reaches 24/30 and the fused policy 18/30 (package README).
- **Thin structures.** The sensor model drops depth on stems narrower than a
  few pixels; a corridor voxel clears only with a depth return behind it.
  VGGT is wrapped (sparse-view batching, metric scale from the D405 or the
  robot's camera baselines, depth fusion) but has not run.

Open: VGGT on the GPU and the Orin; the D405 noise and dropout measured on the
robot; the ablation on real plants (MODULE_TASKS.md A14, B8, C10).
