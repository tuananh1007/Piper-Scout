# Semantic Scene Representation

**Scope:** explicit metric RGB-D geometry for thin-structure plant manipulation.

Primary components:
- RealSense RGB-D integration;
- nvblox TSDF / ESDF;
- semantic classes such as stem, branch, leaf, and selected target;
- class-dependent hard/soft collision policies;
- thin-structure preservation and uncertainty;
- planner-facing semantic clearance queries.

This track provides the metric geometry and collision authority used by whole-body MPC and Piper-JEPA.

**Status:** implementation scaffold and RealSense-to-nvblox smoke path exist; semantic class integration and planner query path remain active work.
