# Piper-Scout Research Program

This directory separates the major research threads within the broader Piper-Scout system. Each subdirectory is a research track with its own hypotheses, experiments, paper scope, and implementation links.

## Structure

- `piper_jepa/` — dense target memory, action-conditioned prediction, visibility/identity-aware whole-body control.
- `whole_body_mpc/` — deterministic non-holonomic Scout + Piper whole-body MPC and geometry-only control baselines.
- `plant_twin/` — explicit deformable leaf/stem digital-twin modeling and contact-conditioned deformation.
- `active_perception/` — uncertainty-driven next-best-view and task-aware multi-view perception.
- `semantic_scene/` — semantic RGB-D scene representation, nvblox/TSDF/ESDF, and class-aware collision geometry.

These research tracks are components of the full Piper-Scout system. They may support separate papers or be combined when the scientific evidence is stronger as one integrated contribution.

The system-level engineering sequence remains in `../ROADMAP.md`.
