# Piper-JEPA

**Scope:** target-conditioned dense video state and predictive whole-body manipulation.

This track studies whether V-JEPA 2.1 dense temporal features and an embodiment-specific action-conditioned predictor can preserve and forecast the exact language-selected plant target under Scout + Piper motion.

## Documents

- `PIPER_JEPA_RESEARCH_PLAN.md` — scientific hypotheses, method, theory, and research positioning.
- `PIPER_JEPA_EXPERIMENTS.md` — dataset, trial, ablation, metric, and statistical protocol.
- `PIPER_JEPA_PAPER_OUTLINE.md` — manuscript structure and claim-to-evidence mapping.

## Relationship to other tracks

Piper-JEPA consumes:
- semantic geometry from `../semantic_scene/`;
- a geometry-only whole-body controller from `../whole_body_mpc/`;
- optional deformation state from `../plant_twin/`;
- later uncertainty-aware viewpoints from `../active_perception/`.

Piper-JEPA is one research component of Piper-Scout, not the entire system.
