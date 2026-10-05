# Plant Twin

**Scope:** explicit deformable modeling of leaf and stem geometry during manipulation and contact.

The implementation currently lives in `../../Codes/src/plant_twin/` and includes:
- deformable leaf mesh fitting;
- stem centerline fitting;
- temporal and structural priors;
- contact-conditioned residuals;
- analytic-Jacobian Gauss-Newton / LM fitting.

Research directions include:
- deformation-state estimation;
- contact-aware manipulation costs;
- validation of latent/world-model deformation representations;
- physical interpretation of learned predictors.

This track is complementary to Piper-JEPA: `plant_twin` models explicit deformation, while Piper-JEPA predicts future visual target state.

**Status:** implementation exists; dedicated research plan/paper scope not yet expanded.
