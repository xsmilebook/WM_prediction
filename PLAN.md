# Plan

Implement merged-feature prediction under `prediction/V_feature_merge` while keeping the existing PLS workflow unchanged. Reuse the baseline `RandIndex.mat` files so merged models can be compared directly against the original GG/GW/WW runs.

## Scope
- In: `prediction/V_feature_merge`, merged-result summarization, and related documentation updates.
- Out: changes to the main single-feature prediction pipeline, preprocessing, or FC generation.

## Action items
- [ ] Refactor the local `PLSr1_CZ_Random_RegressCovariates.py` copy to support arbitrary named feature sets instead of hard-coded GG/GW/WW directories.
- [ ] Add merged-feature helpers that build `GG_GW`, `GG_WW`, `GW_WW`, and `GG_GW_WW` matrices and locate baseline `RandIndex.mat` files.
- [ ] Update `predict_age_RandomCV.py` to write merged outputs into `<target>/V_feature_merge/RegressCovariates_RandomCV`.
- [ ] Update `predict_cognition_RandomCV.py` and `predict_pfactor_RandomCV.py` to run the same merged workflow for all configured targets.
- [ ] Add a results summary script that compares baseline and merged models from `Res_NFold.mat`.
- [ ] Update `README.md`, `ARCHITECTURE.md`, and `docs/` so the new workflow and paths are documented.
- [ ] Run static validation on the modified Python files.
- [ ] Commit the code and documentation changes once validation passes.
