# 06 — Permutation robustness controls

This component reruns the fixed-delta permutation controls reported in the
supplementary material, using the canonical regenerated M0/M2 encoding fits.

## Historical provenance

Release script:

```text
analysis/06_robustness/run_permutation_controls.py
```

is a path-adapted version of the final historically executed source:

```text
permutation_null_fixed_deltas_resume_highlevelvtc.py
```

Historical source SHA-256:

```text
9c78914bb2aab8ca6eb6bddcfdb1d9cc0df58e68203cd35c7339d2c4e614f8a1
```

Release script SHA-256:

```text
fa9de7ca73799af8ceb865c07d7ca12c567aa26d8bc199d8abdec47846b51653
```

The numerical/statistical procedure is unchanged. Changes are limited to
repository-relative paths, reviewer-facing naming, and metadata provenance.

## Inputs

The script reads:

```text
reproduced_outputs/feature_bands/
    band_LowVis.npy
    band_HighVis.npy

reproduced_outputs/diagnostics/
    perceived_calorie_prediction_diagnostics/
        calorie_pred_cv_CLIP.npy

reproduced_outputs/encoding_models/
    M0_visual/sub-*/deltas_folds.npy
    M0_visual/sub-*/combined_mask.npy
    M2_visual_predclip/sub-*/deltas_folds.npy
    M2_visual_predclip/sub-*/combined_mask.npy

data/glmsingle/sub-*/
    betas_trials.npy
    stimorder.npy
    good_grayordinates_mask.npy

resources/atlas/
    Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii
```

The historical script first checked for
`CaloriePredCLIP_loso_group.npy` and otherwise used
`calorie_pred_cv_CLIP.npy`. The release keeps that lookup logic; the standard
pipeline uses the 05a diagnostic output.

## Preserved historical analysis

The following are unchanged from the final historical script:

- 25-participant final cohort
- 100 full-condition shuffles
- 100 PredCLIP-label shuffles
- 5 outer folds
- `RANDOM_STATE = 42`
- fixed per-fold band deltas from the canonical M0/M2 fits
- scale-relative ridge factor `100.0`
- historical train/test kernel-centering implementation
- final ROI definitions:
  - EarlyVisual = V1 + V2 + V3 + V4
  - IntermediateVisual = V8 + PIT + LO1 + LO2 + LO3
  - HighLevelVTC = FFC + VVC + TE1p + TE2p
- resumable execution
- incremental write after every subject
- calorie-shuffle optimization that reuses the unchanged true M0 map
- original group summary, sign-flip, effect-size, FDR, and CI calculations

## Output

```text
reproduced_outputs/robustness/permutation_null/
    permutation_null_results.csv
    permutation_null_summary.csv
    calorie_specificity.csv
    calorie_specificity_summary.csv
    permutation_null_metadata.json
    figures/
```

## Run

From the repository root:

```bash
python analysis/06_robustness/run_permutation_controls.py
```

The script is resumable. If execution stops after one or more completed
participants, rerun the same command. A participant is skipped only when the
output CSV contains complete current-ROI results for the true condition,
all requested full-condition shuffles, and all requested calorie-label
shuffles.

### Important

Do not place the old historical `permutation_null_results.csv` into the new
output directory before the canonical rerun. The point of this run is to
generate robustness results from the newly frozen encoding fits. Historical
outputs remain reference material only.
