# ResCLIP Step 2/3 diagnostics

Created: 2026-10-01T10:38:18

## Purpose

Characterize final CalorieResCLIP against item-level anchors, RDM-level behavioral/model anchors, and the R1 Calorie_profile_residual RDM.

## Inputs

- `final_resclip`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\reproduced_outputs\diagnostics\perceived_calorie_prediction_diagnostics\calorie_res_cv_CLIP.npy`
- `predclip_axis_full_table`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\reproduced_outputs\diagnostics\predCLIP_axis_characterization_v2\stimulus_scores_annotations_axes_residuals_FULL.csv`
- `r1_model_rdm_vectors_csv`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\historical_outputs\diagnostics\resclip_reliability_and_recovery\parent_RSA_input\model_rdm_vectors.csv`

## Outputs

- `step2A_item_anchors`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\reproduced_outputs\diagnostics\resclip_reliability_and_recovery_diagnostics\resclip_step2A_item_anchor_correlations.csv`
- `step2B_rdm_anchors`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\reproduced_outputs\diagnostics\resclip_reliability_and_recovery_diagnostics\resclip_step2B_rdm_anchor_correlations.csv`
- `step3_r1_bridge`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\reproduced_outputs\diagnostics\resclip_reliability_and_recovery_diagnostics\resclip_step3_r1_residual_bridge.csv`
- `pairwise_bridge`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\reproduced_outputs\diagnostics\resclip_reliability_and_recovery_diagnostics\resclip_step3_pairwise_bridge_values.csv`
- `top_pair_overlap`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\reproduced_outputs\diagnostics\resclip_reliability_and_recovery_diagnostics\resclip_step3_top_pair_overlap.csv`

## Main interpretation target

The key Step 3 test is the zero-order and controlled correlation between `ResCLIP_absdiff_RDM` and the R1 `Calorie_profile_residual` RDM.
