# ResCLIP reliability and richer-CLIP recovery diagnostic

Created: 2026-10-01T10:28:19

## Purpose

Step 0: estimate reliability of CalorieResCLIP by recomputing the CLIP residual in random split halves of behavioral rating subjects. Step 1: hold the final CalorieResCLIP target fixed and test whether alternative/richer CLIP representations recover it.

## Inputs

- `REPO_ROOT`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3`
- `FEAT_DIR`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\reproduced_outputs\feature_extraction`
- `BAND_DIR`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\reproduced_outputs\feature_bands`
- `ORIGINAL_DIAG_DIR`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\reproduced_outputs\diagnostics\perceived_calorie_prediction_diagnostics`
- `OUTDIR`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\reproduced_outputs\diagnostics\resclip_reliability_and_recovery_diagnostics`
- `FIGDIR`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\reproduced_outputs\diagnostics\resclip_reliability_and_recovery_diagnostics\figures`
- `Calorie_persubject`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\reproduced_outputs\feature_extraction\Calorie_persubject.npy`
- `CLIP_50`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\reproduced_outputs\feature_extraction\CLIP.npy`
- `CLIP_full512`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\reproduced_outputs\feature_extraction\CLIP_full512.npy`
- `final_ResCLIP`: `E:\NeuroRepFood\NeuroRepFood_CLIP_code_v3\reproduced_outputs\diagnostics\perceived_calorie_prediction_diagnostics\calorie_res_cv_CLIP.npy`

## Original decomposition matching

- CLIP target decomposition follows the logic of `plot_residual_calorie_diagnostic.py`.
- Predictor preprocessing: `StandardScaler(with_mean=True, with_std=True)`.
- Model: `RidgeCV(alphas=np.logspace(-4, 4, 25))`.
- CV: `KFold(n_splits=5, shuffle=True, random_state=42)`.

## Step 0 reliability summary

- ResCLIP split-half Spearman mean: `0.853508`
- ResCLIP Spearman-Brown reliability mean: `0.920634`
- ResCLIP correlation-scale ceiling mean: `0.959446`
- ResCLIP R² ceiling mean: `0.920634`

## Step 1 recovery summary

- `CLIP50_linear_ridge`: CV R²=`-0.038900`, Pearson r=`-0.295017`, Spearman rho=`-0.389813`, p_perm_R²=`0.569431`
- `CLIP50_rbf_kernel_ridge`: CV R²=`-0.000001`, Pearson r=`-0.121432`, Spearman rho=`-0.196795`, p_perm_R²=`nan`
- `CLIP512_linear_ridge`: CV R²=`-0.051560`, Pearson r=`-0.151514`, Spearman rho=`-0.065654`, p_perm_R²=`0.656344`
- `CLIP512_discarded_PC51plus_ridge`: CV R²=`-0.199064`, Pearson r=`-0.209113`, Spearman rho=`-0.185770`, p_perm_R²=`0.991009`
