# 03 — Encoding models

`fit_encoding_models.py` 

Run the previous stages first:

```bash
python analysis/01_feature_extraction/extract_features.py
python analysis/02_feature_bands/build_feature_bands.py
```

Then run:

```bash
python analysis/03_encoding_models/fit_encoding_models.py
```

Inputs are read from:

```text
data/glmsingle/sub-*/
reproduced_outputs/feature_extraction/
reproduced_outputs/feature_bands/
```

Outputs are written to:

```text
reproduced_outputs/encoding_models/
```

The final model families are:

```text
M0_visual
M1_visual_rawcal
M2_visual_predclip
M3_visual_resclip
M4_visual_pred_resclip

D0_visual_noclip
D1_visual_clipband
D2_visual_clipband_predclip
```

Historical choices preserved include 5-fold outer CV, 5-fold inner CV,
100 random-search initializations, 20 hypergradient iterations, LOSO-group
behavioral calorie targets, and fold-specific train-only CLIP-to-calorie
decomposition.

`CaloriePredCLIP` and `CalorieResCLIP` are deliberately generated inside each
outer neural CV fold. They should not be precomputed globally for the main
encoding analysis.

The minimal release does not require full fMRIPrep dtseries files. If the
optional CIFTI template is absent, the historical code skips `.dscalar.nii`
writing; the `.npy` maps used for ROI inference are unaffected.
