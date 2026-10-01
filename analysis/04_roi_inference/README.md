# 04 — ROI inference and model comparison

This stage performs the final ROI-level inference for the encoding analysis.

`run_roi_inference.py` is based on the final executed historical ROI-inference
script containing both the primary M0–M4 family and the CLIP-separated D0–D2
diagnostic family. Only filesystem paths and reviewer-facing naming were
changed; the statistical calculations, model comparisons, ROI definitions,
random seeds, FDR families, and figure-generation code are unchanged.

## Prerequisite

Run:

```bash
python analysis/03_encoding_models/fit_encoding_models.py
```

The script reads the eight model families from:

```text
reproduced_outputs/encoding_models/
```

and also reads:

```text
data/glmsingle/sub-*/
resources/atlas/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii
```

## Run

```bash
python analysis/04_roi_inference/run_roi_inference.py
```

## Final hierarchy ROIs

- EarlyVisual = V1 + V2 + V3 + V4
- IntermediateVisual = V8 + PIT + LO1 + LO2 + LO3
- HighLevelVTC = FFC + VVC + TE1p + TE2p

## Historical inference logic preserved

- Common finite vertex mask across all available models within each subject/ROI.
- Fisher-z transformed `r_joint` for primary model-performance inference.
- Nested model gains reported as raw `delta_r`, with inference on `delta_z`.
- Hierarchy slope inference uses `slope_delta_z_per_step`.
- Hierarchy pairwise inference uses `delta_z`.
- M4 split inference uses raw `r_split`.
- Sign-flip seeds are preserved: +1 for model performance, +2 for model
  comparisons, +30/+31/+32 for slopes, +40 for hierarchy pairwise tests,
  +50 for split-band tests, and +51 for the PredCLIP–ResCLIP split contrast.
- Primary M0–M4 comparisons and D0–D2 diagnostic comparisons remain separate
  FDR families.

## Noise ceiling

The historical split-half/Spearman–Brown definition is preserved:

```text
NC(r) = sqrt(max(r_SB, 0))
```

The script exposes:

```python
NC_MODE = "historical"
```

with three allowed modes:

- `"historical"` — load the frozen NC maps supplied in
  `historical_outputs/`. This is the default for exact manuscript/figure
  reproduction and fails if a required historical NC map is missing.
- `"cached"` — load only maps already present in
  `reproduced_outputs/roi_inference/noise_ceiling_cache/`. Missing maps are
  omitted and are **not** recomputed.
- `"recompute"` — recompute every participant's NC map from the public
  GLMsingle trial betas and overwrite/update the reproduced cache.

The mode can be selected directly from the command line:

```bash
python analysis/04_roi_inference/run_roi_inference.py --historical-nc
python analysis/04_roi_inference/run_roi_inference.py --cached-nc
python analysis/04_roi_inference/run_roi_inference.py --recompute-nc
```

The corresponding single-dash aliases (`-historical-nc`, `-cached-nc`,
`-recompute-nc`) are also accepted. An explicit value form is available as
well:

```bash
python analysis/04_roi_inference/run_roi_inference.py --nc-mode recompute
```

If no NC option is supplied, `historical` is used.

Noise ceilings are descriptive and are not used for model fitting or primary statistical inference. Because their computation is relatively expensive, the default historical mode loads the frozen noise-ceiling maps used for the manuscript. Users who wish to verify the computation from the public GLMsingle outputs can instead use --recompute-nc; the resulting maps are cached for subsequent runs.

## Outputs

All tables, diagnostics, traceability metadata, and analysis figures are written
to:

```text
reproduced_outputs/roi_inference/
```
