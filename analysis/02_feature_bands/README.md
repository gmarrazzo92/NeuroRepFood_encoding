# 02 — Feature-band construction

This component builds the feature bands used in the final encoding analyses.

`build_feature_bands.py` is the executed historical
`banded_ridge_feature_extraction.py` from the project archive
(`Python(2).zip`, 12 June 2026 version), with only filesystem paths redirected
to the release repository.

No numerical operation, PCA definition, normalization, residualization, or
diagnostic-band definition has been rewritten.

## Input

Run feature extraction first:

```bash
python analysis/01_feature_extraction/extract_features.py
```

The band script reads the reconstructed feature matrices from:

```text
reproduced_outputs/feature_extraction/
```

## Run

From the repository root:

```bash
python analysis/02_feature_bands/build_feature_bands.py
```

## Output

```text
reproduced_outputs/
└── feature_bands/
    ├── band_LowVis.npy
    ├── band_HighVis.npy
    ├── band_HighVisNoCLIP.npy
    ├── band_CLIPown.npy
    ├── band_Calorie.npy
    ├── band_Palatability.npy
    ├── band_Nuisance.npy
    └── band_meta.npy
```

## Definitions preserved from the historical code

- `LowVis`: Gabor + Color, each centered and scaled to unit kernel trace,
  concatenated, then PCA to 80 dimensions.
- `HighVis`: AlexNetMid + AlexNetHigh + CORnetIT + CLIP, each centered and
  scaled to unit kernel trace, concatenated, then PCA to 80 dimensions.
- `HighVisNoCLIP`: AlexNetMid + AlexNetHigh + CORnetIT with the same
  construction, PCA to 80 dimensions.
- `CLIPown`: standalone `CLIP.npy`, centered and scaled to unit kernel trace,
  with no additional PCA.
- `Calorie`: z-scored perceived-calorie feature.
- `Palatability`: residualized against Familiarity and Calorie, then z-scored.
- `Nuisance`: z-scored Familiarity.

The main M0–M4 model family uses `LowVis` and `HighVis`.
The CLIP-separated D0–D2 diagnostic uses `LowVis`, `HighVisNoCLIP`, and
`CLIPown`.

