# 05 — Diagnostic and characterization analyses

This component collects the non-neural diagnostic/characterization analyses
used by the manuscript.

The four main scripts are path-adapted versions of the historically executed
analyses. Numerical/statistical procedures are unchanged. The RV helper uses
the exact historical RV formula and persists a matrix that was historically
printed but not saved.

## Historical source provenance

- `05a_perceived_calorie_prediction.py` ← `plot_residual_calorie_diagnostic.py`; historical SHA-256 `80182b2b95fabdeee9977031eb4a4951cbafc735b985650620771800f981510d`
- `05b_predclip_axis_characterization.py` ← `predclip_axis_characterization_v2.py`; historical SHA-256 `59ea72258a8b2b26bb84ab5e305f5aa239db370fe176a8a900e5a3504d1829a4`
- `05c_clip_layer_rsa.py` ← `clip_layer_RSA.py`; historical SHA-256 `59ed96a2baf226da46efea0d408cbf6434535b9a84b832a65c60eff752e75e11`
- `05d_resclip_reliability_and_recovery.py` ← `resclip_reliability_and_clip_recovery_diagnostic.py`; historical SHA-256 `52c3dd8132d7d6b1568507c1b5861e079174b33b2dfca68a3279fe9525d7d90a`

## Inputs

```text
reproduced_outputs/feature_extraction/
reproduced_outputs/feature_bands/
data/stimuli/ordered_stimuli.csv
data/stimuli/images/
```

The ResCLIP RDM bridge uses the bundled parent-RSA input:

```text
resources/
  derived_inputs/
    parent_rsa/
      model_rdm_vectors.csv
```

This is a fixed input from the parent RSA analysis, not something regenerated
by this encoding-analysis repository.

## Outputs

```text
reproduced_outputs/diagnostics/
├── perceived_calorie_prediction_diagnostics/
├── predCLIP_axis_characterization_v2/
├── clip_openai_layerwise_processing_rsa/
├── resclip_reliability_and_recovery_diagnostics/
└── feature_overlap_rv/
```

## Run all diagnostics

From the repository root:

```bash
python analysis/05_diagnostics/run_all_diagnostics.py
```

The launcher runs each analysis as a separate Python process. It contains no
scientific calculations.

Order:

1. `05a` perceived-calorie prediction
2. `05b` PredCLIP axis characterization
3. `05c` CLIP layer RSA
4. `05d` ResCLIP reliability/recovery + item/RDM characterization
5. `05e` feature-space RV overlap

## Resume / partial run

```bash
python analysis/05_diagnostics/run_all_diagnostics.py --start-at 05c
python analysis/05_diagnostics/run_all_diagnostics.py --stop-after 05b
python analysis/05_diagnostics/run_all_diagnostics.py --start-at 05b --stop-after 05d
```

## Runtime

`05c` preserves the historical 10,000-permutation layerwise RSA and extracts
ViT-B/32 layer activations. `05d` preserves its historical reliability and
permutation settings. These two stages can therefore take substantially longer
than `05a`, `05b`, and `05e`.

The neural permutation controls for Supplementary Tables S8/S8b are not part
of this component; those remain a separate robustness stage.
