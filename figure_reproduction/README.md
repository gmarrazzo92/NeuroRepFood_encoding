# Manuscript and Supplementary Output Reproduction

This directory provides two complementary routes for recreating the manuscript figures and supplementary tables.

1. **Direct manuscript reproduction** uses the validated analysis outputs bundled in `historical_outputs/`. It is the fastest route and does not require the public imaging derivatives or rerunning the encoding analysis.
2. **Full-rerun manuscript reproduction** uses newly generated files under `reproduced_outputs/` after analysis steps 01–07 have been completed.

Both routes write their products to `reproduced_outputs/manuscript/`.

---

## Scripts

```text
figure_reproduction/
    recreate_manuscript_outputs.py
    recreate_manuscript_outputs_from_rerun.py
    static_inputs/
        figure3/
            ROI.png
            M0_r_joint_uncorr.png
            M2_r_joint_uncorr.png
            M2-M0_r_joint_uncorr.png
```

### Direct reproduction from bundled reference outputs

Run from the repository root:

```bash
python figure_reproduction/recreate_manuscript_outputs.py
```

or equivalently:

```bash
python run_pipeline.py figures
```

This route reads the validated reference analysis outputs distributed under:

```text
historical_outputs/
```

The directory name is retained for compatibility with the repository structure. Its contents are the reference outputs corresponding to the reported analysis.

No GLMsingle data, feature extraction, model fitting, ROI inference, diagnostic analyses, or permutation analyses are required for this route.

### Reproduction from a fresh full rerun

After analysis steps 01–07 have completed, run:

```bash
python figure_reproduction/recreate_manuscript_outputs_from_rerun.py
```

or equivalently:

```bash
python run_pipeline.py figures-from-rerun
```

This script reads from:

```text
reproduced_outputs/
```

and stops during preflight if required regenerated inputs are missing. It does not fall back to `historical_outputs/`.

---

## Which route should I use?

| Goal | Route | Full analysis rerun required? |
|---|---|---:|
| Recreate the reported figures and supplementary tables | `run_pipeline.py figures` | No |
| Inspect the numerical reference outputs behind the manuscript | `historical_outputs/` | No |
| Verify the complete analysis chain from public GLMsingle derivatives | analysis steps 01–07, then `figures-from-rerun` | Yes |
| Modify the analysis and propagate the changes into manuscript outputs | analysis steps 01–07, then `figures-from-rerun` | Yes |

---

## Bundled reference outputs

The direct reproduction script consumes only the files required to reconstruct the reported manuscript products. They are organized approximately as:

```text
historical_outputs/
    feature_extraction/
    bands/
    diagnostics/
        perceived_calorie_prediction/
        predclip_axis_characterization/
        clip_layer_rsa/
        resclip_reliability_and_recovery/
        feature_overlap_rv/
    roi_inference/
        main/
        clip_separated_diagnostic/
    robustness/
        permutation_null/
    surface_maps/
        workbench_dscalars/
```

These files are included as validated reference outputs. The direct reproduction workflow does not modify them.

---

## Figure 3

Figure 3 is a special case because the numerical cortical maps are generated programmatically, whereas final cortical-surface rendering is performed in Connectome Workbench.

Analysis step 07 creates the Workbench `.dscalar.nii` source maps. The manuscript panels B–D are rendered in `wb_view`; panel A is the ROI illustration. The four image inputs used to assemble the reported figure are:

```text
figure_reproduction/static_inputs/figure3/
    ROI.png
    M0_r_joint_uncorr.png
    M2_r_joint_uncorr.png
    M2-M0_r_joint_uncorr.png
```

Panel mapping:

```text
A  ROI.png
B  M0_r_joint_uncorr.png
C  M2_r_joint_uncorr.png
D  M2-M0_r_joint_uncorr.png
```

For direct manuscript reproduction, these rendered PNGs are already included, so Connectome Workbench is not required.

For a fresh full rerun, regenerate the step-07 Workbench maps, render panels B–D with the manuscript display settings, replace the three corresponding PNGs under `static_inputs/figure3/`, and then run `python run_pipeline.py figures-from-rerun`.

The underlying `.dscalar.nii` files remain available so the plotted surface values can be inspected independently of the screenshots.

---

## Noise ceilings

The manuscript-reproduction scripts do not recompute noise ceilings. Noise ceilings are computed or loaded during ROI inference and the resulting ROI-level values are then consumed during figure/table generation.

Direct reproduction reads:

```text
historical_outputs/roi_inference/main/noise_ceiling_roi_values.csv
```

A fresh rerun reads:

```text
reproduced_outputs/roi_inference/noise_ceiling_roi_values.csv
```

The stored `nc_r` values are already on the final correlation scale. The reproduction scripts use them to calculate the displayed group summaries and do not reapply the Spearman–Brown or square-root transformations.

---

## Generated manuscript products

Both reproduction routes write to:

```text
reproduced_outputs/manuscript/
    figures/
    tables/
```

The analysis-derived products are:

```text
Figures
    Figure 1D
    Figure 2
    Figure 3
    Supplementary Figures S1, S2, S3, S4, S6

Tables
    Supplementary Tables S1, S2, S3, S4, S4b,
    S5, S5b, S5c, S5d, S6, S7, S7b,
    S8, S8b, S9, S10, S11, S12, S13, S14
```

To combine the generated supplementary-table CSV files into a Word document, run:

```bash
python generate_supplementary_tables_docx.py
```

Some manuscript artwork is intentionally static rather than regenerated by the statistical pipeline, including the experimental-design artwork, representative-stimulus panels, and the full stimulus montage.

---

## Complete computational route

A full rerun proceeds in this order:

```text
01  Feature extraction
02  Feature-band construction
03  Encoding-model fitting
04  ROI inference and noise ceilings
05  Diagnostic and characterization analyses
06  Permutation robustness controls
07  Surface-map generation
```

The top-level launcher runs these stages with:

```bash
python run_pipeline.py analysis
```

To recompute the split-half noise ceilings from the downloaded GLMsingle trial betas instead of loading the bundled reference maps, use:

```bash
python run_pipeline.py analysis --nc-mode recompute
```

After step 07, recreate the Figure 3 Workbench screenshots as described above and run:

```bash
python run_pipeline.py figures-from-rerun
python generate_supplementary_tables_docx.py
```

The two manuscript-reproduction routes are therefore complementary: one provides a lightweight recreation of the reported outputs from validated reference results, while the other verifies the complete path from the public analysis inputs through newly generated manuscript products.
