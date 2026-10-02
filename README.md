# Vision-language encoding models reveal an image-computable food-quality dimension in human occipitotemporal cortex

This repository supports two complementary workflows:

1. a **direct manuscript-reproduction route**, which recreates the reported figures and supplementary tables from the included canonical analysis outputs; and
2. a **full analysis rerun**, which starts from the public GLMsingle trial-level derivatives and reruns feature construction, encoding-model fitting, ROI inference, diagnostics, robustness analyses, and surface-map generation.

The top-level `run_pipeline.py` is a convenience launcher. It contains **no scientific calculations**; it calls the individual analysis and reproduction scripts in the documented order.

---

## Installation

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The repository also uses the pretrained OpenAI CLIP package listed in `requirements.txt`.

Connectome Workbench is needed only if you want to recreate the final cortical renderings used in Figure 3. It is not required for model fitting, ROI inference, diagnostics, supplementary tables, or the other manuscript figures.

---

## Quick start

### Recreate the reported manuscript outputs

No model fitting or imaging analysis is required:

```bash
python run_pipeline.py figures
```

This reads the canonical analysis outputs distributed in:

```text
historical_outputs/
```

and writes the recreated manuscript products to:

```text
reproduced_outputs/manuscript/
```

This is the fastest way to regenerate the analysis-derived manuscript figures and supplementary-table CSV files.

### Full analysis rerun

Download the required public inputs:

```bash
python run_pipeline.py download
```

Then run analysis steps 01–07:

```bash
python run_pipeline.py analysis
```

To recompute the descriptive split-half noise ceilings directly from the public GLMsingle trial betas:

```bash
python run_pipeline.py analysis --nc-mode recompute
```

After Step 07, Figure 3 requires one manual rendering step in Connectome Workbench; see **Figure 3** below.

---

# Repository structure

The distributed repository has the following top-level structure:

```text
<repository>/
│
├── README.md
├── requirements.txt
├── run_pipeline.py
├── generate_supplementary_tables_docx.py
│
├── analysis/
│   ├── 01_feature_extraction/
│   ├── 02_feature_bands/
│   ├── 03_encoding_models/
│   ├── 04_roi_inference/
│   ├── 05_diagnostics/
│   ├── 06_robustness/
│   └── 07_surface_maps/
│
├── data_download/
│   ├── README.md
│   ├── download_public_data.py
│   └── public_data_manifest.csv
│
├── figure_reproduction/
│   ├── README.md
│   ├── recreate_manuscript_outputs.py
│   ├── recreate_manuscript_outputs_from_rerun.py
│   └── static_inputs/
│       └── figure3/
│
├── historical_outputs/
│   └── ...
│
└── resources/
    ├── atlas/
    └── derived_inputs/
        ├── deep_features/
        └── parent_rsa/
```

The following directories are created locally when the corresponding workflows are run and are therefore not required to be present in the initial repository download:

```text
data/
reproduced_outputs/
```

`historical_outputs/` contains the canonical precomputed analysis outputs corresponding to the results reported in the manuscript. These files allow the manuscript-facing products to be regenerated without rerunning the computationally expensive encoding models.

---

# Public input data

The standard full-analysis workflow starts from the public GLMsingle trial-level derivatives rather than from raw fMRI data. These data originate from the companion study:

> Marrazzo G, Pimpini L, Kochs S, De Martino F, Valente G, Roefs A. *Representational structure of perceived food attributes in human occipitotemporal cortex*. iScience (2026). DOI: 10.1016/j.isci.2026.117660.

The corresponding public dataset is hosted on DataverseNL: https://doi.org/10.34894/TVPLVR.

Download the required inputs with:

```bash
python run_pipeline.py download

The downloader is pinned to DataverseNL release version 1.0 for DOI:

```text
10.34894/TVPLVR
```

and verifies downloaded files against the bundled manifest.

The downloaded inputs include:

- GLMsingle trial betas, stimulus order, and good-grayordinate masks for the 25 final participants;
- the 96 food images and canonical stimulus-order file;
- subject-level food ratings;
- inherited Gabor/CORnet inputs;
- the HCP-MMP atlas used for ROI definitions.

The exact parent-RSA derivative required by one CalorieResCLIP diagnostic is included under:

```text
resources/derived_inputs/parent_rsa/
```

because the public Dataverse copies are strict column subsets of the file used for that diagnostic.

---

# Full analysis pipeline

The full analysis is organized into seven explicit stages.

## 01 — Feature extraction

```bash
python analysis/01_feature_extraction/extract_features.py
```

Writes:

```text
reproduced_outputs/feature_extraction/
```

The analysis uses AlexNet, CORnet-S, and CLIP-derived image representations together with the low-level feature spaces described in the manuscript.

Reference AlexNet/CLIP feature matrices corresponding to the reported analysis are included under:

```text
resources/derived_inputs/deep_features/
```

Fresh feature extraction can show very small floating-point differences across software and hardware environments; these reference matrices provide the exact feature values associated with the reported analysis.

## 02 — Feature-band construction

```bash
python analysis/02_feature_bands/build_feature_bands.py
```

Writes:

```text
reproduced_outputs/feature_bands/
```

Primary bands:

- `LowVis`
- `HighVis`

Diagnostic bands:

- `HighVisNoCLIP`
- `CLIPown`

## 03 — Encoding-model fitting

```bash
python analysis/03_encoding_models/fit_encoding_models.py
```

Writes:

```text
reproduced_outputs/encoding_models/
```

Primary model family:

```text
M0_visual
M1_visual_rawcal
M2_visual_predclip
M3_visual_resclip
M4_visual_pred_resclip
```

CLIP-separated diagnostic family:

```text
D0_visual_noclip
D1_visual_clipband
D2_visual_clipband_predclip
```

The primary encoding analysis uses 5-fold outer cross-validation, 5-fold inner cross-validation, 100 random-search initializations, and 20 hypergradient iterations.

`CaloriePredCLIP` and `CalorieResCLIP` are generated separately within each outer neural cross-validation fold using the training images only.

## 04 — ROI inference and noise ceilings

```bash
python analysis/04_roi_inference/run_roi_inference.py
```

or, through the top-level launcher:

```bash
python run_pipeline.py analysis --start-step 4 --stop-step 4
```

To recompute the descriptive noise ceilings directly from trial-level GLMsingle outputs:

```bash
python run_pipeline.py analysis --start-step 4 --stop-step 4 --nc-mode recompute
```

Final hierarchy ROIs:

- **EarlyVisual:** V1 + V2 + V3 + V4
- **IntermediateVisual:** V8 + PIT + LO1 + LO2 + LO3
- **HighLevelVTC:** FFC + VVC + TE1p + TE2p

Noise ceilings are reported on the final correlation scale:

```text
NC(r) = sqrt(max(r_SB, 0))
```

They are descriptive and are not used for model fitting or primary statistical inference.

Writes:

```text
reproduced_outputs/roi_inference/
```

## 05 — Diagnostics and characterization

```bash
python analysis/05_diagnostics/run_all_diagnostics.py
```

Runs:

1. perceived-calorie prediction diagnostics;
2. PredCLIP-axis characterization;
3. CLIP layer-wise RSA;
4. CalorieResCLIP reliability, recovery, and anchor diagnostics;
5. feature-space RV-overlap diagnostics.

Writes:

```text
reproduced_outputs/diagnostics/
```

## 06 — Permutation robustness controls

```bash
python analysis/06_robustness/run_permutation_controls.py
```

Runs the final fixed-delta full-condition and PredCLIP-label permutation controls.

Writes:

```text
reproduced_outputs/robustness/permutation_null/
```

This stage is resumable.

## 07 — Surface-map generation

```bash
python analysis/07_surface_maps/run_surface_maps.py
```

Writes the descriptive group-level Workbench CIFTI maps to:

```text
reproduced_outputs/surface_maps/workbench_dscalars/
```

Formal statistical inference remains ROI-level; the surface maps are descriptive visualizations.

---

# Manuscript reproduction

## Reproduce manuscript products from the included canonical outputs

Run:

```bash
python run_pipeline.py figures
```

Equivalent direct command:

```bash
python figure_reproduction/recreate_manuscript_outputs.py
```

Inputs:

```text
historical_outputs/
figure_reproduction/static_inputs/figure3/
```

Outputs:

```text
reproduced_outputs/manuscript/
    figures/
    tables/
```

This route regenerates the reported analysis-derived manuscript figures and supplementary-table CSV files without rerunning the encoding analysis.

## Reproduce manuscript products after a full analysis rerun

After Steps 01–07, and after updating the Figure 3 Workbench screenshots described below, run:

```bash
python run_pipeline.py figures-from-rerun
```

Equivalent direct command:

```bash
python figure_reproduction/recreate_manuscript_outputs_from_rerun.py
```

Inputs are taken from:

```text
reproduced_outputs/
figure_reproduction/static_inputs/figure3/
```

and the resulting manuscript-facing products are written to:

```text
reproduced_outputs/manuscript/
```

---

# Generate the supplementary tables as a Word document

The manuscript-reproduction scripts generate each supplementary table as a CSV under:

```text
reproduced_outputs/manuscript/tables/
```

To combine these canonical table outputs into a single editable Word document, run from the repository root:

```bash
python generate_supplementary_tables_docx.py
```

The generated document is written to:

```text
reproduced_outputs/manuscript/Supplementary_Tables.docx
```

The script includes Supplementary Tables S1, S2, S3, S4, S4b, S5, S5b, S5c, S5d, S6, S7, S7b, S8, S8b, S9, S10, S11, S12, S13, and S14 in manuscript order.

Table values are transferred directly from the generated CSV files as strings; the script does not recalculate or re-round the reported statistical values. Scientific captions and figure panels can therefore be added separately when assembling the final Supplementary Information document.

---

# Figure 3

Figure 3 has one deliberate manual rendering step.

Step 07 programmatically regenerates the numerical CIFTI source maps. Final cortical rendering uses Connectome Workbench (`wb_view`) because camera angle, palette, display range, lighting, borders, and screenshot export are presentation choices rather than statistical calculations.

Panel sources are:

```text
B  mean_r_joint__M0_visual
C  mean_r_joint__M2_visual_predclip
D  mean_delta_r__M2_predclip_gt_M0_visual
```

After a fresh Step-07 rerun, render panels B–D using the same Workbench display settings used for the manuscript and place them at:

```text
figure_reproduction/static_inputs/figure3/
    M0_r_joint_uncorr.png
    M2_r_joint_uncorr.png
    M2-M0_r_joint_uncorr.png
```

Panel A is the ROI illustration:

```text
figure_reproduction/static_inputs/figure3/ROI.png
```

Then regenerate the manuscript products from the fresh analysis:

```bash
python run_pipeline.py figures-from-rerun
```

The included Figure 3 PNG inputs allow the reported Figure 3 layout to be reassembled without requiring Workbench when using `python run_pipeline.py figures`.

---

# Top-level launcher

Show available commands:

```bash
python run_pipeline.py --help
```

Check repository state:

```bash
python run_pipeline.py status
```

Download public inputs:

```bash
python run_pipeline.py download
```

Run all seven analysis stages:

```bash
python run_pipeline.py analysis
```

Run only a subset:

```bash
python run_pipeline.py analysis --start-step 3 --stop-step 5
```

Recompute noise ceilings:

```bash
python run_pipeline.py analysis --nc-mode recompute
```

Download the data and then run the analysis:

```bash
python run_pipeline.py analysis --download --nc-mode recompute
```

Print the analysis commands without executing them:

```bash
python run_pipeline.py analysis --dry-run
```

Recreate manuscript products from the included canonical outputs:

```bash
python run_pipeline.py figures
```

Recreate manuscript products after a full rerun:

```bash
python run_pipeline.py figures-from-rerun
```

Generate only supplementary-table CSV files:

```bash
python run_pipeline.py figures --tables-only
```

Generate only figures:

```bash
python run_pipeline.py figures --figures-only
```

Generate the combined supplementary-table Word document after creating the table CSVs:

```bash
python generate_supplementary_tables_docx.py
```

---

# Expected manuscript-facing products

The manuscript-reproduction scripts generate the analysis-derived products under:

```text
reproduced_outputs/manuscript/
    figures/
    tables/
```

Generated figures include:

```text
Figure 1D
Figure 2
Figure 3
Supplementary Figures S1, S2, S3, S4, S6
```

Generated tables include:

```text
Supplementary Tables S1, S2, S3, S4, S4b,
S5, S5b, S5c, S5d, S6, S7, S7b,
S8, S8b, S9, S10, S11, S12, S13, S14
```

Some manuscript artwork is intentionally static rather than generated by the statistical pipeline, including the experimental-design artwork, representative-stimulus panels, and the full stimulus montage.

---

# Suggested workflows

For a quick reproduction of the reported manuscript outputs:

```bash
python run_pipeline.py status
python run_pipeline.py figures
python generate_supplementary_tables_docx.py
```

For a full computational rerun:

```bash
python run_pipeline.py download
python run_pipeline.py analysis
```

After Step 07, recreate the Figure 3 Workbench screenshots and run:

```bash
python run_pipeline.py figures-from-rerun
python generate_supplementary_tables_docx.py
```

This separation keeps manuscript reproduction lightweight while preserving a complete path from the public input data through the full encoding analysis.
