# Manuscript and Supplementary Output Reproduction

This directory contains two complementary manuscript-output workflows.

The two workflows generate the same manuscript-facing figures and supplementary
tables, but they are intended for different use cases:

1. **Fast reproduction from frozen outputs** — for reviewers or users who want
   to recreate the manuscript figures/tables without rerunning the full analysis.
2. **Full-rerun reproduction** — for users who rerun analysis steps 01–07 and
   want the manuscript products to be regenerated from those newly computed
   outputs.

The distinction is important. The fast workflow should normally be sufficient
for checking the manuscript figures and tables. The full-rerun workflow is only
needed when the underlying analysis itself is rerun.

---

## Recommended script names

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

### `recreate_manuscript_outputs.py`

This is the **fast manuscript-reproduction script**.

It reads the frozen analysis products stored under:

```text
historical_outputs/
```

and recreates the manuscript-facing quantitative figures and supplementary
tables without requiring the imaging data, GLMsingle, feature extraction,
encoding-model fitting, ROI inference, diagnostics, or permutation analyses to
be rerun.

Despite the legacy directory name, `historical_outputs/` should be understood
in the release as the **frozen canonical manuscript-source snapshot**. Its
contents are the validated outputs corresponding to the final reported
analysis. The directory name is retained for compatibility with the existing
reproduction script and repository structure.

This is the workflow most reviewers should use.

Run from the repository root:

```bash
python figure_reproduction/recreate_manuscript_outputs.py
```

---

### `recreate_manuscript_outputs_from_rerun.py`

This is the **full-rerun manuscript-reproduction script**.

It reads directly from:

```text
reproduced_outputs/
```

after analysis steps 01–07 have been rerun. It should be used when a user wants
to verify that a fresh execution of the analysis pipeline reproduces the
manuscript-facing results.

Run from the repository root only after the upstream analysis has completed:

```bash
python figure_reproduction/recreate_manuscript_outputs_from_rerun.py
```

This script performs a preflight check and stops if required regenerated inputs
are missing. It does not fall back to `historical_outputs/`.

---

# Which workflow should I use?

| Goal | Script | Need to rerun analysis? |
|---|---|---:|
| Recreate the reported manuscript figures/tables | `recreate_manuscript_outputs.py` | No |
| Inspect the numerical source files behind the manuscript | `historical_outputs/` | No |
| Verify that a fresh analysis rerun gives the same manuscript results | `recreate_manuscript_outputs_from_rerun.py` | Yes |
| Modify the analysis and propagate the changes into manuscript outputs | `recreate_manuscript_outputs_from_rerun.py` | Yes |

The fast workflow is therefore the default reproducibility route. The
full-rerun workflow is a deeper computational validation route.

---

# Frozen manuscript-source snapshot

For the fast workflow to be self-contained, the final validated outputs from
the full analysis are copied into the corresponding `historical_outputs/`
subdirectories.

The active snapshot should contain the final canonical results required by the
manuscript reproduction script, approximately as follows:

```text
historical_outputs/
    feature_extraction/

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

The exact filenames are those consumed by
`recreate_manuscript_outputs.py`. Files not needed for manuscript reproduction
do not need to be duplicated into this snapshot.

The original pre-rerun analysis bundle should be preserved separately for
author-side provenance and should not be silently overwritten. The active
`historical_outputs/` directory in the release should contain the **validated
canonical snapshot used to regenerate the final manuscript**, not a mixture of
old and new analysis states.

---


## Automated snapshot refresh

The release includes:

```text
refresh_historical_outputs_from_rerun.py
```

This maintainer script performs the refresh described above automatically. It
maps the validated regenerated output directories onto the legacy-compatible
`historical_outputs/` layout, replaces the managed snapshot directories rather
than merging them, and regenerates the checksum manifest.

Before making any changes, inspect the plan with:

```bash
python refresh_historical_outputs_from_rerun.py --dry-run
```

Then refresh the snapshot with:

```bash
python refresh_historical_outputs_from_rerun.py
```

On its first run, the script preserves the existing pre-canonical snapshot as:

```text
historical_outputs_original_backup/
```

This backup is for author-side provenance and should not be included in the
public release.

The refresh script maps:

```text
reproduced_outputs/feature_extraction
    -> historical_outputs/feature_extraction

reproduced_outputs/feature_bands
    -> historical_outputs/bands

reproduced_outputs/diagnostics/perceived_calorie_prediction_diagnostics
    -> historical_outputs/diagnostics/perceived_calorie_prediction

reproduced_outputs/diagnostics/predCLIP_axis_characterization_v2
    -> historical_outputs/diagnostics/predclip_axis_characterization

reproduced_outputs/diagnostics/clip_openai_layerwise_processing_rsa
    -> historical_outputs/diagnostics/clip_layer_rsa

reproduced_outputs/diagnostics/resclip_reliability_and_recovery_diagnostics
    -> historical_outputs/diagnostics/resclip_reliability_and_recovery

reproduced_outputs/diagnostics/feature_overlap_rv
    -> historical_outputs/diagnostics/feature_overlap_rv

reproduced_outputs/roi_inference
    -> historical_outputs/roi_inference/main
    -> historical_outputs/roi_inference/clip_separated_diagnostic

reproduced_outputs/robustness/permutation_null
    -> historical_outputs/robustness/permutation_null

reproduced_outputs/surface_maps/workbench_dscalars
    -> historical_outputs/surface_maps/workbench_dscalars
```

The same canonical ROI-inference directory is frozen under both legacy ROI
aliases because the original manuscript-reproduction layout expects the main
outputs and CLIP-separated diagnostic under separate historical paths, whereas
the final rerun writes them together.


# Refreshing `historical_outputs/` after a validated full rerun

This step is a **maintainer/release step**, not something a reviewer needs to
perform.

After analysis steps 01–07 have been rerun and validated:

1. Copy the required final outputs from `reproduced_outputs/` into the matching
   `historical_outputs/` locations expected by
   `recreate_manuscript_outputs.py`.
2. Preserve the directory/file naming expected by the fast reproduction
   script, even where the regenerated analysis uses slightly different
   directory names.
3. Copy the final Figure 3 screenshot inputs into the release static-input
   location.
4. Freeze the refreshed `historical_outputs/` snapshot and regenerate its
   checksum manifest.
5. Run `recreate_manuscript_outputs.py` to recreate the final manuscript-facing
   figures and tables from the frozen snapshot.

Only after this validation should the frozen snapshot be treated as the source
for the fast reviewer-facing reproduction workflow.

---

# Figure 3

Figure 3 is a special case because the underlying cortical maps are generated
programmatically, while final surface rendering is performed in Connectome
Workbench.

Analysis step 07 generates the canonical Workbench source maps. Panels B–D are
then rendered manually in `wb_view` using the same display settings used for
the manuscript. Panel A is the unchanged ROI illustration.

The four assembly inputs are:

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

For a full rerun, panels B–D should be recreated from the regenerated step-07
Workbench maps before running
`recreate_manuscript_outputs_from_rerun.py`.

For the fast reproduction route, the already rendered canonical PNG inputs are
provided, so Connectome Workbench is **not required** simply to reassemble the
reported Figure 3.

The underlying `.dscalar.nii` maps are retained separately so the displayed
surface values can still be inspected independently of the screenshot
assembly.

---

# Noise ceilings

Noise ceilings are **not recomputed by either manuscript-reproduction script**.

They are computed upstream during ROI inference and then read from the
corresponding subject-level ROI output:

Fast/frozen route:

```text
historical_outputs/roi_inference/main/noise_ceiling_roi_values.csv
```

Full-rerun route:

```text
reproduced_outputs/roi_inference/noise_ceiling_roi_values.csv
```

The stored `nc_r` values are already on the final correlation scale. The
manuscript-reproduction scripts use these values to calculate the displayed
group mean and 95% confidence interval and to draw the noise-ceiling bands.
They do not apply the Spearman–Brown correction or square-root transformation
again.

---

# Manuscript-facing outputs

Both workflows write to:

```text
reproduced_outputs/manuscript/
    figures/
    tables/
```

The analysis-derived products include:

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

Some manuscript artwork is intentionally static rather than regenerated by the
analysis code, including the experimental-design artwork, stimulus-image panel,
and stimulus montage. These are presentation/stimulus assets rather than
outputs of the statistical analysis.

---

# Full analysis route

A complete computational rerun follows the analysis stages in order:

```text
01  Feature extraction
02  Feature-band construction
03  Encoding models
04  ROI inference and noise ceilings
05  Diagnostic analyses
06  Permutation robustness controls
07  Surface-map generation
```

After step 07:

1. render the three regenerated Figure 3 surface screenshots in Workbench;
2. place them with the unchanged `ROI.png` under
   `figure_reproduction/static_inputs/figure3/`;
3. run `recreate_manuscript_outputs_from_rerun.py`.

This route generates the manuscript products entirely from the new analysis
outputs.

---

# Fast reviewer-facing route

A reviewer who only wants to reproduce the reported manuscript outputs does
**not** need to run steps 01–07.

The intended route is simply:

```bash
python figure_reproduction/recreate_manuscript_outputs.py
```

This consumes the frozen canonical snapshot under `historical_outputs/` plus
the static Figure 3 render inputs and recreates the manuscript-facing
quantitative figures and tables.

This separation keeps manuscript reproduction lightweight while retaining a
second, independent route for full computational rerunning of the analysis.

---



# Release principle

The two routes should converge on the same manuscript-facing results:

```text
frozen canonical outputs
        ↓
recreate_manuscript_outputs.py
        ↓
manuscript figures/tables

fresh analysis rerun (01–07)
        ↓
recreate_manuscript_outputs_from_rerun.py
        ↓
manuscript figures/tables
```

For the release, the frozen snapshot is refreshed from the validated full
rerun. This prevents reviewers from having to rerun the full imaging analysis
merely to reproduce the reported figures and tables, while still providing
the complete rerun pathway for users who want to verify the entire analysis
chain.
