# 07 — Surface-map generation

This stage regenerates the descriptive Workbench CIFTI maps from the canonical
25-participant encoding outputs.

It does **not** perform vertexwise statistical inference. Formal inference
remains the participant-level ROI analysis from `04_roi_inference`.

## Historical provenance

`07a_generate_group_workbench_maps.py` is a path-adapted version of:

```text
regenerate_workbench_group_maps_neurorepfood.py
```

Historical SHA-256:

```text
bad6e0132ea7cc71330c9994653be4fc8a5ea32b212b53048cbb3815d7587df4
```

`07b` and `07c` use the same historical implementation from:

```text
mask_workbench_maps.py
```

Historical SHA-256:

```text
ff26c8facee4841836234d372fbd8f53a01fc3bdbadce747665b336c2222ed38
```

The historical mask script was run for both the group `r_joint` dscalar and the
group model-gain dscalar. Two explicit release scripts are provided so both
historically produced outputs can be recreated without manually editing a path.

No map averaging, masking, fraction, or CIFTI-writing formula was changed.

## Inputs

```text
reproduced_outputs/encoding_models/
    M0_visual/
    M1_visual_rawcal/
    M2_visual_predclip/
    M3_visual_resclip/
    M4_visual_pred_resclip/

resources/atlas/
    Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii
```

All five model directories must contain the complete subject-level map outputs.
The script discovers the intersection of `sub-*` folders and verifies required
`r_joint` and M4 split maps before inclusion.

## Outputs

```text
reproduced_outputs/surface_maps/workbench_dscalars/
    wb_group_mean_r_joint_by_model.dscalar.nii
    wb_group_mean_r_joint_by_model_map_names.txt
    wb_group_mean_r_joint_by_model_summary.csv

    wb_group_mean_model_gains_delta_r.dscalar.nii
    wb_group_mean_model_gains_delta_r_map_names.txt
    wb_group_mean_model_gains_delta_r_summary.csv

    wb_group_m4_split_maps.dscalar.nii
    wb_group_m4_split_maps_map_names.txt
    wb_group_m4_split_maps_summary.csv

    wb_group_m4_split_and_fraction_maps.dscalar.nii
    wb_group_m4_split_and_fraction_maps_map_names.txt
    wb_group_m4_split_and_fraction_maps_summary.csv

    wb_group_mean_r_joint_by_model__HighLevelVTCmasked.dscalar.nii
    wb_group_mean_r_joint_by_model__HighLevelVTCmasked_notes.txt

    wb_group_mean_model_gains_delta_r__HighLevelVTCmasked.dscalar.nii
    wb_group_mean_model_gains_delta_r__HighLevelVTCmasked_notes.txt

    workbench_group_map_subject_inclusion.csv
    workbench_map_notes.txt
```

### Legacy masked filenames

The two historical files ending in `__HighLevelVTCmasked.dscalar.nii` have a
legacy filename. The historically executed configuration actually applies the
**broad visual-cortex display mask**, not the inferential HighLevelVTC ROI.

The broad display mask includes:

```text
V1 V2 V3 V4 V3A V3B V3CD V6 V6A V7 V8
LO1 LO2 LO3 PIT MT MST FST
FFC VVC VMV1 VMV2 VMV3 TE1a TE1m TE1p TE2p
PHA1 PHA2 PHA3 PH
```

Thus TE1a/TE1m occur only in the **display mask**. The formal HighLevelVTC ROI
used for statistical inference remains:

```text
FFC + VVC + TE1p + TE2p
```

## Figure 3 source maps

The manuscript Figure 3 panels use descriptive maps:

- Panel B: `mean_r_joint__M0_visual`
- Panel C: `mean_r_joint__M2_visual_predclip`
- Panel D: `mean_delta_r__M2_predclip_gt_M0_visual`

These are stored in the first two group dscalar files above. Workbench
rendering/screenshot capture remains a manual visualization step, as in the
historical workflow.

## Run

From the repository root:

```bash
python analysis/07_surface_maps/run_surface_maps.py
```

## Release script hashes

- `07a_generate_group_workbench_maps.py`: `4175dc76eea7baf173cd584f6f806cd1564f62796698f8db08dbc7f5c334a775`
- `07b_apply_broad_visual_display_mask_rjoint.py`: `d164a15f765b715d318636f504b02c685fb15d0b906a3905ddd1f8be9d909f83`
- `07c_apply_broad_visual_display_mask_gains.py`: `a4e7e8c40c9c403214eefcfd85b35f2cb210c1b50b029fd79b7d97ce6960aca7`
- `run_surface_maps.py`: `f951e89a49e9feb31577df174ff6b01d0e54d64784586a0dd1afe78cdda7e820`
