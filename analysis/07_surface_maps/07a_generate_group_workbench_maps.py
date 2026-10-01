# -*- coding: utf-8 -*-
r"""
Regenerate group Workbench dscalar maps
======================================================

Regenerates the four descriptive group-mean CIFTI dscalar files used for
Workbench visualization:

1. wb_group_mean_r_joint_by_model.dscalar.nii
2. wb_group_mean_model_gains_delta_r.dscalar.nii
3. wb_group_m4_split_maps.dscalar.nii
4. wb_group_m4_split_and_fraction_maps.dscalar.nii

These maps are descriptive group means. They are not vertexwise inferential
maps. Formal inference remains ROI-level with subject as the inferential
unit.

Expected input folder structure
-------------------------------
MODEL_ROOT/
    M0_visual/sub-XXX/r_joint.npy
    M1_visual_rawcal/sub-XXX/r_joint.npy
    M2_visual_predclip/sub-XXX/r_joint.npy
    M3_visual_resclip/sub-XXX/r_joint.npy
    M4_visual_pred_resclip/sub-XXX/r_joint.npy
    M4_visual_pred_resclip/sub-XXX/r_split_LowVis.npy
    M4_visual_pred_resclip/sub-XXX/r_split_HighVis.npy
    M4_visual_pred_resclip/sub-XXX/r_split_CaloriePredCLIP.npy
    M4_visual_pred_resclip/sub-XXX/r_split_CalorieResCLIP.npy

Output
------
OUTDIR/workbench_dscalars/*.dscalar.nii
plus CSV/TXT audit files.

Notes
-----
- The script assumes full 91k grayordinate maps have cortical data in the first
  59412 entries, matching the convention used in your ROI scripts.
- If input maps are already cortical-only 59412, they are used as-is.
- The output CIFTI brain axis is taken from the HCP-MMP dlabel file, which is
  cortical fsLR-32k.

Author: generated for NeuroRepFood manuscript workflow.
"""

# =============================================================================
# [0] IMPORTS
# =============================================================================

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.cifti2 import Cifti2Image
from nibabel.cifti2.cifti2_axes import ScalarAxis

print("Imports OK.")


# =============================================================================
# [1] CONFIG
# =============================================================================

# RELEASE PATH ADAPTATION ONLY.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

MODEL_ROOT = os.path.join(REPO_ROOT, "reproduced_outputs", "encoding_models")

OUTDIR = os.path.join(REPO_ROOT, "reproduced_outputs", "surface_maps")
WBDIR = os.path.join(OUTDIR, "workbench_dscalars")
os.makedirs(WBDIR, exist_ok=True)

ATLAS_DIR = os.path.join(REPO_ROOT, "resources", "atlas")
HCP_DLABEL = os.path.join(
    ATLAS_DIR,
    "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors"
    ".32k_fs_LR.dlabel.nii",
)

N_GRAYORD = 91282
N_CORTICAL = 59412

# Optional explicit subject list. Leave as None to auto-discover sub-* folders.
SUBJECTS = None

# Optional exclusions. Useful if you know one subject should be dropped.
EXCLUDE_SUBJECTS = set([
    # "sub-141",
])

# Controls fraction maps r_split / r_joint. Very small denominators are set to NaN.
FRACTION_EPS = 1e-6

# If True, when computing fraction maps, negative split contributions are clipped
# to zero and normalized by positive total split contribution. This is NOT used
# by default because the old map names say "over_r_joint". Keep False to match
# the original map naming most closely.
USE_POSITIVE_NORMALIZED_FRACTIONS = False

# Models expected in MODEL_ROOT
MODELS = OrderedDict([
    ("M0_visual", "M0_visual"),
    ("M1_visual_rawcal", "M1_visual_rawcal"),
    ("M2_visual_predclip", "M2_visual_predclip"),
    ("M3_visual_resclip", "M3_visual_resclip"),
    ("M4_visual_pred_resclip", "M4_visual_pred_resclip"),
])

# Gain maps to export: output_name -> (model_A, model_B)
GAIN_COMPARISONS = OrderedDict([
    ("M1_rawcal_gt_M0_visual", ("M1_visual_rawcal", "M0_visual")),
    ("M2_predclip_gt_M0_visual", ("M2_visual_predclip", "M0_visual")),
    ("M3_resclip_gt_M0_visual", ("M3_visual_resclip", "M0_visual")),
    ("M4_pred_res_gt_M0_visual", ("M4_visual_pred_resclip", "M0_visual")),
    ("M2_predclip_gt_M3_resclip", ("M2_visual_predclip", "M3_visual_resclip")),
])

SPLIT_FILES = OrderedDict([
    ("LowVis", "r_split_LowVis.npy"),
    ("HighVis", "r_split_HighVis.npy"),
    ("CaloriePredCLIP", "r_split_CaloriePredCLIP.npy"),
    ("CalorieResCLIP", "r_split_CalorieResCLIP.npy"),
])


# =============================================================================
# [2] HELPERS
# =============================================================================

def model_dir(model_key):
    return os.path.join(MODEL_ROOT, MODELS[model_key])


def subject_sort_key(s):
    try:
        return int(str(s).replace("sub-", ""))
    except Exception:
        return str(s)


def discover_subjects():
    if SUBJECTS is not None:
        subs = [s if str(s).startswith("sub-") else f"sub-{s}" for s in SUBJECTS]
        return sorted(set(subs) - set(EXCLUDE_SUBJECTS), key=subject_sort_key)

    subject_sets = []
    for mk in MODELS:
        d = model_dir(mk)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing model directory for {mk}:\n  {d}")
        subs = {
            x for x in os.listdir(d)
            if x.startswith("sub-") and os.path.isdir(os.path.join(d, x))
        }
        subject_sets.append(subs)

    common = set.intersection(*subject_sets)
    common = common - set(EXCLUDE_SUBJECTS)
    return sorted(common, key=subject_sort_key)


def load_cortex_map(path):
    """Load a map and return cortical vector length 59412."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    arr = np.load(path).astype(np.float32).squeeze()

    if arr.ndim != 1:
        raise ValueError(f"Expected 1D map, got shape {arr.shape}:\n  {path}")

    if arr.shape[0] == N_CORTICAL:
        return arr.astype(np.float32)

    if arr.shape[0] == N_GRAYORD:
        return arr[:N_CORTICAL].astype(np.float32)

    if arr.shape[0] > N_CORTICAL:
        print(f"WARNING: map length {arr.shape[0]} not {N_GRAYORD}; using first {N_CORTICAL}: {path}")
        return arr[:N_CORTICAL].astype(np.float32)

    raise ValueError(
        f"Unexpected map length {arr.shape[0]}; expected {N_CORTICAL} or {N_GRAYORD}:\n  {path}"
    )


def load_r_joint(model_key, sub):
    return load_cortex_map(os.path.join(model_dir(model_key), sub, "r_joint.npy"))


def load_split(sub, split_key):
    return load_cortex_map(os.path.join(model_dir("M4_visual_pred_resclip"), sub, SPLIT_FILES[split_key]))


def finite_mean_stack(maps):
    """NaN-aware mean over list of cortical maps."""
    if len(maps) == 0:
        raise ValueError("No maps provided.")
    stack = np.stack(maps, axis=0).astype(np.float32)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stack, axis=0).astype(np.float32)


def safe_fraction(numer, denom):
    numer = np.asarray(numer, dtype=np.float32)
    denom = np.asarray(denom, dtype=np.float32)
    out = np.full_like(numer, np.nan, dtype=np.float32)
    ok = np.isfinite(numer) & np.isfinite(denom) & (np.abs(denom) > FRACTION_EPS)
    out[ok] = numer[ok] / denom[ok]
    return out


def compute_fraction_maps_for_subject(split_maps, r_joint_m4):
    """Return PredCLIP, ResCLIP, Visual fraction maps for one subject."""
    low = split_maps["LowVis"]
    high = split_maps["HighVis"]
    pred = split_maps["CaloriePredCLIP"]
    res = split_maps["CalorieResCLIP"]
    visual = low + high

    if USE_POSITIVE_NORMALIZED_FRACTIONS:
        low_p = np.maximum(low, 0)
        high_p = np.maximum(high, 0)
        pred_p = np.maximum(pred, 0)
        res_p = np.maximum(res, 0)
        total_p = low_p + high_p + pred_p + res_p
        return {
            "PredCLIP": safe_fraction(pred_p, total_p),
            "ResCLIP": safe_fraction(res_p, total_p),
            "Visual": safe_fraction(low_p + high_p, total_p),
        }

    return {
        "PredCLIP": safe_fraction(pred, r_joint_m4),
        "ResCLIP": safe_fraction(res, r_joint_m4),
        "Visual": safe_fraction(visual, r_joint_m4),
    }


def load_brain_axis_from_dlabel(dlabel_path):
    if not os.path.isfile(dlabel_path):
        raise FileNotFoundError(f"HCP dlabel not found:\n  {dlabel_path}")
    img = nib.load(dlabel_path)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D CIFTI dlabel/dscalar, got ndim={img.ndim}: {dlabel_path}")
    brain_axis = img.header.get_axis(1)
    if brain_axis.size != N_CORTICAL:
        print(f"WARNING: brain axis size is {brain_axis.size}, expected {N_CORTICAL}")
    return brain_axis


def save_dscalar(data, names, out_path, brain_axis):
    """Save data as CIFTI dscalar. data shape = (n_maps, n_cortex)."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data, got {data.shape}")
    if data.shape[0] != len(names):
        raise ValueError(f"Data rows {data.shape[0]} != names {len(names)}")
    if data.shape[1] != brain_axis.size:
        raise ValueError(f"Data columns {data.shape[1]} != brain axis size {brain_axis.size}")

    scalar_axis = ScalarAxis(names)
    img = Cifti2Image(dataobj=data, header=(scalar_axis, brain_axis))
    img.nifti_header.set_intent("NIFTI_INTENT_CONNECTIVITY_DENSE_SCALARS")
    nib.save(img, out_path)
    print(f"  saved: {out_path}")


def write_map_names(out_path, names):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, n in enumerate(names):
            f.write(f"{i:02d}\t{n}\n")


def summarize_maps(names, data, out_csv):
    rows = []
    for name, arr in zip(names, data):
        arr = np.asarray(arr, dtype=np.float64)
        finite = np.isfinite(arr)
        rows.append({
            "map": name,
            "n_finite": int(finite.sum()),
            "mean": float(np.nanmean(arr)) if finite.any() else np.nan,
            "std": float(np.nanstd(arr)) if finite.any() else np.nan,
            "min": float(np.nanmin(arr)) if finite.any() else np.nan,
            "max": float(np.nanmax(arr)) if finite.any() else np.nan,
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def require_subject_inputs(sub):
    """Check all required files for one subject."""
    missing = []

    for mk in MODELS:
        p = os.path.join(model_dir(mk), sub, "r_joint.npy")
        if not os.path.isfile(p):
            missing.append(p)

    for split_key, fname in SPLIT_FILES.items():
        p = os.path.join(model_dir("M4_visual_pred_resclip"), sub, fname)
        if not os.path.isfile(p):
            missing.append(p)

    return missing


# =============================================================================
# [3] MAIN EXPORT
# =============================================================================

def main():
    print("\nResolving subjects...")
    subjects_all = discover_subjects()
    print(f"  discovered common model subjects: {len(subjects_all)}")
    print("  " + ", ".join(subjects_all))

    included = []
    audit_rows = []
    for sub in subjects_all:
        missing = require_subject_inputs(sub)
        ok = len(missing) == 0
        audit_rows.append({
            "subject": sub,
            "included": ok,
            "n_missing_files": len(missing),
            "missing_files": " | ".join(missing),
        })
        if ok:
            included.append(sub)

    if len(included) == 0:
        raise RuntimeError("No subjects with complete inputs.")

    audit_csv = os.path.join(WBDIR, "workbench_group_map_subject_inclusion.csv")
    pd.DataFrame(audit_rows).to_csv(audit_csv, index=False)

    print(f"\nIncluded subjects: {len(included)}")
    print("  " + ", ".join(included))
    print(f"Subject audit: {audit_csv}")

    print("\nLoading CIFTI brain axis...")
    brain_axis = load_brain_axis_from_dlabel(HCP_DLABEL)
    print(f"  brain axis size: {brain_axis.size}")

    # -------------------------------------------------------------------------
    # 1. Group mean r_joint by model
    # -------------------------------------------------------------------------
    print("\n[1] Exporting group mean r_joint by model")

    rjoint_names = []
    rjoint_maps = []

    for mk in MODELS:
        maps = [load_r_joint(mk, sub) for sub in included]
        mean_map = finite_mean_stack(maps)
        name = f"mean_r_joint__{mk}"
        rjoint_names.append(name)
        rjoint_maps.append(mean_map)
        print(f"  {mk:28s}: mean finite={np.nanmean(mean_map):+.5f}")

    rjoint_data = np.stack(rjoint_maps, axis=0)
    out_rjoint = os.path.join(WBDIR, "wb_group_mean_r_joint_by_model.dscalar.nii")
    save_dscalar(rjoint_data, rjoint_names, out_rjoint, brain_axis)
    write_map_names(os.path.join(WBDIR, "wb_group_mean_r_joint_by_model_map_names.txt"), rjoint_names)
    summarize_maps(rjoint_names, rjoint_data, os.path.join(WBDIR, "wb_group_mean_r_joint_by_model_summary.csv"))

    # -------------------------------------------------------------------------
    # 2. Group mean model gains delta r
    # -------------------------------------------------------------------------
    print("\n[2] Exporting group mean model gains delta r")

    gain_names = []
    gain_maps = []

    for cname, (ma, mb) in GAIN_COMPARISONS.items():
        maps = []
        for sub in included:
            a = load_r_joint(ma, sub)
            b = load_r_joint(mb, sub)
            maps.append((a - b).astype(np.float32))
        mean_map = finite_mean_stack(maps)
        name = f"mean_delta_r__{cname}"
        gain_names.append(name)
        gain_maps.append(mean_map)
        print(f"  {cname:32s}: mean finite={np.nanmean(mean_map):+.5f}")

    gain_data = np.stack(gain_maps, axis=0)
    out_gain = os.path.join(WBDIR, "wb_group_mean_model_gains_delta_r.dscalar.nii")
    save_dscalar(gain_data, gain_names, out_gain, brain_axis)
    write_map_names(os.path.join(WBDIR, "wb_group_mean_model_gains_delta_r_map_names.txt"), gain_names)
    summarize_maps(gain_names, gain_data, os.path.join(WBDIR, "wb_group_mean_model_gains_delta_r_summary.csv"))

    # -------------------------------------------------------------------------
    # 3. M4 split maps and fractions
    # -------------------------------------------------------------------------
    print("\n[3] Exporting M4 split maps")

    # Load all subject split maps once.
    split_by_sub = OrderedDict()
    rjoint_m4_by_sub = OrderedDict()

    for sub in included:
        split_by_sub[sub] = OrderedDict()
        for split_key in SPLIT_FILES:
            split_by_sub[sub][split_key] = load_split(sub, split_key)
        rjoint_m4_by_sub[sub] = load_r_joint("M4_visual_pred_resclip", sub)

    mean_low = finite_mean_stack([split_by_sub[sub]["LowVis"] for sub in included])
    mean_high = finite_mean_stack([split_by_sub[sub]["HighVis"] for sub in included])
    mean_pred = finite_mean_stack([split_by_sub[sub]["CaloriePredCLIP"] for sub in included])
    mean_res = finite_mean_stack([split_by_sub[sub]["CalorieResCLIP"] for sub in included])

    # Subject-level derived maps before averaging.
    visual_maps = []
    calorie_total_maps = []
    delta_pred_res_maps = []
    frac_pred_maps = []
    frac_res_maps = []
    frac_visual_maps = []

    for sub in included:
        low = split_by_sub[sub]["LowVis"]
        high = split_by_sub[sub]["HighVis"]
        pred = split_by_sub[sub]["CaloriePredCLIP"]
        res = split_by_sub[sub]["CalorieResCLIP"]
        rj = rjoint_m4_by_sub[sub]

        visual_maps.append((low + high).astype(np.float32))
        calorie_total_maps.append((pred + res).astype(np.float32))
        delta_pred_res_maps.append((pred - res).astype(np.float32))

        fracs = compute_fraction_maps_for_subject(split_by_sub[sub], rj)
        frac_pred_maps.append(fracs["PredCLIP"])
        frac_res_maps.append(fracs["ResCLIP"])
        frac_visual_maps.append(fracs["Visual"])

    mean_visual = finite_mean_stack(visual_maps)
    mean_calorie_total = finite_mean_stack(calorie_total_maps)
    mean_delta_pred_res = finite_mean_stack(delta_pred_res_maps)
    mean_frac_pred = finite_mean_stack(frac_pred_maps)
    mean_frac_res = finite_mean_stack(frac_res_maps)
    mean_frac_visual = finite_mean_stack(frac_visual_maps)

    # Original-style split maps file: 9 maps.
    split_names = [
        "mean_r_split__LowVis",
        "mean_r_split__HighVis",
        "mean_r_split__CaloriePredCLIP",
        "mean_r_split__CalorieResCLIP",
        "mean_r_split__Visual",
        "mean_r_split__CalorieTotal",
        "mean_delta_r_split__PredCLIP_minus_ResCLIP",
        "fraction_M4__PredCLIP_over_r_joint",
        "fraction_M4__ResCLIP_over_r_joint",
    ]
    split_data = np.stack([
        mean_low,
        mean_high,
        mean_pred,
        mean_res,
        mean_visual,
        mean_calorie_total,
        mean_delta_pred_res,
        mean_frac_pred,
        mean_frac_res,
    ], axis=0)

    out_split = os.path.join(WBDIR, "wb_group_m4_split_maps.dscalar.nii")
    save_dscalar(split_data, split_names, out_split, brain_axis)
    write_map_names(os.path.join(WBDIR, "wb_group_m4_split_maps_map_names.txt"), split_names)
    summarize_maps(split_names, split_data, os.path.join(WBDIR, "wb_group_m4_split_maps_summary.csv"))

    # Split + fraction maps file: 10 maps, matching the uploaded map names.
    split_frac_names = [
        "mean_r_split__LowVis",
        "mean_r_split__HighVis",
        "mean_r_split__CaloriePredCLIP",
        "mean_r_split__CalorieResCLIP",
        "mean_r_split__Visual_LowVis_plus_HighVis",
        "mean_r_split__Calorie_Pred_plus_Res",
        "mean_delta_r_split__PredCLIP_minus_ResCLIP",
        "fraction_M4__PredCLIP_over_r_joint",
        "fraction_M4__ResCLIP_over_r_joint",
        "fraction_M4__Visual_over_r_joint",
    ]
    split_frac_data = np.stack([
        mean_low,
        mean_high,
        mean_pred,
        mean_res,
        mean_visual,
        mean_calorie_total,
        mean_delta_pred_res,
        mean_frac_pred,
        mean_frac_res,
        mean_frac_visual,
    ], axis=0)

    out_split_frac = os.path.join(WBDIR, "wb_group_m4_split_and_fraction_maps.dscalar.nii")
    save_dscalar(split_frac_data, split_frac_names, out_split_frac, brain_axis)
    write_map_names(os.path.join(WBDIR, "wb_group_m4_split_and_fraction_maps_map_names.txt"), split_frac_names)
    summarize_maps(split_frac_names, split_frac_data, os.path.join(WBDIR, "wb_group_m4_split_and_fraction_maps_summary.csv"))

    # -------------------------------------------------------------------------
    # Notes file
    # -------------------------------------------------------------------------
    notes_path = os.path.join(WBDIR, "workbench_map_notes.txt")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("Workbench map notes\n")
        f.write("================================\n\n")
        f.write("These maps are descriptive group-mean maps, not vertexwise inferential maps.\n")
        f.write("Formal inference remains ROI-level with subject as inferential unit.\n\n")
        f.write("Recommended main Workbench panels:\n")
        f.write("1. wb_group_mean_r_joint_by_model.dscalar.nii\n")
        f.write("   - mean_r_joint__M0_visual\n")
        f.write("2. wb_group_mean_model_gains_delta_r.dscalar.nii\n")
        f.write("   - mean_delta_r__M2_predclip_gt_M0_visual\n")
        f.write("   - mean_delta_r__M3_resclip_gt_M0_visual\n")
        f.write("   - mean_delta_r__M2_predclip_gt_M3_resclip\n")
        f.write("3. wb_group_m4_split_maps.dscalar.nii\n")
        f.write("   - mean_r_split__CaloriePredCLIP\n")
        f.write("   - mean_r_split__CalorieResCLIP\n")
        f.write("   - mean_delta_r_split__PredCLIP_minus_ResCLIP\n\n")
        f.write("Display suggestions:\n")
        f.write("- Use a positive sequential palette for absolute r_joint maps.\n")
        f.write("- Use a diverging palette centered at zero for gain/split-difference maps.\n")
        f.write("- Keep the same color scale across related gain maps.\n\n")
        f.write(f"Regenerated from {len(included)} subjects.\n")
        f.write(f"USE_POSITIVE_NORMALIZED_FRACTIONS = {USE_POSITIVE_NORMALIZED_FRACTIONS}\n")

    print(f"\nNotes saved: {notes_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
