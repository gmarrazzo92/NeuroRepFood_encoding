# -*- coding: utf-8 -*-
"""
Permutation null test with fixed deltas, resumable final-ROI version
=================================================================================

Purpose
-------
Runs two fixed-delta permutation controls for the nested calorie encoding model:

1. FULL CONDITION SHUFFLE
   Permutes the 96 condition labels before building the beta matrix.
   This breaks the image-response correspondence and tests whether the
   modeling/cross-validation procedure produces positive r_joint under the null.

2. CALORIE-LABEL SHUFFLE
   Keeps visual features and neural betas aligned, but permutes the PredCLIP
   feature rows. This tests whether the M2-M0 gain depends on the true ordering
   of images along the PredCLIP axis, rather than on the generic benefit of
   adding an extra one-dimensional predictor.

Main changes relative to the earlier script
-------------------------------------------
1. The high-level ROI is now HighLevelVTC, matching the main analysis:
       HighLevelVTC = FFC + VVC + TE1p + TE2p
   The old FoodVTC definition is not used.

2. The script is resumable. It reads permutation_null_results.csv if present and
   runs only subjects that do not already have complete rows for the CURRENT ROI
   definitions and N_SHUFFLES.

   Important: old rows labelled FoodVTC do not count as HighLevelVTC. If the
   existing CSV only contains the old FoodVTC ROI, subjects will be considered
   incomplete for the current final-ROI analysis and will be rerun. This avoids
   mixing incompatible ROI definitions.

3. The calorie-label shuffle no longer recomputes M0 for every shuffle, because
   M0 is unchanged when only PredCLIP is permuted. It reuses the true M0 map for
   the shuffled delta, which is mathematically equivalent and substantially faster.

Outputs
-------
OUTDIR/permutation_null_results.csv
OUTDIR/permutation_null_summary.csv
OUTDIR/calorie_specificity.csv
OUTDIR/calorie_specificity_summary.csv
OUTDIR/permutation_null_metadata.json
OUTDIR/figures/null_full_shuffle.png
OUTDIR/figures/calorie_specificity_per_subject.png
OUTDIR/figures/true_vs_shuffle_scatter.png
"""

# =============================================================================
# [0] IMPORTS
# =============================================================================

import os
import json
import shutil
from datetime import datetime
from collections import OrderedDict
from itertools import product as itertools_product

import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests

print("Imports OK.")

# =============================================================================
# [1] CONFIG
# =============================================================================

# RELEASE PATH ADAPTATION ONLY.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

FEAT_DIR = os.path.join(REPO_ROOT, "reproduced_outputs", "feature_extraction")
BAND_DIR = os.path.join(REPO_ROOT, "reproduced_outputs", "feature_bands")
BETA_DIR = os.path.join(REPO_ROOT, "data", "glmsingle")

MODEL_ROOT = os.path.join(REPO_ROOT, "reproduced_outputs", "encoding_models")
M0_DIR     = os.path.join(MODEL_ROOT, "M0_visual")
M2_DIR     = os.path.join(MODEL_ROOT, "M2_visual_predclip")

ATLAS_DIR  = os.path.join(REPO_ROOT, "resources", "atlas")
HCP_DLABEL = os.path.join(
    ATLAS_DIR,
    "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors"
    ".32k_fs_LR.dlabel.nii",
)

OUTDIR = os.path.join(
    REPO_ROOT, "reproduced_outputs", "robustness", "permutation_null"
)
os.makedirs(OUTDIR, exist_ok=True)
FIGDIR = os.path.join(OUTDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

RESULTS_CSV = os.path.join(OUTDIR, "permutation_null_results.csv")
SUMMARY_CSV = os.path.join(OUTDIR, "permutation_null_summary.csv")
SPEC_CSV    = os.path.join(OUTDIR, "calorie_specificity.csv")
SPEC_SUMMARY_CSV = os.path.join(OUTDIR, "calorie_specificity_summary.csv")
METADATA_JSON = os.path.join(OUTDIR, "permutation_null_metadata.json")

ALL_SUBJECTS = [
    104, 105, 109, 112, 115, 117, 118, 122, 124, 125, 126,
    129, 132, 134, 137, 139, 140, 141, 143, 144, 145, 146, 149,
    150, 151,
]

N_SHUFFLES    = 100
N_CONDITIONS  = 96
N_OUTER_FOLDS = 5
N_GRAYORD     = 91282
N_CORTICAL    = 59412
MIN_VERTICES  = 25
RANDOM_STATE  = 42
ALPHA_FDR     = 0.05

# Scale-relative regularization — identical for true and all shuffles.
# Sensitivity check: rerun with REGULARIZATION_FACTOR = 10 and 1000.
REGULARIZATION_FACTOR = 100.0

# Resume behavior.
RUN_ONLY_MISSING_SUBJECTS = True
BACKUP_EXISTING_RESULTS_BEFORE_FIRST_WRITE = True
WRITE_AFTER_EACH_SUBJECT = True

# Final ROI definition matching the main manuscript.
ROI_GROUPS = OrderedDict([
    ("EarlyVisual",        ["V1", "V2", "V3", "V4"]),
    ("IntermediateVisual", ["V8", "PIT", "LO1", "LO2", "LO3"]),
    ("HighLevelVTC",       ["FFC", "VVC", "TE1p", "TE2p"]),
])

# =============================================================================
# [2] LOAD FEATURES
# =============================================================================

def load_band(name):
    p = os.path.join(BAND_DIR, f"band_{name}.npy")
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    return np.load(p).astype(np.float32)

LowVis  = load_band("LowVis")   # expected: (96, 80)
HighVis = load_band("HighVis")  # expected: (96, 80)

# Preserve the historical lookup order. The first candidate was supported by
# the original script; the historically used fallback is generated by 05a.
PREDCLIP_PATH = os.path.join(FEAT_DIR, "CaloriePredCLIP_loso_group.npy")
if not os.path.isfile(PREDCLIP_PATH):
    PREDCLIP_PATH = os.path.join(
        REPO_ROOT,
        "reproduced_outputs",
        "diagnostics",
        "perceived_calorie_prediction_diagnostics",
        "calorie_pred_cv_CLIP.npy",
    )
if not os.path.isfile(PREDCLIP_PATH):
    raise FileNotFoundError(
        "PredCLIP file not found. Checked:\n"
        f"  {os.path.join(FEAT_DIR, 'CaloriePredCLIP_loso_group.npy')}\n"
        f"  {os.path.join(REPO_ROOT, 'reproduced_outputs', 'diagnostics', 'perceived_calorie_prediction_diagnostics', 'calorie_pred_cv_CLIP.npy')}"
    )

PredCLIP = np.load(PREDCLIP_PATH).astype(np.float32)
if PredCLIP.ndim == 1:
    PredCLIP = PredCLIP.reshape(-1, 1)

assert LowVis.shape[0]   == N_CONDITIONS
assert HighVis.shape[0]  == N_CONDITIONS
assert PredCLIP.shape[0] == N_CONDITIONS

print(f"LowVis  : {LowVis.shape}")
print(f"HighVis : {HighVis.shape}")
print(f"PredCLIP: {PredCLIP.shape}")
print(f"PredCLIP source: {PREDCLIP_PATH}")

# =============================================================================
# [3] LOAD ATLAS / ROI MASKS
# =============================================================================

def load_roi_masks(dlabel_path):
    dlabel    = nib.load(dlabel_path)
    data      = dlabel.get_fdata(dtype=np.float32).squeeze().reshape(-1)
    label_map = data[:N_CORTICAL]

    label_axis = dlabel.header.get_axis(0)
    label_dict = label_axis.label[0]

    name_to_values = {}
    for val, (full_name, _rgba) in label_dict.items():
        val = int(val)
        if val == 0:
            continue

        base = full_name.replace("_ROI", "")

        for pfx in ["L_", "R_"]:
            if base.startswith(pfx):
                base = base[len(pfx):]

        for sfx in ["_L", "_R"]:
            if base.endswith(sfx):
                base = base[:-len(sfx)]

        name_to_values.setdefault(base, []).append(val)

    roi_masks = OrderedDict()
    for roi_name, parcels in ROI_GROUPS.items():
        mask = np.zeros(N_CORTICAL, dtype=bool)
        missing = []
        for parcel in parcels:
            vals = name_to_values.get(parcel, [])
            if len(vals) == 0:
                missing.append(parcel)
            mask |= np.isin(label_map, vals)

        if missing:
            print(f"  WARNING {roi_name}: missing parcels in atlas: {missing}")

        roi_masks[roi_name] = mask
        print(f"  {roi_name:22s}: {int(mask.sum()):5d} vertices | {parcels}")

    return roi_masks

print("\nLoading ROI masks...")
roi_masks = load_roi_masks(HCP_DLABEL)

# =============================================================================
# [4] GENERAL HELPERS
# =============================================================================

def subject_str(sid):
    if isinstance(sid, str) and sid.startswith("sub-"):
        return sid
    return f"sub-{sid}"


def normalize_stimorder_zero_based(stimorder):
    stimorder = np.asarray(stimorder).astype(int)
    mn = int(np.nanmin(stimorder))
    mx = int(np.nanmax(stimorder))
    if mn == 0 and mx == N_CONDITIONS - 1:
        return stimorder
    if mn == 1 and mx == N_CONDITIONS:
        print("  WARNING: stimorder appears 1-based; converting to 0-based.")
        return stimorder - 1
    raise ValueError(
        f"Unexpected stimorder range: min={mn}, max={mx}. "
        f"Expected 0..{N_CONDITIONS - 1} or 1..{N_CONDITIONS}."
    )


def pearson_r_columns(A, B):
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    num   = np.sum(A * B, axis=0)
    denom = np.sqrt(np.sum(A**2, axis=0) * np.sum(B**2, axis=0))
    r     = np.full(A.shape[1], np.nan, dtype=np.float32)
    ok    = denom > 1e-10
    r[ok] = (num[ok] / denom[ok]).astype(np.float32)
    return r


def center_normalize_kernel(K_tr, K_te=None):
    n  = K_tr.shape[0]
    H  = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / n
    Kc = H @ K_tr.astype(np.float64) @ H
    fro = float(np.linalg.norm(Kc, "fro"))
    if fro < 1e-10:
        fro = 1.0
    Kc = (Kc / fro).astype(np.float32)

    if K_te is not None:
        # Test-set centering is performed relative to the train-set column space.
        # This matches the original analysis implementation.
        n_te = K_te.shape[0]
        H_te = np.eye(n_te, dtype=np.float64) - np.ones((n_te, n_te), dtype=np.float64) / n_te
        Kc_te = (H_te @ K_te.astype(np.float64) @ H / fro).astype(np.float32)
        return Kc, Kc_te

    return Kc


def deltas_to_band_weights(deltas_folds, fold_idx):
    """Convert fold deltas to average softmax band weights."""
    d = np.nanmean(deltas_folds[fold_idx], axis=-1)
    exp_d = np.exp(d - np.max(d))
    return (exp_d / exp_d.sum()).astype(np.float32)


def build_combined_kernel(bands_tr, bands_te, weights):
    n_tr = bands_tr[0].shape[0]
    n_te = bands_te[0].shape[0]
    K_tr = np.zeros((n_tr, n_tr), dtype=np.float32)
    K_te = np.zeros((n_te, n_tr), dtype=np.float32)

    for w, Xtr, Xte in zip(weights, bands_tr, bands_te):
        if w < 1e-10:
            continue
        Kk_tr = Xtr @ Xtr.T
        Kk_te = Xte @ Xtr.T
        Kc_tr, Kc_te = center_normalize_kernel(Kk_tr, Kk_te)
        K_tr += w * Kc_tr
        K_te += w * Kc_te

    return K_tr, K_te


def scale_relative_alpha(K_tr, factor=REGULARIZATION_FACTOR):
    return float(factor * np.mean(np.diag(K_tr)))


def solve_and_predict(K_tr, K_te, Y_tr, alpha):
    n = K_tr.shape[0]
    dual = np.linalg.solve(
        K_tr.astype(np.float64) + alpha * np.eye(n, dtype=np.float64),
        Y_tr.astype(np.float64),
    )
    return (K_te.astype(np.float64) @ dual).astype(np.float32)


def expand_to_cortex(compact_vals, compact_mask):
    full = np.full(N_CORTICAL, np.nan, dtype=np.float32)
    full[compact_mask] = compact_vals
    return full


def roi_mean(compact_vals, compact_mask, roi_mask):
    cortex = expand_to_cortex(compact_vals, compact_mask)
    combined = roi_mask & compact_mask & np.isfinite(cortex)
    if int(combined.sum()) < MIN_VERTICES:
        return np.nan
    return float(np.nanmean(cortex[combined]))


def signflip_1d(values, alternative="greater", seed=RANDOM_STATE):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    n = vals.size
    if n == 0:
        return np.nan

    obs = float(np.mean(vals))

    if n <= 20:
        exceed = 0
        for signs_tuple in itertools_product([-1.0, 1.0], repeat=n):
            signs = np.asarray(signs_tuple, dtype=np.float64)
            null = float(np.mean(signs * vals))
            if alternative == "greater":
                exceed += null >= obs
            elif alternative == "less":
                exceed += null <= obs
            elif alternative == "two-sided":
                exceed += abs(null) >= abs(obs)
            else:
                raise ValueError("alternative must be greater, less, or two-sided")
        return max(exceed / (2 ** n), 1.0 / (2 ** n))

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(50000, n))
    null = np.mean(signs * vals[None, :], axis=1)
    if alternative == "greater":
        exceed = int(np.sum(null >= obs))
    elif alternative == "less":
        exceed = int(np.sum(null <= obs))
    elif alternative == "two-sided":
        exceed = int(np.sum(np.abs(null) >= abs(obs)))
    else:
        raise ValueError("alternative must be greater, less, or two-sided")
    return float((exceed + 1) / (50000 + 1))


def ci95(values):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    if vals.size == 1:
        return float(vals[0]), float(vals[0])
    mean = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
    tcrit = stats.t.ppf(0.975, df=vals.size - 1)
    return mean - tcrit * sem, mean + tcrit * sem


def cohens_dz(values):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.nan
    sd = float(np.std(vals, ddof=1))
    if sd < 1e-12:
        return np.nan
    return float(np.mean(vals) / sd)

# =============================================================================
# [5] ENCODING MODEL — FIXED DELTAS, SCALE-RELATIVE ALPHA
# =============================================================================

def run_fixed_delta_model(
    betas_compact,
    feature_bands,
    deltas_folds,
    outer_splits,
    condition_perm=None,
    predclip_perm=None,
):
    """
    condition_perm : (96,) or None
        Full condition shuffle. Permutes beta rows and destroys image-response alignment.

    predclip_perm : (96,) or None
        Calorie-label shuffle. Permutes the final feature band only. The final band
        must be PredCLIP when this argument is used.
    """
    assert not (condition_perm is not None and predclip_perm is not None)

    if condition_perm is not None:
        betas_use = betas_compact[condition_perm, :]
    else:
        betas_use = betas_compact

    if predclip_perm is not None:
        feature_bands = list(feature_bands)
        feature_bands[-1] = feature_bands[-1][predclip_perm, :]

    r_folds = []

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_splits):
        Y_tr = betas_use[tr_idx].astype(np.float32)
        Y_te = betas_use[te_idx].astype(np.float32)

        weights = deltas_to_band_weights(deltas_folds, fold_idx)

        bands_tr_fit = []
        bands_te_fit = []
        for X in feature_bands:
            Xtr = X[tr_idx].astype(np.float64)
            Xte = X[te_idx].astype(np.float64)
            m = Xtr.mean(axis=0, keepdims=True)
            s = Xtr.std(axis=0, keepdims=True) + 1e-8
            bands_tr_fit.append(((Xtr - m) / s).astype(np.float32))
            bands_te_fit.append(((Xte - m) / s).astype(np.float32))

        K_tr, K_te = build_combined_kernel(bands_tr_fit, bands_te_fit, weights)
        alpha = scale_relative_alpha(K_tr)
        Y_pred = solve_and_predict(K_tr, K_te, Y_tr, alpha)
        r_folds.append(pearson_r_columns(Y_te, Y_pred))

    return np.nanmean(np.stack(r_folds, axis=0), axis=0).astype(np.float32)

# =============================================================================
# [6] LOAD SUBJECT DATA
# =============================================================================

def _map_compact_columns_from_source(betas_trials, source_indices, desired_indices, sub, source_label):
    """
    Select desired grayordinates from an already-compressed beta matrix.

    Parameters
    ----------
    betas_trials : array, shape (n_trials, n_source_columns)
        Trial beta matrix whose columns correspond to source_indices in order.
    source_indices : array of int
        Full/cortical grayordinate indices represented by the columns of betas_trials.
    desired_indices : array of int
        Indices requested by the model combined_mask.
    """
    source_indices = np.asarray(source_indices, dtype=np.int64)
    desired_indices = np.asarray(desired_indices, dtype=np.int64)

    if betas_trials.shape[1] != source_indices.size:
        raise ValueError(
            f"{sub}: beta columns ({betas_trials.shape[1]}) do not match "
            f"{source_label} source index count ({source_indices.size})."
        )

    pos = np.searchsorted(source_indices, desired_indices)
    ok = (pos < source_indices.size) & (source_indices[pos] == desired_indices)

    if not np.all(ok):
        missing = desired_indices[~ok][:10].tolist()
        raise ValueError(
            f"{sub}: {np.sum(~ok)} compact-mask vertices are not present in "
            f"{source_label}. First missing indices: {missing}"
        )

    return betas_trials[:, pos]


def select_compact_betas(betas_trials, compact_mask_full, compact_mask_cortical, n_compact, sub):
    """
    Return trial betas restricted to the model compact cortical mask.

    This handles all output formats encountered in the project:
      1. Full 91k grayordinate matrix:                 (trials, 91282)
      2. Full cortical matrix only:                    (trials, 59412)
      3. Already restricted to model combined_mask:     (trials, n_compact)
      4. GLMsingle good-grayordinate compact matrix:    (trials, sum(good_grayordinates_mask))
         This is the case that produced e.g. (1344, 90646) for sub-117.
    """
    n_cols = int(betas_trials.shape[1])
    desired_cortical = np.flatnonzero(compact_mask_cortical)

    # Case 1: full 91k grayordinates.
    if n_cols == N_GRAYORD:
        return betas_trials[:, :N_CORTICAL][:, compact_mask_cortical]

    # Case 2: full cortical surface only.
    if n_cols == N_CORTICAL:
        return betas_trials[:, compact_mask_cortical]

    # Case 3: already in the exact model compact space.
    if n_cols == n_compact:
        return betas_trials

    # Case 4: GLMsingle output compacted by good_grayordinates_mask.npy.
    good_mask_path = os.path.join(BETA_DIR, sub, "good_grayordinates_mask.npy")
    if not os.path.isfile(good_mask_path):
        raise ValueError(
            f"Unexpected betas_trials shape for {sub}: {betas_trials.shape}. "
            f"Expected columns {N_GRAYORD}, {N_CORTICAL}, or {n_compact}; "
            f"could not map compressed columns because good_grayordinates_mask.npy was not found."
        )

    good_mask = np.load(good_mask_path).astype(bool)

    if good_mask.shape[0] == N_GRAYORD:
        good_full = good_mask
        good_cortical = good_full[:N_CORTICAL]

        # 4a. Columns correspond to all good grayordinates in 91k order
        #     (cortex + subcortex). Example: (1344, 90646).
        if n_cols == int(good_full.sum()):
            source_indices = np.flatnonzero(good_full)
            desired_full_indices = desired_cortical  # cortical CIFTI indices are 0..59411
            return _map_compact_columns_from_source(
                betas_trials,
                source_indices,
                desired_full_indices,
                sub,
                "good_grayordinates_mask full-91k",
            )

        # 4b. Columns correspond only to good cortical grayordinates.
        if n_cols == int(good_cortical.sum()):
            source_indices = np.flatnonzero(good_cortical)
            return _map_compact_columns_from_source(
                betas_trials,
                source_indices,
                desired_cortical,
                sub,
                "good_grayordinates_mask cortical",
            )

    elif good_mask.shape[0] == N_CORTICAL:
        # Columns correspond to good cortical grayordinates.
        if n_cols == int(good_mask.sum()):
            source_indices = np.flatnonzero(good_mask)
            return _map_compact_columns_from_source(
                betas_trials,
                source_indices,
                desired_cortical,
                sub,
                "good cortical mask",
            )

    raise ValueError(
        f"Unexpected betas_trials shape for {sub}: {betas_trials.shape}. "
        f"Expected columns {N_GRAYORD}, {N_CORTICAL}, {n_compact}, "
        f"sum(good_grayordinates_mask), or sum(good cortical mask). "
        f"good_grayordinates_mask shape={good_mask.shape}, sum={int(good_mask.sum())}."
    )


def load_subject_data(sid):
    sub = subject_str(sid)

    m0_delta_path = os.path.join(M0_DIR, sub, "deltas_folds.npy")
    m2_delta_path = os.path.join(M2_DIR, sub, "deltas_folds.npy")
    for p in [m0_delta_path, m2_delta_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    m0_deltas = np.load(m0_delta_path).astype(np.float32)  # (5, 2, n_compact)
    m2_deltas = np.load(m2_delta_path).astype(np.float32)  # (5, 3, n_compact)
    n_compact = int(m0_deltas.shape[2])
    assert m2_deltas.shape[2] == n_compact

    mask_path = os.path.join(M2_DIR, sub, "combined_mask.npy")
    if not os.path.isfile(mask_path):
        mask_path = os.path.join(M0_DIR, sub, "combined_mask.npy")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"combined_mask.npy not found for {sub}")

    compact_mask_full = np.load(mask_path).astype(bool)
    assert compact_mask_full.shape[0] == N_GRAYORD, (
        f"Mask shape {compact_mask_full.shape[0]}, expected {N_GRAYORD}"
    )
    compact_mask = compact_mask_full[:N_CORTICAL]
    assert int(compact_mask.sum()) == n_compact, (
        f"Mask has {int(compact_mask.sum())} True cortical vertices but deltas have {n_compact}"
    )

    trial_path = os.path.join(BETA_DIR, sub, "betas_trials.npy")
    stim_path  = os.path.join(BETA_DIR, sub, "stimorder.npy")
    if not os.path.isfile(trial_path):
        raise FileNotFoundError(trial_path)
    if not os.path.isfile(stim_path):
        raise FileNotFoundError(stim_path)

    betas_trials = np.load(trial_path).astype(np.float32)
    stimorder = normalize_stimorder_zero_based(np.load(stim_path).astype(int))

    betas_cortical = select_compact_betas(
        betas_trials=betas_trials,
        compact_mask_full=compact_mask_full,
        compact_mask_cortical=compact_mask,
        n_compact=n_compact,
        sub=sub,
    )

    if betas_cortical.shape[1] != n_compact:
        raise ValueError(
            f"{sub}: selected beta matrix has {betas_cortical.shape[1]} columns, "
            f"but model compact space has {n_compact}."
        )

    betas_cond = np.zeros((N_CONDITIONS, n_compact), dtype=np.float32)
    for c in range(N_CONDITIONS):
        idx = np.where(stimorder == c)[0]
        if len(idx) == 0:
            betas_cond[c] = np.nan
        else:
            betas_cond[c] = np.nanmean(betas_cortical[idx], axis=0)

    if np.isnan(betas_cond).any():
        missing = np.where(np.isnan(betas_cond).any(axis=1))[0]
        raise ValueError(f"{sub}: missing condition betas for conditions {missing.tolist()}")

    m = betas_cond.mean(axis=0, keepdims=True)
    s = betas_cond.std(axis=0, keepdims=True) + 1e-8
    betas_cond = ((betas_cond - m) / s).astype(np.float32)

    return betas_cond, compact_mask, m0_deltas, m2_deltas

# =============================================================================
# [7] RESUME HELPERS
# =============================================================================

def load_existing_results():
    if not os.path.isfile(RESULTS_CSV):
        print("\nNo existing permutation_null_results.csv found. Starting from scratch.")
        return pd.DataFrame(columns=["subject", "roi", "perm_type", "perm_idx", "r_m0", "r_m2", "delta"])

    df = pd.read_csv(RESULTS_CSV)
    required = {"subject", "roi", "perm_type", "perm_idx", "r_m0", "r_m2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Existing results CSV is missing columns: {missing}")

    if "delta" not in df.columns:
        df["delta"] = df["r_m2"] - df["r_m0"]

    print(f"\nLoaded existing results: {RESULTS_CSV}")
    print(f"  rows       : {len(df)}")
    print(f"  subjects   : {df['subject'].nunique()}")
    print(f"  ROIs found : {sorted(df['roi'].dropna().unique().tolist())}")

    return df


def subject_has_complete_current_results(df, sub):
    """Return True if subject has all current ROI rows for true/full/calorie shuffles."""
    if df.empty:
        return False

    subdf = df[df["subject"] == sub]
    if subdf.empty:
        return False

    for roi in ROI_GROUPS.keys():
        roi_df = subdf[subdf["roi"] == roi]
        if roi_df.empty:
            return False

        true_df = roi_df[roi_df["perm_type"] == "true"]
        if true_df["perm_idx"].nunique() < 1 or -1 not in set(true_df["perm_idx"].astype(int)):
            return False

        full_df = roi_df[roi_df["perm_type"] == "full_shuffle"]
        cal_df  = roi_df[roi_df["perm_type"] == "calorie_shuffle"]
        if full_df["perm_idx"].nunique() < N_SHUFFLES:
            return False
        if cal_df["perm_idx"].nunique() < N_SHUFFLES:
            return False

        # Basic finite-value check for the rows used in summaries.
        check_df = pd.concat([true_df, full_df, cal_df], axis=0)
        if check_df[["r_m0", "r_m2"]].isna().all(axis=None):
            return False

    return True


def get_subjects_to_run(existing_df):
    subjects = [subject_str(s) for s in ALL_SUBJECTS]
    if not RUN_ONLY_MISSING_SUBJECTS:
        return subjects

    missing = []
    complete = []
    for sub in subjects:
        if subject_has_complete_current_results(existing_df, sub):
            complete.append(sub)
        else:
            missing.append(sub)

    print("\nResume check using CURRENT ROI definitions:")
    print(f"  Complete subjects: {len(complete)}")
    print(f"  Missing subjects : {len(missing)}")
    if complete:
        print(f"  Complete: {complete}")
    if missing:
        print(f"  To run  : {missing}")

    return missing


def remove_subject_current_roi_rows(df, sub):
    """Remove current-ROI rows for a subject before appending rerun rows."""
    if df.empty:
        return df
    current_rois = set(ROI_GROUPS.keys())
    keep = ~((df["subject"] == sub) & (df["roi"].isin(current_rois)))
    return df.loc[keep].copy()


def backup_existing_file_once(path):
    if not os.path.isfile(path):
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.replace(".csv", f"_backup_{timestamp}.csv")
    shutil.copy2(path, backup)
    print(f"  Backed up existing file:\n    {path}\n    -> {backup}")
    return backup

# =============================================================================
# [8] RUN PERMUTATIONS FOR ONE SUBJECT
# =============================================================================

def run_subject(sid, full_perms, calorie_perms, outer_splits):
    sub = subject_str(sid)
    print(f"\n{'='*70}\nSubject {sub}\n{'='*70}")

    betas, compact_mask, m0_deltas, m2_deltas = load_subject_data(sid)
    print(f"  Betas: {betas.shape} | Compact vertices: {int(compact_mask.sum())}")

    M0_BANDS = [LowVis, HighVis]
    M2_BANDS = [LowVis, HighVis, PredCLIP]

    rows = []

    print("  Fitting true M0...")
    r_true_m0 = run_fixed_delta_model(betas, M0_BANDS, m0_deltas, outer_splits)
    print("  Fitting true M2...")
    r_true_m2 = run_fixed_delta_model(betas, M2_BANDS, m2_deltas, outer_splits)

    for roi, roi_mask in roi_masks.items():
        rows.append({
            "subject": sub,
            "roi": roi,
            "perm_type": "true",
            "perm_idx": -1,
            "r_m0": roi_mean(r_true_m0, compact_mask, roi_mask),
            "r_m2": roi_mean(r_true_m2, compact_mask, roi_mask),
        })

    print(f"  Running {N_SHUFFLES} full-condition shuffles...")
    for pi, perm in enumerate(full_perms):
        r_p_m0 = run_fixed_delta_model(
            betas, M0_BANDS, m0_deltas, outer_splits, condition_perm=perm
        )
        r_p_m2 = run_fixed_delta_model(
            betas, M2_BANDS, m2_deltas, outer_splits, condition_perm=perm
        )
        for roi, roi_mask in roi_masks.items():
            rows.append({
                "subject": sub,
                "roi": roi,
                "perm_type": "full_shuffle",
                "perm_idx": pi,
                "r_m0": roi_mean(r_p_m0, compact_mask, roi_mask),
                "r_m2": roi_mean(r_p_m2, compact_mask, roi_mask),
            })
        if (pi + 1) % 20 == 0:
            print(f"    full shuffle {pi + 1}/{N_SHUFFLES}")

    print(f"  Running {N_SHUFFLES} calorie-label shuffles...")
    for pi, perm in enumerate(calorie_perms):
        # M0 is unchanged by PredCLIP row permutation.
        r_p_m0_cal = r_true_m0
        r_p_m2_cal = run_fixed_delta_model(
            betas, M2_BANDS, m2_deltas, outer_splits, predclip_perm=perm
        )
        for roi, roi_mask in roi_masks.items():
            rows.append({
                "subject": sub,
                "roi": roi,
                "perm_type": "calorie_shuffle",
                "perm_idx": pi,
                "r_m0": roi_mean(r_p_m0_cal, compact_mask, roi_mask),
                "r_m2": roi_mean(r_p_m2_cal, compact_mask, roi_mask),
            })
        if (pi + 1) % 20 == 0:
            print(f"    calorie shuffle {pi + 1}/{N_SHUFFLES}")

    out = pd.DataFrame(rows)
    out["delta"] = out["r_m2"] - out["r_m0"]
    return out

# =============================================================================
# [9] SUMMARY TABLES
# =============================================================================

def compute_null_summary(df):
    print(f"\n{'='*70}")
    print("Summary: null distribution means")
    print(f"{'='*70}")

    summary_rows = []
    current_df = df[df["roi"].isin(ROI_GROUPS.keys())].copy()

    for roi in ROI_GROUPS.keys():
        for perm_type in ["full_shuffle", "calorie_shuffle"]:
            null = current_df[(current_df["roi"] == roi) & (current_df["perm_type"] == perm_type)]
            true = current_df[(current_df["roi"] == roi) & (current_df["perm_type"] == "true")]

            for col, label in [("r_m0", "M0"), ("r_m2", "M2"), ("delta", "M2-M0")]:
                null_vals = null[col].dropna().values
                true_vals = true[col].dropna().values
                if len(null_vals) == 0 or len(true_vals) == 0:
                    continue

                t, p = stats.ttest_1samp(null_vals, popmean=0)
                null_mean = float(np.mean(null_vals))
                true_mean = float(np.mean(true_vals))
                true_gt_null_p = float(np.mean(null_vals >= true_mean))

                summary_rows.append({
                    "roi": roi,
                    "perm_type": perm_type,
                    "quantity": label,
                    "n_null_values": int(len(null_vals)),
                    "n_subjects_true": int(len(true_vals)),
                    "null_mean": null_mean,
                    "null_std": float(np.std(null_vals, ddof=1)),
                    "null_frac_gt0": float(np.mean(null_vals > 0)),
                    "t_null_vs_0": float(t),
                    "p_null_vs_0": float(p),
                    "true_mean": true_mean,
                    "true_gt_null_p": true_gt_null_p,
                })

                print(
                    f"  {roi:22s} {perm_type:18s} {label:6s}: "
                    f"null={null_mean:+.5f} true={true_mean:+.5f} "
                    f"p_null_vs_0={p:.4f}"
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved: {SUMMARY_CSV}")
    return summary_df


def compute_calorie_specificity(df):
    print(f"\n{'='*70}")
    print("Calorie specificity: true delta - mean(shuffle delta)")
    print(f"{'='*70}")

    current_df = df[df["roi"].isin(ROI_GROUPS.keys())].copy()
    spec_rows = []

    for sub in sorted(current_df["subject"].dropna().unique()):
        for roi in ROI_GROUPS.keys():
            true_val = current_df[
                (current_df["subject"] == sub)
                & (current_df["roi"] == roi)
                & (current_df["perm_type"] == "true")
            ]["delta"].dropna().values

            shuf_vals = current_df[
                (current_df["subject"] == sub)
                & (current_df["roi"] == roi)
                & (current_df["perm_type"] == "calorie_shuffle")
            ]["delta"].dropna().values

            if len(true_val) >= 1 and len(shuf_vals) >= N_SHUFFLES:
                tv = float(true_val[0])
                sm = float(np.mean(shuf_vals))
                spec_rows.append({
                    "subject": sub,
                    "roi": roi,
                    "true_delta": tv,
                    "shuf_mean": sm,
                    "specificity": tv - sm,
                    "n_shuffles": int(len(shuf_vals)),
                })

    spec_df = pd.DataFrame(spec_rows)
    spec_df.to_csv(SPEC_CSV, index=False)
    print(f"Saved: {SPEC_CSV}")

    summary_rows = []
    pvals = []
    row_indices = []

    for roi in ROI_GROUPS.keys():
        vals = spec_df[spec_df["roi"] == roi]["specificity"].dropna().values
        if len(vals) == 0:
            continue

        obs = float(np.mean(vals))
        lo, hi = ci95(vals)
        p_signflip = signflip_1d(vals, alternative="greater", seed=RANDOM_STATE + 99)
        t, p_t = stats.ttest_1samp(vals, popmean=0)
        dz = cohens_dz(vals)

        row = {
            "roi": roi,
            "n_subjects": int(len(vals)),
            "mean_specificity": obs,
            "ci95_lo": float(lo),
            "ci95_hi": float(hi),
            "cohens_dz": float(dz),
            "p_signflip": float(p_signflip),
            "p_ttest": float(p_t),
            "frac_gt0": float(np.mean(vals > 0)),
        }
        summary_rows.append(row)
        pvals.append(p_signflip)
        row_indices.append(len(summary_rows) - 1)

    spec_summary = pd.DataFrame(summary_rows)
    if len(spec_summary) > 0:
        q = multipletests(spec_summary["p_signflip"].values, method="fdr_bh")[1]
        spec_summary["q_fdr_bh"] = q
        spec_summary["sig_fdr_05"] = q < ALPHA_FDR

    spec_summary.to_csv(SPEC_SUMMARY_CSV, index=False)
    print(f"Saved: {SPEC_SUMMARY_CSV}")

    print("\nPer-ROI specificity summary:")
    for _, row in spec_summary.iterrows():
        sig = "***" if row["q_fdr_bh"] < 0.001 else "**" if row["q_fdr_bh"] < 0.01 else "*" if row["q_fdr_bh"] < 0.05 else "ns"
        print(
            f"  {row['roi']:22s}: mean={row['mean_specificity']:+.5f} "
            f"95% CI [{row['ci95_lo']:+.5f}, {row['ci95_hi']:+.5f}] "
            f"dz={row['cohens_dz']:.3f} p={row['p_signflip']:.4f} "
            f"q={row['q_fdr_bh']:.4f} n={int(row['n_subjects'])} {sig}"
        )

    return spec_df, spec_summary

# =============================================================================
# [10] FIGURES
# =============================================================================

def make_figures(df, spec_df):
    current_df = df[df["roi"].isin(ROI_GROUPS.keys())].copy()

    # Figure 1: full shuffle null distributions.
    fig, axes = plt.subplots(len(ROI_GROUPS), 3, figsize=(14, 4 * len(ROI_GROUPS)))
    if len(ROI_GROUPS) == 1:
        axes = np.asarray([axes])

    for ri, roi in enumerate(ROI_GROUPS.keys()):
        null = current_df[(current_df["roi"] == roi) & (current_df["perm_type"] == "full_shuffle")]
        true = current_df[(current_df["roi"] == roi) & (current_df["perm_type"] == "true")]

        for ci, (col, label, color) in enumerate([
            ("r_m0",  "M0 r_joint",  "#4878CF"),
            ("r_m2",  "M2 r_joint",  "#D65F5F"),
            ("delta", "M2-M0 delta", "#8C564B"),
        ]):
            ax = axes[ri, ci]
            all_null = null[col].dropna().values
            true_vals = true[col].dropna().values
            if len(all_null) == 0 or len(true_vals) == 0:
                ax.set_axis_off()
                continue

            ax.hist(all_null, bins=40, color=color, alpha=0.60, label="Null")
            ax.axvline(0, color="black", lw=1.2, ls="--")
            ax.axvline(float(np.mean(all_null)), color=color, lw=1.8,
                       label=f"Null μ={np.mean(all_null):+.4f}")
            ax.axvline(float(np.mean(true_vals)), color="red", lw=2.0,
                       label=f"True μ={np.mean(true_vals):+.4f}")
            t, p = stats.ttest_1samp(all_null, popmean=0)
            ax.set_title(f"{roi} | {label}\np(null≠0)={p:.4f}", fontsize=8)
            ax.set_xlabel("r", fontsize=8)
            ax.legend(fontsize=6, frameon=False)

    fig.suptitle(
        f"Full condition shuffle null\n"
        f"Fixed deltas, scale-relative alpha factor={REGULARIZATION_FACTOR}",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    p = os.path.join(FIGDIR, "null_full_shuffle.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

    # Figure 2: calorie specificity per subject per ROI.
    fig, axes = plt.subplots(1, len(ROI_GROUPS), figsize=(5 * len(ROI_GROUPS), 5))
    if len(ROI_GROUPS) == 1:
        axes = [axes]

    for ai, roi in enumerate(ROI_GROUPS.keys()):
        ax = axes[ai]
        subdf = spec_df[spec_df["roi"] == roi].copy()
        vals = subdf["specificity"].values
        subs = subdf["subject"].values
        colors = ["#D62728" if v > 0 else "#4878CF" for v in vals]
        ax.bar(np.arange(len(vals)), vals, color=colors, alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels([s.replace("sub-", "") for s in subs], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{roi}\nmean={np.mean(vals):+.5f}", fontsize=9)
        ax.set_ylabel("Specificity (true − shuffle mean)")

    fig.suptitle(
        "Calorie specificity per subject\n"
        "Positive = true PredCLIP ordering outperforms random permutation",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    p = os.path.join(FIGDIR, "calorie_specificity_per_subject.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

    # Figure 3: true delta vs shuffle mean delta.
    fig, axes = plt.subplots(1, len(ROI_GROUPS), figsize=(5 * len(ROI_GROUPS), 5))
    if len(ROI_GROUPS) == 1:
        axes = [axes]

    for ai, roi in enumerate(ROI_GROUPS.keys()):
        ax = axes[ai]
        subdf = spec_df[spec_df["roi"] == roi].copy()
        ax.scatter(subdf["shuf_mean"], subdf["true_delta"], s=50, color="#333333", alpha=0.8)

        vals_for_lim = np.r_[subdf["shuf_mean"].values, subdf["true_delta"].values]
        lim = max(0.001, float(np.nanmax(np.abs(vals_for_lim))) * 1.2)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Mean shuffle delta")
        ax.set_ylabel("True delta")
        ax.set_title(roi, fontsize=9)

    fig.suptitle(
        "True vs shuffle mean delta per subject\n"
        "Points above diagonal = calorie-specific gain",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    p = os.path.join(FIGDIR, "true_vs_shuffle_scatter.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

# =============================================================================
# [11] MAIN
# =============================================================================

def main():
    rng = np.random.default_rng(RANDOM_STATE)
    outer_kf = KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    outer_splits = list(outer_kf.split(np.arange(N_CONDITIONS)))

    # Fixed permutations shared across subjects, matching the original design.
    full_perms = [rng.permutation(N_CONDITIONS) for _ in range(N_SHUFFLES)]
    calorie_perms = [rng.permutation(N_CONDITIONS) for _ in range(N_SHUFFLES)]

    existing_df = load_existing_results()
    subjects_to_run = get_subjects_to_run(existing_df)

    backup_path = None
    if BACKUP_EXISTING_RESULTS_BEFORE_FIRST_WRITE and len(subjects_to_run) > 0:
        backup_path = backup_existing_file_once(RESULTS_CSV)

    skipped = []
    df_all = existing_df.copy()

    if len(subjects_to_run) == 0:
        print("\nNo missing subjects under the current ROI definition. Recomputing summaries only.")
    else:
        for sub in subjects_to_run:
            sid = int(sub.replace("sub-", ""))
            try:
                sub_df = run_subject(sid, full_perms, calorie_perms, outer_splits)
            except Exception as exc:
                print(f"\nSKIPPED {sub}: {type(exc).__name__}: {exc}")
                skipped.append({"subject": sub, "error": repr(exc)})
                continue

            # Replace any existing current-ROI rows for this subject, then append rerun rows.
            df_all = remove_subject_current_roi_rows(df_all, sub)
            df_all = pd.concat([df_all, sub_df], axis=0, ignore_index=True)
            df_all["delta"] = df_all["r_m2"] - df_all["r_m0"]

            if WRITE_AFTER_EACH_SUBJECT:
                df_all.to_csv(RESULTS_CSV, index=False)
                print(f"  Incremental save: {RESULTS_CSV}")

    # Final write and summaries.
    df_all["delta"] = df_all["r_m2"] - df_all["r_m0"]
    df_all.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved full results: {RESULTS_CSV}")

    summary_df = compute_null_summary(df_all)
    spec_df, spec_summary = compute_calorie_specificity(df_all)
    make_figures(df_all, spec_df)

    metadata = {
        "script_name": "run_permutation_controls.py",
        "historical_source_script": "permutation_null_fixed_deltas_resume_highlevelvtc.py",
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Fixed-delta permutation controls for M0/M2, resumable, final HighLevelVTC ROI definition.",
        "outdir": OUTDIR,
        "results_csv": RESULTS_CSV,
        "summary_csv": SUMMARY_CSV,
        "specificity_csv": SPEC_CSV,
        "specificity_summary_csv": SPEC_SUMMARY_CSV,
        "backup_existing_results": backup_path,
        "run_only_missing_subjects": RUN_ONLY_MISSING_SUBJECTS,
        "n_shuffles": N_SHUFFLES,
        "n_conditions": N_CONDITIONS,
        "n_outer_folds": N_OUTER_FOLDS,
        "regularization_factor": REGULARIZATION_FACTOR,
        "subjects_requested": [subject_str(s) for s in ALL_SUBJECTS],
        "subjects_run_this_execution": subjects_to_run,
        "subjects_skipped_this_execution": skipped,
        "roi_groups": ROI_GROUPS,
        "predclip_path": PREDCLIP_PATH,
        "model_dirs": {
            "M0": M0_DIR,
            "M2": M2_DIR,
        },
        "note": (
            "Existing FoodVTC rows from older scripts are not treated as HighLevelVTC. "
            "Subjects are considered complete only if all current ROI names have true, full_shuffle, "
            "and calorie_shuffle rows for the requested N_SHUFFLES."
        ),
    }

    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {METADATA_JSON}")

    print("\nDone.")
    print(f"All outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()
