# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:20:13 2026

@author: G.Marrazzo
"""

r"""
NeuroRepFood — Encoding model comparison for calorie semantic decomposition
===========================================================================

Purpose
-------
Fit a clean model family testing whether perceived-calorie information improves
neural prediction beyond visual features.

Main model family
-----------------
M0_visual:
    LowVis + HighVis

M1_visual_rawcal:
    LowVis + HighVis + CalorieRaw

M2_visual_predclip:
    LowVis + HighVis + CaloriePredCLIP

M3_visual_resclip:
    LowVis + HighVis + CalorieResCLIP

M4_visual_pred_resclip:
    LowVis + HighVis + CaloriePredCLIP + CalorieResCLIP
    
Diagnostic CLIP-separated model family
--------------------------------------
D0_visual_noclip:
    LowVis + HighVisNoCLIP

D1_visual_clipband:
    LowVis + HighVisNoCLIP + CLIPown

D2_visual_clipband_predclip:
    LowVis + HighVisNoCLIP + CLIPown + CaloriePredCLIP

Interpretation of diagnostic family
-----------------------------------
This diagnostic family tests whether the weak HighLevelVTC prediction of the
primary visual baseline reflects joint PCA compression of CLIP together with
AlexNet and CORnet features. In D1, CLIP is given its own feature band and its
own ridge regularisation. D2 then tests whether the targeted CaloriePredCLIP
axis still improves prediction beyond this CLIP-separated visual baseline.

The diagnostic family is additive and does not redefine the primary M0-M4
model family.

Interpretation
--------------
The primary question is:

    Does adding a semantically meaningful calorie dimension improve neural
    prediction beyond visual features?

Therefore Palatability and Familiarity are deliberately excluded from the
main model family. They can be added later as secondary control models.

Critical leakage control
------------------------
For each outer neural CV fold:

    1. The calorie target is z-scored using training-image statistics only.
    2. CLIP -> calorie prediction is fitted on training images only.
    3. CaloriePredCLIP and CalorieResCLIP are generated for all 96 images
       using that fold-specific fit.
    4. Each neural model is trained on outer_train and evaluated on outer_test.

Recommended setting
-------------------
CALORIE_TARGET_MODE = "loso_group"

For each fMRI subject, this uses the mean perceived-calorie rating from all
other subjects. This avoids defining the calorie predictor from the same
participant's own behavioral ratings.

Inputs
------
Betas:
    betas_trials.npy (n_available_trials, n_good_grayord)
    stimorder.npy (n_available_trials,)
    Subjects with fewer valid runs are accepted if all 96 conditions have at
    least one beta estimate.

Features:
    feature_models/bands/band_LowVis.npy
    feature_models/bands/band_HighVis.npy
    feature_models/CLIP.npy or feature_models/bands/band_CLIP.npy
    feature_models/Calorie_persubject.npy

Outputs
-------
BASE_OUTDIR:
    reproduced_outputs/encoding_models

For each model:
    BASE_OUTDIR/<model_key>/sub-XXX/r_joint.npy
    BASE_OUTDIR/<model_key>/sub-XXX/r_split_<band>.npy
    BASE_OUTDIR/<model_key>/group_r_joint.npy
    BASE_OUTDIR/<model_key>/group_r_split_<band>.npy

Notes
-----
- HighVis currently contains AlexNet + CORnetIT + CLIP PCA in your setup.
  Therefore CaloriePredCLIP tests whether a targeted 1D CLIP-predicted calorie
  axis improves prediction beyond the broader HighVis feature space.
- Diagnostic models D0-D2 use HighVisNoCLIP and CLIPown to test whether the
  original HighVis baseline underestimates prediction because CLIP-relevant
  directions were jointly compressed with AlexNet and CORnet features.
"""

# =============================================================================
# [0] IMPORTS
# =============================================================================

import os
import gc
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import nibabel as nib
import torch

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.base import TransformerMixin

from himalaya.backend import set_backend
from himalaya.kernel_ridge import MultipleKernelRidgeCV, Kernelizer, ColumnKernelizer
from himalaya.scoring import correlation_score_split

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.base")
warnings.filterwarnings("ignore", category=RuntimeWarning)

print("Imports OK.")
print("AUTHOR-SIDE SENSITIVITY RUN: subjects 104/129; models M0/M2/D1/D2")


# =============================================================================
# [1] CONFIGURATION
# =============================================================================

# RELEASE PATH ADAPTATION ONLY.
# Numerical model-fitting code below is unchanged from the executed historical
# fit_banded_ridge_per_subj_v2_skip.py archived in Python(2).zip.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

MAINDIR = REPO_ROOT
FEAT_DIR = os.path.join(REPO_ROOT, "reproduced_outputs", "feature_extraction")
BAND_DIR = os.path.join(REPO_ROOT, "reproduced_outputs", "feature_bands")
BETA_DIR = os.path.join(REPO_ROOT, "data", "glmsingle")

STIM_CSV = os.path.join(
    REPO_ROOT, "data", "stimuli", "ordered_stimuli.csv"
)

BASE_OUTDIR = os.path.join(REPO_ROOT, "reproduced_outputs", "encoding_sensitivity")
os.makedirs(BASE_OUTDIR, exist_ok=True)

# Full behavioural-rating subject order.
# IMPORTANT: this order must match columns in Calorie_persubject.npy.
PILOT_SUBJECTS = [
    104, 105, 109, 112, 115, 117, 118, 122, 124, 125, 126,
    129, 132, 134, 137, 139, 140, 141, 143, 144, 145, 146, 149,
    150, 151,
]

# Optional subject exclusions from the fMRI encoding analysis.
# Do NOT add subjects here merely because they have fewer runs: shorter but
# valid GLMsingle outputs are accepted below as long as all 96 conditions exist.
EXCLUDE_SUBJECTS = {}
ANALYSIS_SUBJECTS = [s for s in PILOT_SUBJECTS if s not in EXCLUDE_SUBJECTS]

N_CONDITIONS = 96
FULL_DESIGN_REPS = 14
FULL_DESIGN_TRIALS = N_CONDITIONS * FULL_DESIGN_REPS
MIN_REPS_PER_CONDITION = 1
REQUIRE_ALL_CONDITIONS_PRESENT = True
N_GRAYORD = 91282
N_CORTICAL = 59412

N_ITER_SEARCH_INIT = 100

USE_HYPERGRADIENT = True
HG_MAX_ITER = 20
HG_MAX_ITER_INNER = 20
HG_TOL = 1e-4

ALPHAS = np.logspace(-5, 10, 15)
N_INNER_FOLDS = 5
N_OUTER_FOLDS = 5

RANDOM_STATE = 42



N_TARGETS_BATCH = 90000
PERC_THRESHOLD = 0.05
STD_PERCENTILE = 50

# If True, already-completed model/subject folders are not refit.
# Completion is checked using the expected output files and array shapes, not only DONE.txt.
SKIP_COMPLETED = True

# If False, outputs can be skipped when all required maps exist even if DONE.txt is missing.
# This is useful when a previous run finished all arrays but failed before writing DONE.txt.
REQUIRE_DONE_FOR_SKIP = False

CALORIE_TARGET_MODE = "loso_group"
CALORIE_PRED_ALPHAS = np.logspace(-4, 4, 25)

# Optional CIFTI template. The minimal public reproduction starts from
# GLMsingle derivatives, so missing fMRIPrep dtseries only suppress dscalar
# writing; all numerical .npy outputs are unaffected.
CIFTI_TEMPLATE_PATTERN = os.path.join(
    REPO_ROOT,
    "optional_preprocessing",
    "fmriprep",
    "derivatives",
    "sub-{sub_id}",
    "ses-1",
    "func",
    "sub-{sub_id}_ses-1_task-food_dir-AP_run-01"
    "_space-fsLR_den-91k_bold.dtseries.nii",
)

MODEL_SPECS = OrderedDict([
    # -------------------------------------------------------------------------
    # Primary model family — unchanged
    # -------------------------------------------------------------------------
    ("M0_visual", {
        "label": "LowVis + HighVis",
        "bands": ["LowVis", "HighVis"],
        "family": "primary",
    }),
    ("M1_visual_rawcal", {
        "label": "LowVis + HighVis + CalorieRaw",
        "bands": ["LowVis", "HighVis", "CalorieRaw"],
        "family": "primary",
    }),
    ("M2_visual_predclip", {
        "label": "LowVis + HighVis + CaloriePredCLIP",
        "bands": ["LowVis", "HighVis", "CaloriePredCLIP"],
        "family": "primary",
    }),
    ("M3_visual_resclip", {
        "label": "LowVis + HighVis + CalorieResCLIP",
        "bands": ["LowVis", "HighVis", "CalorieResCLIP"],
        "family": "primary",
    }),
    ("M4_visual_pred_resclip", {
        "label": "LowVis + HighVis + CaloriePredCLIP + CalorieResCLIP",
        "bands": ["LowVis", "HighVis", "CaloriePredCLIP", "CalorieResCLIP"],
        "family": "primary",
    }),

    # -------------------------------------------------------------------------
    # Diagnostic CLIP-separated baseline family — additive, not a replacement
    # -------------------------------------------------------------------------
    ("D0_visual_noclip", {
        "label": "LowVis + HighVisNoCLIP",
        "bands": ["LowVis", "HighVisNoCLIP"],
        "family": "clipband_diagnostic",
    }),
    ("D1_visual_clipband", {
        "label": "LowVis + HighVisNoCLIP + CLIPown",
        "bands": ["LowVis", "HighVisNoCLIP", "CLIPown"],
        "family": "clipband_diagnostic",
    }),
    ("D2_visual_clipband_predclip", {
        "label": "LowVis + HighVisNoCLIP + CLIPown + CaloriePredCLIP",
        "bands": ["LowVis", "HighVisNoCLIP", "CLIPown", "CaloriePredCLIP"],
        "family": "clipband_diagnostic",
    }),
])

# AUTHOR-SIDE SENSITIVITY TEST ONLY.
# Keep ANALYSIS_SUBJECTS as the full cohort so LOSO targets and the common
# beta-variance mask remain defined exactly as in the historical analysis.
SENSITIVITY_FIT_SUBJECTS = [104, 129]
SENSITIVITY_MODEL_KEYS = [
    "M0_visual",
    "M2_visual_predclip",
    "D1_visual_clipband",
    "D2_visual_clipband_predclip",
]

try:
    backend = set_backend("torch_cuda", on_error="warn")
    print("Backend: torch_cuda")
except Exception:
    backend = set_backend("numpy", on_error="warn")
    print("Backend: numpy")


# =============================================================================
# [2] LOAD FEATURES AND RATINGS
# =============================================================================

print("\n[2] Loading features and ratings")

stim_df = pd.read_csv(STIM_CSV)
CONDS = stim_df["Value"].tolist()
assert len(CONDS) == N_CONDITIONS

visual_bands = {}

for fs in ["LowVis", "HighVis", "HighVisNoCLIP", "CLIPown"]:
    fpath = os.path.join(BAND_DIR, f"band_{fs}.npy")
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Missing visual band: {fpath}")

    arr = np.load(fpath).astype(np.float32)

    if arr.shape[0] != N_CONDITIONS:
        raise ValueError(f"{fs}: expected 96 rows, got {arr.shape}")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{fs}: contains non-finite values")

    visual_bands[fs] = arr
    print(f"  {fs:18s}: {arr.shape}")

clip_candidates = [
    os.path.join(FEAT_DIR, "CLIP.npy"),
    os.path.join(BAND_DIR, "band_CLIP.npy"),
]

CLIP_FEATURES = None
CLIP_SOURCE = None

for p in clip_candidates:
    if os.path.isfile(p):
        CLIP_FEATURES = np.load(p).astype(np.float32)
        CLIP_SOURCE = p
        break

if CLIP_FEATURES is None:
    raise FileNotFoundError(
        "Could not find CLIP features. Tried:\n" + "\n".join(clip_candidates)
    )

if CLIP_FEATURES.shape[0] != N_CONDITIONS:
    raise ValueError(f"CLIP shape should be (96, D), got {CLIP_FEATURES.shape}")

if not np.all(np.isfinite(CLIP_FEATURES)):
    raise ValueError("CLIP_FEATURES contains non-finite values")

print(f"  {'CLIP predictor':18s}: {CLIP_FEATURES.shape} from {CLIP_SOURCE}")

cal_path = os.path.join(FEAT_DIR, "Calorie_persubject.npy")
if not os.path.isfile(cal_path):
    raise FileNotFoundError(f"Missing perceived calorie file: {cal_path}")

CALORIE_PERSUB = np.load(cal_path).astype(np.float32)

if CALORIE_PERSUB.shape != (N_CONDITIONS, len(PILOT_SUBJECTS)):
    raise ValueError(
        f"Calorie_persubject expected {(N_CONDITIONS, len(PILOT_SUBJECTS))}, "
        f"got {CALORIE_PERSUB.shape}"
    )

print(f"  {'Calorie_persub':18s}: {CALORIE_PERSUB.shape}")

SUB_ID_TO_COL = {sid: idx for idx, sid in enumerate(PILOT_SUBJECTS)}
ANALYSIS_COLS = [SUB_ID_TO_COL[sid] for sid in ANALYSIS_SUBJECTS]

print(f"  Analysis subjects : {ANALYSIS_SUBJECTS}")
print(f"  Excluded subjects : {sorted(EXCLUDE_SUBJECTS)}")


# =============================================================================
# [3] HELPERS
# =============================================================================

def normalize_stimorder_zero_based(stimorder, sub_id=None):
    """Return 0-based condition indices in [0, 95]."""
    stimorder = np.asarray(stimorder).astype(int).reshape(-1)

    if stimorder.size == 0:
        raise ValueError(f"sub-{sub_id}: empty stimorder")

    mn = int(np.nanmin(stimorder))
    mx = int(np.nanmax(stimorder))

    if mn >= 0 and mx <= N_CONDITIONS - 1:
        return stimorder

    if mn >= 1 and mx <= N_CONDITIONS:
        print(f"  WARNING sub-{sub_id}: stimorder appears 1-based; converting to 0-based.")
        return stimorder - 1

    raise ValueError(
        f"sub-{sub_id}: unexpected stimorder range min={mn}, max={mx}. "
        f"Expected 0..{N_CONDITIONS - 1} or 1..{N_CONDITIONS}."
    )


def load_betas_subject(sub_id):
    """
    Load trial-wise GLMsingle betas and average them to one beta per image.

    This function intentionally accepts subjects with fewer than the full 14 runs.
    It only requires that:
      1. betas_trials and stimorder have the same number of rows;
      2. all 96 conditions are represented at least MIN_REPS_PER_CONDITION times;
      3. good_grayordinates_mask is compatible with the beta matrix.
    """
    sub_dir = os.path.join(BETA_DIR, f"sub-{sub_id}")

    betas_path = os.path.join(sub_dir, "betas_trials.npy")
    stim_path = os.path.join(sub_dir, "stimorder.npy")
    mask_path = os.path.join(sub_dir, "good_grayordinates_mask.npy")

    for pth in [betas_path, stim_path, mask_path]:
        if not os.path.isfile(pth):
            raise FileNotFoundError(f"sub-{sub_id}: missing {pth}")

    betas_trials = np.load(betas_path).astype(np.float32)
    stimorder = np.load(stim_path).astype(int)
    good_mask = np.load(mask_path).astype(bool)

    if betas_trials.ndim != 2:
        raise ValueError(
            f"sub-{sub_id}: betas_trials must be 2D, got {betas_trials.shape}"
        )

    if good_mask.shape[0] != N_GRAYORD:
        raise ValueError(
            f"sub-{sub_id}: good_grayordinates_mask length {good_mask.shape[0]}, "
            f"expected {N_GRAYORD}"
        )

    if betas_trials.shape[1] != int(good_mask.sum()):
        raise ValueError(
            f"sub-{sub_id}: betas columns={betas_trials.shape[1]} but "
            f"good mask sum={int(good_mask.sum())}"
        )

    stimorder = normalize_stimorder_zero_based(stimorder, sub_id=sub_id)

    if betas_trials.shape[0] != stimorder.shape[0]:
        raise ValueError(
            f"sub-{sub_id}: betas_trials rows={betas_trials.shape[0]} but "
            f"stimorder length={stimorder.shape[0]}"
        )

    counts = np.bincount(stimorder, minlength=N_CONDITIONS)[:N_CONDITIONS]

    missing = np.where(counts < MIN_REPS_PER_CONDITION)[0]
    if REQUIRE_ALL_CONDITIONS_PRESENT and missing.size > 0:
        preview = [(int(c + 1), int(counts[c])) for c in missing[:20]]
        raise ValueError(
            f"sub-{sub_id}: missing/underrepresented conditions: {preview}. "
            f"Cannot build a 96-condition beta matrix."
        )

    if not np.all(counts == FULL_DESIGN_REPS):
        unique_counts = sorted(set(int(x) for x in counts.tolist()))
        print(
            f"  NOTE sub-{sub_id}: flexible trial count accepted. "
            f"Total trials={betas_trials.shape[0]} "
            f"(full design would be {FULL_DESIGN_TRIALS}); "
            f"condition reps range={int(counts.min())}..{int(counts.max())}; "
            f"unique counts={unique_counts}"
        )

    betas_avg = np.full(
        (N_CONDITIONS, betas_trials.shape[1]),
        np.nan,
        dtype=np.float32,
    )

    for c in range(N_CONDITIONS):
        idx = stimorder == c
        if idx.sum() >= MIN_REPS_PER_CONDITION:
            betas_avg[c] = np.nanmean(betas_trials[idx], axis=0).astype(np.float32)

    if not np.all(np.isfinite(betas_avg)):
        raise ValueError(
            f"sub-{sub_id}: non-finite values in condition-averaged beta matrix"
        )

    return betas_avg, good_mask


def get_calorie_target(sub_id):
    col = SUB_ID_TO_COL[sub_id]

    if CALORIE_TARGET_MODE == "subject_specific":
        return CALORIE_PERSUB[:, col].astype(np.float32)

    if CALORIE_TARGET_MODE == "loso_group":
        # Use only subjects retained for the fMRI analysis, excluding the current
        # fMRI subject. This avoids letting excluded subjects influence the
        # group behavioural calorie target.
        other_cols = [
            SUB_ID_TO_COL[sid]
            for sid in ANALYSIS_SUBJECTS
            if sid != sub_id
        ]
        return np.nanmean(CALORIE_PERSUB[:, other_cols], axis=1).astype(np.float32)

    if CALORIE_TARGET_MODE == "group_mean":
        return np.nanmean(CALORIE_PERSUB[:, ANALYSIS_COLS], axis=1).astype(np.float32)

    raise ValueError(
        "CALORIE_TARGET_MODE must be 'subject_specific', 'loso_group', or 'group_mean'"
    )

def zscore_using_train(x, train_idx):
    x = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(x[train_idx])
    sd = np.nanstd(x[train_idx]) + 1e-8
    return ((x - mu) / sd).astype(np.float32)


def predict_and_residualize_calorie_from_clip(calorie_raw, outer_train, outer_test):
    cal_z = zscore_using_train(calorie_raw, outer_train)

    ridge = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        RidgeCV(alphas=CALORIE_PRED_ALPHAS),
    )

    ridge.fit(CLIP_FEATURES[outer_train], cal_z[outer_train])

    cal_pred = ridge.predict(CLIP_FEATURES).astype(np.float32)
    cal_res = (cal_z - cal_pred).astype(np.float32)

    cal_pred_z = zscore_using_train(cal_pred, outer_train)
    cal_res_z = zscore_using_train(cal_res, outer_train)

    y_true = cal_z[outer_test]
    y_pred = cal_pred[outer_test]

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8)
    test_r2 = 1.0 - ss_res / ss_tot

    if np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        test_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        test_r = np.nan

    alpha_selected = np.nan
    try:
        alpha_selected = float(ridge.named_steps["ridgecv"].alpha_)
    except Exception:
        pass

    diag = {
        "calorie_pred_test_r2": float(test_r2),
        "calorie_pred_test_r": float(test_r),
        "calorie_pred_alpha": alpha_selected,
        "calorie_pred_train_var": float(np.var(cal_pred[outer_train])),
        "calorie_res_train_var": float(np.var(cal_res[outer_train])),
        "calorie_target_mode": CALORIE_TARGET_MODE,
    }

    return cal_z, cal_pred_z, cal_res_z, diag


def build_fold_feature_dict(sub_id, outer_train, outer_test):
    cal_raw = get_calorie_target(sub_id)

    cal_raw_z, cal_pred_z, cal_res_z, cal_diag = predict_and_residualize_calorie_from_clip(
        calorie_raw=cal_raw,
        outer_train=outer_train,
        outer_test=outer_test,
    )

    return {
        # Primary visual bands
        "LowVis": visual_bands["LowVis"],
        "HighVis": visual_bands["HighVis"],

        # Diagnostic CLIP-separated visual bands
        "HighVisNoCLIP": visual_bands["HighVisNoCLIP"],
        "CLIPown": visual_bands["CLIPown"],

        # Fold-specific calorie predictors
        "CalorieRaw": cal_raw_z.reshape(-1, 1),
        "CaloriePredCLIP": cal_pred_z.reshape(-1, 1),
        "CalorieResCLIP": cal_res_z.reshape(-1, 1),
    }, cal_diag


class KernelNormalizer(TransformerMixin):
    def fit(self, Ks, y=None):
        Ks_np = (
            Ks.detach().cpu().numpy()
            if isinstance(Ks, torch.Tensor)
            else np.asarray(Ks)
        ).astype(np.float32)

        n_k, n_tr, _ = Ks_np.shape

        self.H = (
            np.eye(n_tr, dtype=np.float32)
            - np.ones((n_tr, n_tr), dtype=np.float32) / n_tr
        )

        self.fros = []

        for K in Ks_np:
            Kc = self.H @ K @ self.H
            fro = float(np.linalg.norm(Kc, "fro"))
            self.fros.append(np.float32(fro if fro > 0 else 1.0))

        return self

    def transform(self, Ks):
        device = None

        if isinstance(Ks, torch.Tensor):
            device = Ks.device
            Ks_np = Ks.detach().cpu().numpy().astype(np.float32)
        else:
            Ks_np = np.asarray(Ks, dtype=np.float32)

        n_k, n1, n2 = Ks_np.shape
        out = []

        if n1 == n2 == self.H.shape[0]:
            for i, K in enumerate(Ks_np):
                out.append((self.H @ K @ self.H / self.fros[i]).astype(np.float32))
        else:
            for i, K in enumerate(Ks_np):
                out.append((K @ self.H / self.fros[i]).astype(np.float32))

        out = np.stack(out, axis=0).astype(np.float32)

        if device is not None:
            return torch.from_numpy(out).to(device)

        return out


def build_design_matrix_and_slices(feature_dict, model_bands):
    arrays = []
    dims = []

    for band in model_bands:
        arr = feature_dict[band].astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arrays.append(arr)
        dims.append(arr.shape[1])

    X = np.hstack(arrays).astype(np.float32)

    boundaries = np.concatenate([[0], np.cumsum(dims)])
    slices = [
        slice(int(s), int(e))
        for s, e in zip(boundaries[:-1], boundaries[1:])
    ]

    return X, slices, dims


def make_pipeline_random(model_bands, band_slices, inner_cv):
    base = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        Kernelizer(kernel="linear"),
    )

    kernels = [
        (band, base, sl)
        for band, sl in zip(model_bands, band_slices)
    ]

    mkr = MultipleKernelRidgeCV(
        kernels="precomputed",
        solver="random_search",
        solver_params=dict(
            n_iter=N_ITER_SEARCH_INIT,
            alphas=ALPHAS,
            n_targets_batch=N_TARGETS_BATCH,
            diagonalize_method="svd",
        ),
        cv=inner_cv,
        random_state=RANDOM_STATE,
    )

    return make_pipeline(ColumnKernelizer(kernels), KernelNormalizer(), mkr)


def make_pipeline_hypergradient(model_bands, band_slices, inner_cv, initial_deltas):
    base = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        Kernelizer(kernel="linear"),
    )

    kernels = [
        (band, base, sl)
        for band, sl in zip(model_bands, band_slices)
    ]

    mkr = MultipleKernelRidgeCV(
        kernels="precomputed",
        solver="hyper_gradient",
        solver_params=dict(
            initial_deltas=initial_deltas,
            max_iter=HG_MAX_ITER,
            tol=HG_TOL,
            max_iter_inner_hyper=HG_MAX_ITER_INNER,
            hyper_gradient_method="direct",
            n_targets_batch=N_TARGETS_BATCH,
        ),
        cv=inner_cv,
        random_state=RANDOM_STATE,
    )

    return make_pipeline(ColumnKernelizer(kernels), KernelNormalizer(), mkr)


def pearson_r_columns(A, B):
    A = A - np.nanmean(A, axis=0, keepdims=True)
    B = B - np.nanmean(B, axis=0, keepdims=True)

    num = np.nansum(A * B, axis=0)
    den = np.sqrt(np.nansum(A ** 2, axis=0) * np.nansum(B ** 2, axis=0))

    r = np.full(A.shape[1], np.nan, dtype=np.float32)
    valid = den > 1e-10
    r[valid] = (num[valid] / den[valid]).astype(np.float32)

    return r


def expand(compact, mask, fill=np.nan):
    full = np.full(N_GRAYORD, fill, dtype=np.float32)
    full[mask] = compact
    return full


def save_dscalar(data_91k, out_path, template_path, map_name):
    try:
        t = nib.load(template_path)
        bm = t.header.get_axis(1)
        sa = nib.cifti2.ScalarAxis([map_name])
        hd = nib.cifti2.Cifti2Header.from_axes((sa, bm))

        img = nib.Cifti2Image(
            data_91k.reshape(1, -1).astype(np.float32),
            header=hd,
        )

        nib.save(img, out_path)
        print(f"      saved: {os.path.basename(out_path)}")

    except Exception as e:
        print(f"      CIFTI failed ({map_name}): {e}")


def _safe_load_array(path, expected_shape=None):
    """Load an array for output-integrity checking."""
    if not os.path.isfile(path):
        return False, f"missing: {os.path.basename(path)}"

    try:
        arr = np.load(path, allow_pickle=True)
    except Exception as e:
        return False, f"could not load {os.path.basename(path)}: {e}"

    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        return False, (
            f"bad shape for {os.path.basename(path)}: "
            f"got {arr.shape}, expected {expected_shape}"
        )

    if arr.dtype != object and not np.all(np.isfinite(arr[np.isfinite(arr)])):
        return False, f"non-finite loading issue in {os.path.basename(path)}"

    return True, arr


def model_completed(model_outdir, sub_id, model_bands):
    """
    Robust completion check for one model/subject output folder.

    Returns
    -------
    complete : bool
        True only if all downstream-required outputs exist and have expected shapes.
    reason : str
        Human-readable reason for skipping or rerunning.
    """
    subdir = os.path.join(model_outdir, f"sub-{sub_id}")

    if not os.path.isdir(subdir):
        return False, "output folder missing"

    done_flag = os.path.join(subdir, "DONE.txt")
    if REQUIRE_DONE_FOR_SKIP and not os.path.isfile(done_flag):
        return False, "DONE.txt missing"

    map_names = [
        "r_joint",
        "r2_joint",
        "combined_mask",
    ]

    for band in model_bands:
        map_names.extend([
            f"r_split_{band}",
            f"perc_r_{band}",
        ])

    for name in map_names:
        ok, msg_or_arr = _safe_load_array(
            os.path.join(subdir, f"{name}.npy"),
            expected_shape=(N_GRAYORD,),
        )
        if not ok:
            return False, str(msg_or_arr)

    ok, fs_arr = _safe_load_array(os.path.join(subdir, "feature_spaces.npy"))
    if not ok:
        return False, str(fs_arr)

    try:
        saved_bands = list(fs_arr.tolist())
    except Exception:
        saved_bands = list(fs_arr)

    if saved_bands != list(model_bands):
        return False, f"feature_spaces mismatch: got {saved_bands}, expected {model_bands}"

    ok, deltas = _safe_load_array(os.path.join(subdir, "deltas_folds.npy"))
    if not ok:
        return False, str(deltas)

    if deltas.ndim != 3 or deltas.shape[0] != N_OUTER_FOLDS or deltas.shape[1] != len(model_bands):
        return False, (
            f"bad deltas_folds shape: got {deltas.shape}, "
            f"expected ({N_OUTER_FOLDS}, {len(model_bands)}, n_vertices)"
        )

    diag_path = os.path.join(subdir, "calorie_clip_decomposition_folds.csv")
    if not os.path.isfile(diag_path):
        return False, "calorie_clip_decomposition_folds.csv missing"

    return True, "complete"


# =============================================================================
# [4] LOAD BETAS AND BUILD COMMON STD MASK
# =============================================================================

print("\n[4] Loading betas + building std mask")

candidate_subjects = [
    s for s in ANALYSIS_SUBJECTS
    if os.path.isfile(os.path.join(BETA_DIR, f"sub-{s}", "betas_trials.npy"))
]

missing_beta_subjects = [
    s for s in ANALYSIS_SUBJECTS
    if not os.path.isfile(os.path.join(BETA_DIR, f"sub-{s}", "betas_trials.npy"))
]

print(f"  Candidate subjects with betas: {candidate_subjects}")
print(f"  Excluded subjects          : {sorted(EXCLUDE_SUBJECTS)}")
if missing_beta_subjects:
    print(f"  Missing beta subjects      : {missing_beta_subjects}")

if len(candidate_subjects) == 0:
    raise RuntimeError(f"No betas_trials.npy files found in {BETA_DIR}")

available_subjects = []
cached_betas = {}
beta_std_stack = []
good_gray_stack = []

for sub_id in candidate_subjects:
    try:
        betas_avg, good_mask = load_betas_subject(sub_id)
    except Exception as e:
        print(f"  WARNING: excluding sub-{sub_id}; beta loading failed: {e}")
        continue

    available_subjects.append(sub_id)
    cached_betas[sub_id] = (betas_avg, good_mask)

    std_full = np.full(N_GRAYORD, np.nan, dtype=np.float32)
    std_full[good_mask] = betas_avg.std(axis=0)

    beta_std_stack.append(std_full)
    good_gray_stack.append(good_mask)

    print(
        f"  sub-{sub_id}: std={betas_avg.std(axis=0).mean():.5f}  "
        f"good={good_mask.sum():,}"
    )

if len(available_subjects) == 0:
    raise RuntimeError("No valid subjects remained after beta integrity checks.")

print(f"  Valid analysis subjects    : {available_subjects}")

good_gray_arr = np.stack(good_gray_stack, axis=0)
common_good = good_gray_arr.sum(axis=0) >= int(np.ceil(0.9 * len(available_subjects)))

beta_std_mean = np.nanmean(np.stack(beta_std_stack, axis=0), axis=0)
beta_std_mean[~common_good] = np.nan

cortical_common = common_good[:N_CORTICAL] & np.isfinite(beta_std_mean[:N_CORTICAL])

std_threshold = np.nanpercentile(
    beta_std_mean[:N_CORTICAL][cortical_common],
    STD_PERCENTILE,
)

beta_std_mask = np.zeros(N_GRAYORD, dtype=bool)
beta_std_mask[:N_CORTICAL] = (
    cortical_common
    & (beta_std_mean[:N_CORTICAL] >= std_threshold)
)

np.save(os.path.join(BASE_OUTDIR, "beta_std_mask.npy"), beta_std_mask)
np.save(os.path.join(BASE_OUTDIR, "beta_std_mean.npy"), beta_std_mean)

print(f"\n  Std threshold ({STD_PERCENTILE}th percentile): {std_threshold:.5f}")
print(f"  Cortical vertices retained: {beta_std_mask[:N_CORTICAL].sum():,}")

template_any = CIFTI_TEMPLATE_PATTERN.format(sub_id=available_subjects[0])

if os.path.isfile(template_any):
    save_dscalar(
        beta_std_mask.astype(np.float32),
        os.path.join(BASE_OUTDIR, "beta_std_mask.dscalar.nii"),
        template_any,
        "beta_std_mask",
    )


# =============================================================================
# [5] FIT MODELS
# =============================================================================

all_processed = {model_key: [] for model_key in MODEL_SPECS}
all_failed = {model_key: [] for model_key in MODEL_SPECS}

outer_kf = KFold(
    n_splits=N_OUTER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

outer_splits = list(outer_kf.split(np.arange(N_CONDITIONS)))

for model_key, spec in MODEL_SPECS.items():

    if model_key not in SENSITIVITY_MODEL_KEYS:
        continue

    model_bands = spec["bands"]
    model_label = spec["label"]
    model_outdir = os.path.join(BASE_OUTDIR, model_key)
    os.makedirs(model_outdir, exist_ok=True)

    print(f"\n{'#' * 80}")
    print(f"MODEL: {model_key}")
    print(f"LABEL: {model_label}")
    print(f"BANDS: {model_bands}")
    print(f"OUTDIR: {model_outdir}")
    print(f"{'#' * 80}")

    np.save(
        os.path.join(model_outdir, "feature_spaces.npy"),
        np.array(model_bands, dtype=object),
    )

    with open(os.path.join(model_outdir, "model_description.txt"), "w") as f:
        f.write(f"Model key: {model_key}\n")
        f.write(f"Label: {model_label}\n")
        f.write(f"Bands: {model_bands}\n")
        f.write(f"Family: {spec.get('family', 'unspecified')}\n")
        f.write(f"Calorie target mode: {CALORIE_TARGET_MODE}\n")

    for sub_id in available_subjects:

        if sub_id not in SENSITIVITY_FIT_SUBJECTS:
            continue

        print(f"\n{'=' * 60}")
        print(f"  {model_key} | sub-{sub_id}")
        print(f"{'=' * 60}")

        sub_outdir = os.path.join(model_outdir, f"sub-{sub_id}")
        os.makedirs(sub_outdir, exist_ok=True)

        if SKIP_COMPLETED:
            completed, completion_reason = model_completed(model_outdir, sub_id, model_bands)
            if completed:
                print("  Existing complete output found — skipping fit.")
                all_processed[model_key].append(sub_id)
                continue
            else:
                print(f"  Existing output incomplete or absent — fitting. Reason: {completion_reason}")

        try:
            betas_avg, good_gray = cached_betas[sub_id]

            combined_mask = good_gray & beta_std_mask
            compact_keep_idx = np.where(combined_mask[good_gray])[0]

            Y = betas_avg[:, compact_keep_idx].astype(np.float32)
            Y -= Y.mean(axis=0, keepdims=True)

            full_cortical = np.zeros(N_GRAYORD, dtype=bool)
            full_cortical[:N_CORTICAL] = True

            cortical_compact = np.where(
                (full_cortical & combined_mask)[combined_mask]
            )[0]

            print(f"  Y        : {Y.shape}")
            print(f"  Cortical : {len(cortical_compact):,} vertices")

            deltas_per_fold = []
            r_joint_folds = []
            r_split_folds = []
            calorie_diag_rows = []

            for fold_idx, (outer_train, outer_test) in enumerate(outer_splits):

                feature_dict, cal_diag = build_fold_feature_dict(
                    sub_id=sub_id,
                    outer_train=outer_train,
                    outer_test=outer_test,
                )

                cal_diag["subject"] = sub_id
                cal_diag["fold"] = fold_idx + 1
                cal_diag["model_key"] = model_key
                calorie_diag_rows.append(cal_diag)

                X_all, band_slices, band_dims = build_design_matrix_and_slices(
                    feature_dict=feature_dict,
                    model_bands=model_bands,
                )

                X_tr = X_all[outer_train].astype(np.float32)
                X_te = X_all[outer_test].astype(np.float32)

                Y_tr = Y[outer_train]
                Y_te = Y[outer_test]

                inner_kf = KFold(
                    n_splits=N_INNER_FOLDS,
                    shuffle=True,
                    random_state=RANDOM_STATE + fold_idx,
                )

                inner_cv = list(inner_kf.split(np.arange(len(outer_train))))

                print(
                    f"  Fold {fold_idx + 1}/{N_OUTER_FOLDS} — random search ...",
                    end=" ",
                    flush=True,
                )

                pipe = make_pipeline_random(model_bands, band_slices, inner_cv)
                pipe.fit(X_tr, Y_tr)

                print("done.")

                initial_deltas = pipe.named_steps["multiplekernelridgecv"].deltas_

                if USE_HYPERGRADIENT:
                    print(
                        f"  Fold {fold_idx + 1}/{N_OUTER_FOLDS} — hyper-gradient ...",
                        end=" ",
                        flush=True,
                    )

                    try:
                        pipe_hg = make_pipeline_hypergradient(
                            model_bands=model_bands,
                            band_slices=band_slices,
                            inner_cv=inner_cv,
                            initial_deltas=initial_deltas,
                        )
                        pipe_hg.fit(X_tr, Y_tr)
                        pipe = pipe_hg
                        print("done.")
                    except Exception as e:
                        print(f"failed ({e}) — keeping random-search result")

                Y_pred_joint = backend.to_numpy(
                    pipe.predict(X_te)
                ).astype(np.float32)

                Y_pred_split = backend.to_numpy(
                    pipe.predict(X_te, split=True)
                ).astype(np.float32)

                fold_deltas = backend.to_numpy(
                    pipe.named_steps["multiplekernelridgecv"].deltas_
                ).astype(np.float32)

                r_joint = pearson_r_columns(Y_te, Y_pred_joint)

                r_split = backend.to_numpy(
                    correlation_score_split(Y_te, Y_pred_split)
                ).astype(np.float32)

                deltas_per_fold.append(fold_deltas)
                r_joint_folds.append(r_joint)
                r_split_folds.append(r_split)

                med_deltas = np.nanmedian(fold_deltas, axis=1).round(2).tolist()

                print(
                    f"    cortex sample r={np.nanmean(r_joint[cortical_compact[:200]]):+.4f} | "
                    f"deltas={med_deltas} | "
                    f"CLIP->cal r={cal_diag['calorie_pred_test_r']:+.3f}, "
                    f"R²={cal_diag['calorie_pred_test_r2']:+.3f}"
                )

                del X_all, X_tr, X_te, Y_tr, Y_te, pipe
                gc.collect()

            r_joint = np.nanmean(np.stack(r_joint_folds, axis=0), axis=0)
            r_split = np.nanmean(np.stack(r_split_folds, axis=0), axis=0)
            r2_joint = r_joint ** 2

            best_deltas = np.mean(np.stack(deltas_per_fold, axis=0), axis=0)

            safe = np.where(r_joint >= PERC_THRESHOLD, r_joint, np.nan)
            perc_r = r_split / safe[np.newaxis, :] * 100.0

            np.save(
                os.path.join(sub_outdir, "r_joint.npy"),
                expand(r_joint, combined_mask),
            )

            np.save(
                os.path.join(sub_outdir, "r2_joint.npy"),
                expand(r2_joint, combined_mask),
            )

            np.save(
                os.path.join(sub_outdir, "combined_mask.npy"),
                combined_mask,
            )

            np.save(
                os.path.join(sub_outdir, "deltas_folds.npy"),
                np.stack(deltas_per_fold, axis=0).astype(np.float32),
            )

            np.save(
                os.path.join(sub_outdir, "feature_spaces.npy"),
                np.array(model_bands, dtype=object),
            )

            pd.DataFrame(calorie_diag_rows).to_csv(
                os.path.join(sub_outdir, "calorie_clip_decomposition_folds.csv"),
                index=False,
            )

            for k, band in enumerate(model_bands):
                np.save(
                    os.path.join(sub_outdir, f"r_split_{band}.npy"),
                    expand(r_split[k], combined_mask),
                )

                np.save(
                    os.path.join(sub_outdir, f"perc_r_{band}.npy"),
                    expand(perc_r[k], combined_mask),
                )

                np.save(
                    os.path.join(sub_outdir, f"deltas_{band}.npy"),
                    expand(np.nanmedian(best_deltas[k]), combined_mask, fill=0),
                )

            template_path = CIFTI_TEMPLATE_PATTERN.format(sub_id=sub_id)

            if os.path.isfile(template_path):
                save_dscalar(
                    expand(r_joint, combined_mask),
                    os.path.join(sub_outdir, "r_joint.dscalar.nii"),
                    template_path,
                    "r_joint",
                )

                save_dscalar(
                    expand(r2_joint, combined_mask),
                    os.path.join(sub_outdir, "r2_joint.dscalar.nii"),
                    template_path,
                    "r2_joint",
                )

                for k, band in enumerate(model_bands):
                    save_dscalar(
                        expand(r_split[k], combined_mask),
                        os.path.join(sub_outdir, f"r_split_{band}.dscalar.nii"),
                        template_path,
                        f"r_split_{band}",
                    )

                    save_dscalar(
                        expand(perc_r[k], combined_mask),
                        os.path.join(sub_outdir, f"perc_r_{band}.dscalar.nii"),
                        template_path,
                        f"perc_r_{band}",
                    )
            else:
                print(f"  WARNING: CIFTI template not found: {template_path}")

            with open(os.path.join(sub_outdir, "DONE.txt"), "w") as f:
                f.write(f"Completed sub-{sub_id}\n")
                f.write(f"Model key: {model_key}\n")
                f.write(f"Model label: {model_label}\n")
                f.write(f"Bands: {model_bands}\n")
                f.write(f"Family: {spec.get('family', 'unspecified')}\n")
                f.write(f"Calorie target mode: {CALORIE_TARGET_MODE}\n")

            all_processed[model_key].append(sub_id)

            print(
                f"\n  {model_key} | sub-{sub_id} done — "
                f"r_joint cortex mean={np.nanmean(r_joint[cortical_compact]):+.5f}"
            )

            del Y, r_joint_folds, r_split_folds, deltas_per_fold
            gc.collect()

        except Exception as e:
            print(f"\n  ERROR {model_key} | sub-{sub_id}: {e}")
            import traceback
            traceback.print_exc()
            all_failed[model_key].append(sub_id)
            continue


# =============================================================================
# [6] GROUP MAPS
# =============================================================================

print(f"\n{'#' * 80}")
print("GROUP MAPS")
print(f"{'#' * 80}")

template_path = CIFTI_TEMPLATE_PATTERN.format(sub_id=available_subjects[0])

for model_key, spec in MODEL_SPECS.items():

    if model_key not in SENSITIVITY_MODEL_KEYS:
        continue

    model_bands = spec["bands"]
    model_outdir = os.path.join(BASE_OUTDIR, model_key)

    processed = sorted(set(all_processed[model_key]))

    print(f"\n{'=' * 60}")
    print(f"{model_key}: {len(processed)} subjects | failed: {all_failed[model_key]}")
    print(f"{'=' * 60}")

    map_names = (
        ["r_joint", "r2_joint"]
        + [f"r_split_{band}" for band in model_bands]
        + [f"perc_r_{band}" for band in model_bands]
    )

    for map_name in map_names:
        maps = []

        for sid in processed:
            fpath = os.path.join(model_outdir, f"sub-{sid}", f"{map_name}.npy")

            if os.path.isfile(fpath):
                arr = np.load(fpath).astype(np.float32)

                if arr.shape[0] == N_GRAYORD:
                    maps.append(arr)
                else:
                    print(f"  WARNING: {fpath} shape {arr.shape}; skipping")

        if not maps:
            print(f"  WARNING: no maps found for {map_name}")
            continue

        gm = np.nanmean(np.stack(maps, axis=0), axis=0).astype(np.float32)

        np.save(os.path.join(model_outdir, f"group_{map_name}.npy"), gm)

        if os.path.isfile(template_path):
            save_dscalar(
                gm,
                os.path.join(model_outdir, f"group_{map_name}.dscalar.nii"),
                template_path,
                f"group_{map_name}",
            )

        print(f"  {map_name:35s}: cortex mean={np.nanmean(gm[:N_CORTICAL]):+.5f}")


# =============================================================================
# [7] CALORIE-DECOMPOSITION DIAGNOSTIC SUMMARY
# =============================================================================

print(f"\n{'#' * 80}")
print("CALORIE CLIP-DECOMPOSITION DIAGNOSTICS")
print(f"{'#' * 80}")

diag_rows = []

for model_key, spec in MODEL_SPECS.items():
    if model_key not in SENSITIVITY_MODEL_KEYS:
        continue
    model_outdir = os.path.join(BASE_OUTDIR, model_key)

    for sid in sorted(set(all_processed[model_key])):
        p = os.path.join(
            model_outdir,
            f"sub-{sid}",
            "calorie_clip_decomposition_folds.csv",
        )

        if os.path.isfile(p):
            tmp = pd.read_csv(p)
            diag_rows.append(tmp)

if diag_rows:
    diag_df = pd.concat(diag_rows, ignore_index=True)

    diag_out = os.path.join(BASE_OUTDIR, "all_models_calorie_clip_decomposition_folds.csv")
    diag_df.to_csv(diag_out, index=False)

    print(f"Saved: {diag_out}")

    print("\nFold-level CLIP -> perceived-calorie prediction:")
    print(
        diag_df[
            [
                "calorie_pred_test_r2",
                "calorie_pred_test_r",
                "calorie_pred_alpha",
                "calorie_pred_train_var",
                "calorie_res_train_var",
            ]
        ]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .to_string()
    )
else:
    print("No diagnostic files found.")


# =============================================================================
# [8] FINAL SUMMARY
# =============================================================================

print(f"\n{'#' * 80}")
print("SUMMARY")
print(f"{'#' * 80}")

for model_key in MODEL_SPECS:
    if model_key not in SENSITIVITY_MODEL_KEYS:
        continue
    print(
        f"{model_key:25s}: "
        f"processed={len(set(all_processed[model_key])):2d}, "
        f"failed={all_failed[model_key]}"
    )

print(f"\nResults saved to:\n{BASE_OUTDIR}")
print("Done.")
