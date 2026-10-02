# -*- coding: utf-8 -*-

"""
Created on Mon Jun 15 08:54:48 2026

@author: G.Marrazzo
"""

r"""
NeuroRepFood — ResCLIP reliability and richer-CLIP recovery diagnostic
=====================================================================

Script filename
---------------
resclip_reliability_and_clip_recovery_diagnostic.py

Purpose
-------
Characterize the existing CalorieResCLIP component used in the NeuroRepFood
encoding-model analysis.

This script performs:

Step 0. Reliability gate
    Estimate whether the CLIP-residual calorie component is reliable across
    independent splits of the behavioral rating subjects.

    For each random split of rating subjects:
        1. compute consensus perceived calorie separately in split A and split B
        2. z-score each split consensus target across stimuli
        3. predict each split target from the same CLIP representation using the
           same CV ridge procedure as plot_residual_calorie_diagnostic.py
        4. compute residuals
        5. correlate residual A and residual B across the 96 stimuli

    Outputs:
        resclip_step0_reliability_iterations.csv
        resclip_step0_reliability_summary.csv

Step 1. Richer-CLIP recovery diagnostic
    Hold the final existing CalorieResCLIP vector fixed and test whether it can
    be predicted from richer or alternative CLIP feature representations.

    Models:
        - original 50-D CLIP ridge
        - nonlinear RBF kernel ridge on original 50-D CLIP
        - full 512-D CLIP ridge, if CLIP_full512.npy exists
        - discarded full-512 PCA dimensions, if CLIP_full512.npy exists

    Outputs:
        resclip_step1_clip_recovery_summary.csv
        resclip_step1_predictions_by_item.csv
        resclip_step1_outer_fold_params.csv
        resclip_step1_permutation_distribution.csv
        figures/*.png/.pdf
        resclip_reliability_and_clip_recovery_diagnostic_METADATA.md

Important interpretation
------------------------
If Step 0 shows near-zero residual reliability, do not interpret ResCLIP
substantively. In that case, Step 1/2/3 characterization is bounded by a weak
target reliability.

If richer CLIP predicts ResCLIP above chance, the residual is not necessarily
beyond CLIP; it is beyond the original linear 50-D CLIP decomposition.
"""

# =============================================================================
# [0] IMPORTS
# =============================================================================

import os
import json
import platform
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import pearsonr, spearmanr

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.decomposition import PCA

print("Imports OK.")


# =============================================================================
# [1] CONFIG
# =============================================================================

# RELEASE PATH ADAPTATION ONLY.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

FEAT_DIR = os.path.join(REPO_ROOT, "reproduced_outputs", "feature_extraction")
BAND_DIR = os.path.join(REPO_ROOT, "reproduced_outputs", "feature_bands")

ORIGINAL_DIAG_DIR = os.path.join(
    REPO_ROOT,
    "reproduced_outputs",
    "diagnostics",
    "perceived_calorie_prediction_diagnostics",
)

OUTDIR = os.path.join(
    REPO_ROOT,
    "reproduced_outputs",
    "diagnostics",
    "resclip_reliability_and_recovery_diagnostics",
)
FIGDIR = os.path.join(OUTDIR, "figures")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

N_CONDITIONS = 96
RANDOM_STATE = 42

# Must match plot_residual_calorie_diagnostic.py
ALPHAS = np.logspace(-4, 4, 25)
N_SPLITS_ORIGINAL = 5

# Step 0 reliability
N_RELIABILITY_SPLITS = 1000
MIN_VALID_ITEMS_PER_RATING_SUBJECT = 90
MIN_VALID_ITEMS_PER_SPLIT_TARGET = 90

# Step 1 recovery
N_OUTER_SPLITS = 5
N_INNER_SPLITS_KERNEL = 5

# Permutations. Increase to 1000+ for final if runtime is acceptable.
N_PERMUTATIONS_LINEAR = 1000
N_PERMUTATIONS_KERNEL = 0

RUN_KERNEL_RIDGE = True
RUN_PERMUTATIONS = True

# For discarded-PC test from full 512-D CLIP.
# With 5-fold outer CV and 96 stimuli, the training fold has ~76 items,
# so the largest available training-set PCA index is about 75.
DISCARDED_PC_START_0BASED = 50      # PC51
DISCARDED_PC_STOP_0BASED = 95       # exclusive; truncated inside each fold if needed

SAVE_PNG = True
SAVE_PDF = True
DPI = 300

# Excel-friendly duplicate CSVs avoid European locale issues.
SAVE_EXCEL_FRIENDLY_CSV = True
SAVE_XLSX = True


# =============================================================================
# [2] PATHS
# =============================================================================

PATHS = {
    "calorie_persubject": os.path.join(FEAT_DIR, "Calorie_persubject.npy"),
    "clip_50": os.path.join(FEAT_DIR, "CLIP.npy"),
    "clip_50_fallback_band": os.path.join(BAND_DIR, "band_CLIP.npy"),
    "calorie_res_final": os.path.join(ORIGINAL_DIAG_DIR, "calorie_res_cv_CLIP.npy"),
    "calorie_pred_final": os.path.join(ORIGINAL_DIAG_DIR, "calorie_pred_cv_CLIP.npy"),
    "calorie_raw_final": os.path.join(ORIGINAL_DIAG_DIR, "calorie_group_raw_z.npy"),
    "ordered_stimuli": os.path.join(REPO_ROOT, "data", "stimuli", "ordered_stimuli.csv"),
}

CLIP_FULL512_CANDIDATES = [
    os.path.join(FEAT_DIR, "CLIP_full512.npy"),
    os.path.join(FEAT_DIR, "CLIP_ViTB32_full512.npy"),
    os.path.join(FEAT_DIR, "clip_full512.npy"),
    os.path.join(BAND_DIR, "CLIP_full512.npy"),
]


# =============================================================================
# [3] I/O HELPERS
# =============================================================================

def load_required_npy(path, name):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing required file for {name}: {path}")
    arr = np.load(path).astype(np.float32)
    return np.asarray(arr).squeeze()


def load_optional_npy(path):
    if os.path.isfile(path):
        return np.asarray(np.load(path).astype(np.float32)).squeeze()
    return None


def find_first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def save_table(df, path_no_ext):
    """
    Save table as:
        .csv       standard comma/dot CSV for code
        .excel.csv semicolon/comma CSV for European Excel
        .xlsx      Excel workbook, if available
    """
    csv_path = f"{path_no_ext}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  saved: {csv_path}")

    if SAVE_EXCEL_FRIENDLY_CSV:
        excel_csv_path = f"{path_no_ext}.excel.csv"
        df.to_csv(
            excel_csv_path,
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
            float_format="%.8g",
        )
        print(f"  saved: {excel_csv_path}")

    if SAVE_XLSX:
        try:
            xlsx_path = f"{path_no_ext}.xlsx"
            df.to_excel(xlsx_path, index=False)
            print(f"  saved: {xlsx_path}")
        except Exception as e:
            print(f"  NOTE: could not save XLSX for {path_no_ext}: {e}")


def savefig(fig, name):
    if SAVE_PNG:
        path = os.path.join(FIGDIR, f"{name}.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  saved: {path}")

    if SAVE_PDF:
        path = os.path.join(FIGDIR, f"{name}.pdf")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  saved: {path}")


# =============================================================================
# [4] NUMERIC HELPERS
# =============================================================================

def zscore_nan(y):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.nanmean(y)
    s = np.nanstd(y)
    if not np.isfinite(s) or s < 1e-12:
        return np.full_like(y, np.nan, dtype=np.float64)
    return ((y - m) / s).astype(np.float64)


def safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return np.nan
    if np.nanstd(x[good]) < 1e-12 or np.nanstd(y[good]) < 1e-12:
        return np.nan
    return float(pearsonr(x[good], y[good])[0])


def safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return np.nan
    if np.nanstd(x[good]) < 1e-12 or np.nanstd(y[good]) < 1e-12:
        return np.nan
    return float(spearmanr(x[good], y[good]).correlation)


def cv_r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    good = np.isfinite(y_true) & np.isfinite(y_pred)
    if good.sum() < 3:
        return np.nan
    ss_res = np.nansum((y_true[good] - y_pred[good]) ** 2)
    ss_tot = np.nansum((y_true[good] - np.nanmean(y_true[good])) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def sem(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if x.size == 1:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def ci95(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return float(x[0]), float(x[0])
    m = float(np.mean(x))
    s = sem(x)
    tcrit = stats.t.ppf(0.975, df=x.size - 1)
    return float(m - tcrit * s), float(m + tcrit * s)


def fisher_mean_r(rs):
    rs = np.asarray(rs, dtype=np.float64)
    rs = rs[np.isfinite(rs)]
    if rs.size == 0:
        return np.nan
    rs = np.clip(rs, -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(rs))))


def spearman_brown(r):
    if not np.isfinite(r):
        return np.nan
    if abs(1.0 + r) < 1e-12:
        return np.nan
    return float((2.0 * r) / (1.0 + r))


def reliability_ceiling_from_split_half(r_split):
    """
    Given a split-half correlation, return:
        r_SB = Spearman-Brown reliability
        corr_ceiling = sqrt(max(r_SB, 0))
        r2_ceiling = max(r_SB, 0)

    If a target has reliability r_SB, the maximum expected correlation with a
    noiseless external predictor is approximately sqrt(r_SB), while the maximum
    explainable variance is approximately r_SB.
    """
    r_sb = spearman_brown(r_split)
    if not np.isfinite(r_sb):
        return np.nan, np.nan, np.nan
    r2_ceiling = max(r_sb, 0.0)
    corr_ceiling = float(np.sqrt(r2_ceiling))
    return float(r_sb), corr_ceiling, float(r2_ceiling)


def nice_ylim(vals, include_zero=True, pad_fraction=0.18, min_span=0.05):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        lo, hi = -min_span, min_span
    else:
        lo, hi = float(np.min(vals)), float(np.max(vals))
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    span = max(hi - lo, min_span)
    pad = pad_fraction * span
    return lo - pad, hi + pad


# =============================================================================
# [5] MODEL HELPERS — MATCH ORIGINAL DECOMPOSITION WHERE POSSIBLE
# =============================================================================

def make_ridge():
    """
    Match plot_residual_calorie_diagnostic.py:
        StandardScaler(with_mean=True, with_std=True)
        RidgeCV(alphas=np.logspace(-4,4,25))
    """
    return make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        RidgeCV(alphas=ALPHAS),
    )


def get_pipeline_param_summary(est):
    """
    Extract fitted hyperparameters from an estimator/pipeline.
    """
    out = {}

    if isinstance(est, Pipeline):
        for name, step in est.named_steps.items():
            if hasattr(step, "alpha_"):
                out[f"{name}__alpha"] = float(step.alpha_)
            if hasattr(step, "best_params_"):
                for k, v in step.best_params_.items():
                    out[f"{name}__{k}"] = v
    else:
        if hasattr(est, "alpha_"):
            out["alpha"] = float(est.alpha_)
        if hasattr(est, "best_params_"):
            out.update(est.best_params_)

    return out


def cv_predict_manual(X, y, estimator_factory, n_splits=5, random_state=RANDOM_STATE):
    """
    Manual cross-validated prediction so we can record fold parameters.

    If y contains NaNs, CV is performed only on valid target rows.
    The returned prediction vector has length N with NaNs for invalid items.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X rows {X.shape[0]} != y length {y.shape[0]}")

    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    idx = np.where(valid)[0]

    pred = np.full(y.shape[0], np.nan, dtype=np.float64)
    fold_rows = []

    if idx.size < max(n_splits, 3):
        raise ValueError(f"Too few valid items for CV: {idx.size}")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_i, (train_rel, test_rel) in enumerate(cv.split(idx), start=1):
        train_idx = idx[train_rel]
        test_idx = idx[test_rel]

        est = estimator_factory()
        est.fit(X[train_idx], y[train_idx])
        pred[test_idx] = np.asarray(est.predict(X[test_idx])).reshape(-1)

        params = get_pipeline_param_summary(est)
        fold_row = {
            "fold": fold_i,
            "n_train": int(train_idx.size),
            "n_test": int(test_idx.size),
        }
        fold_row.update(params)
        fold_rows.append(fold_row)

    metrics = {
        "n_items": int(idx.size),
        "cv_r2": cv_r2_score(y, pred),
        "pearson_r": safe_pearson(y, pred),
        "spearman_rho": safe_spearman(y, pred),
        "target_var": float(np.nanvar(y[idx])),
        "pred_var": float(np.nanvar(pred[idx])),
        "residual_var": float(np.nanvar(y[idx] - pred[idx])),
        "residual_var_fraction": float(
            np.nanvar(y[idx] - pred[idx]) / (np.nanvar(y[idx]) + 1e-12)
        ),
    }

    return pred.astype(np.float32), pd.DataFrame(fold_rows), metrics


def compute_clip_residual_for_target(y, X_clip):
    """
    Given an item-level calorie target, reproduce the original diagnostic
    CLIP residualization:
        z-score target
        5-fold CV ridge prediction from CLIP
        residual = y_z - predicted_y_z
    """
    yz = zscore_nan(y)

    pred, fold_df, metrics = cv_predict_manual(
        X=X_clip,
        y=yz,
        estimator_factory=make_ridge,
        n_splits=N_SPLITS_ORIGINAL,
        random_state=RANDOM_STATE,
    )
    res = yz - pred
    return pred, res, fold_df, metrics


# =============================================================================
# [6] PCA-SLICE TRANSFORMER FOR DISCARDED FULL-CLIP PCs
# =============================================================================

class PCAComponentSlice(BaseEstimator, TransformerMixin):
    """
    Fit PCA on the training data and return only a selected component range.

    start and stop are 0-based indices.
    Example:
        start=50, stop=95 returns PC51..PC95, truncated to the maximum
        possible number of training-set components.

    This avoids fitting PCA on the full dataset before CV.
    """

    def __init__(self, start=50, stop=95, random_state=42):
        self.start = int(start)
        self.stop = int(stop)
        self.random_state = int(random_state)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        n_max = min(X.shape[0], X.shape[1], self.stop)

        if n_max <= self.start:
            raise ValueError(
                f"Cannot select PCs {self.start}:{self.stop}; "
                f"training data only allows {n_max} components."
            )

        self.n_components_ = int(n_max)
        self.actual_start_ = int(self.start)
        self.actual_stop_ = int(n_max)

        self.pca_ = PCA(
            n_components=self.n_components_,
            svd_solver="full",
            random_state=self.random_state,
        )
        self.pca_.fit(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        Z = self.pca_.transform(X)
        return Z[:, self.actual_start_:self.actual_stop_].astype(np.float32)


def make_discarded_pc_ridge():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "pca_slice",
            PCAComponentSlice(
                start=DISCARDED_PC_START_0BASED,
                stop=DISCARDED_PC_STOP_0BASED,
                random_state=RANDOM_STATE,
            ),
        ),
        ("ridgecv", RidgeCV(alphas=ALPHAS)),
    ])


def make_kernel_ridge_rbf():
    """
    Nonlinear readout on the same CLIP features.

    GridSearchCV is inside each outer fold, so this is nested relative to the
    outer CV prediction.
    """
    inner_cv = KFold(
        n_splits=N_INNER_SPLITS_KERNEL,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("kernelridge", KernelRidge(kernel="rbf")),
    ])

    param_grid = {
        "kernelridge__alpha": np.logspace(-3, 3, 13),
        "kernelridge__gamma": np.logspace(-3, 1, 9),
    }

    return GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
    )


# =============================================================================
# [7] LOAD DATA
# =============================================================================

def load_inputs():
    print("\n[1] Loading inputs")

    calorie_sub = load_required_npy(
        PATHS["calorie_persubject"],
        "Calorie_persubject",
    )
    if calorie_sub.ndim != 2 or calorie_sub.shape[0] != N_CONDITIONS:
        raise ValueError(
            f"Calorie_persubject must be {N_CONDITIONS} x subjects, "
            f"got {calorie_sub.shape}"
        )

    clip_path = PATHS["clip_50"]
    if not os.path.isfile(clip_path):
        clip_path = PATHS["clip_50_fallback_band"]

    clip_50 = load_required_npy(clip_path, "CLIP 50-D")
    if clip_50.ndim != 2 or clip_50.shape[0] != N_CONDITIONS:
        raise ValueError(f"CLIP 50-D has invalid shape: {clip_50.shape}")

    final_res = load_required_npy(
        PATHS["calorie_res_final"],
        "final calorie_res_cv_CLIP",
    )
    final_pred = load_optional_npy(PATHS["calorie_pred_final"])
    final_raw = load_optional_npy(PATHS["calorie_raw_final"])

    if final_res.shape[0] != N_CONDITIONS:
        raise ValueError(f"final ResCLIP has invalid shape: {final_res.shape}")

    full512_path = find_first_existing(CLIP_FULL512_CANDIDATES)
    clip_512 = None
    if full512_path is not None:
        clip_512 = load_optional_npy(full512_path)
        if clip_512.ndim != 2 or clip_512.shape[0] != N_CONDITIONS:
            print(f"  WARNING: found full512 file but invalid shape: {clip_512.shape}")
            clip_512 = None
            full512_path = None

    ordered_stimuli = None
    if os.path.isfile(PATHS["ordered_stimuli"]):
        try:
            ordered_stimuli = pd.read_csv(PATHS["ordered_stimuli"])
        except Exception as e:
            print(f"  NOTE: could not read ordered_stimuli.csv: {e}")

    print(f"  Calorie_persubject : {calorie_sub.shape} | {PATHS['calorie_persubject']}")
    print(f"  CLIP 50-D          : {clip_50.shape} | {clip_path}")
    print(f"  final ResCLIP      : {final_res.shape} | {PATHS['calorie_res_final']}")
    print(f"  CLIP full512       : {None if clip_512 is None else clip_512.shape} | {full512_path}")

    return {
        "calorie_sub": calorie_sub.astype(np.float32),
        "clip_50": clip_50.astype(np.float32),
        "clip_50_path": clip_path,
        "final_res": final_res.astype(np.float32).reshape(-1),
        "final_pred": None if final_pred is None else final_pred.astype(np.float32).reshape(-1),
        "final_raw": None if final_raw is None else final_raw.astype(np.float32).reshape(-1),
        "clip_512": None if clip_512 is None else clip_512.astype(np.float32),
        "clip_512_path": full512_path,
        "ordered_stimuli": ordered_stimuli,
    }


# =============================================================================
# [8] STEP 0 — RESCLIP RELIABILITY
# =============================================================================

def valid_rating_subject_columns(calorie_sub):
    valid_cols = []
    for j in range(calorie_sub.shape[1]):
        y = calorie_sub[:, j]
        n_valid = int(np.isfinite(y).sum())
        sd = float(np.nanstd(y)) if n_valid > 1 else np.nan
        if n_valid >= MIN_VALID_ITEMS_PER_RATING_SUBJECT and np.isfinite(sd) and sd > 1e-8:
            valid_cols.append(j)
    return np.asarray(valid_cols, dtype=int)


def consensus_from_subject_columns(calorie_sub, cols):
    y = np.nanmean(calorie_sub[:, cols], axis=1)
    return y.astype(np.float64)


def run_step0_reliability(calorie_sub, clip_50):
    print("\n[2] Step 0 — ResCLIP split-half reliability")

    rng = np.random.default_rng(RANDOM_STATE)

    valid_cols = valid_rating_subject_columns(calorie_sub)
    if valid_cols.size < 4:
        raise RuntimeError(
            f"Too few valid rating subjects for split-half reliability: {valid_cols.size}"
        )

    rows = []

    for it in range(1, N_RELIABILITY_SPLITS + 1):
        perm = rng.permutation(valid_cols)
        n_a = perm.size // 2
        cols_a = perm[:n_a]
        cols_b = perm[n_a:]

        y_a = consensus_from_subject_columns(calorie_sub, cols_a)
        y_b = consensus_from_subject_columns(calorie_sub, cols_b)

        valid_items = np.isfinite(y_a) & np.isfinite(y_b)
        if int(valid_items.sum()) < MIN_VALID_ITEMS_PER_SPLIT_TARGET:
            rows.append({
                "iteration": it,
                "n_subjects_A": int(cols_a.size),
                "n_subjects_B": int(cols_b.size),
                "n_valid_items": int(valid_items.sum()),
                "skipped": True,
            })
            continue

        # Raw calorie split reliability
        raw_rho = safe_spearman(y_a, y_b)
        raw_pearson = safe_pearson(y_a, y_b)

        # Recompute CLIP residual independently for each split target.
        _, res_a, _, metrics_a = compute_clip_residual_for_target(y_a, clip_50)
        _, res_b, _, metrics_b = compute_clip_residual_for_target(y_b, clip_50)

        res_rho = safe_spearman(res_a, res_b)
        res_pearson = safe_pearson(res_a, res_b)

        res_rsb, res_corr_ceiling, res_r2_ceiling = reliability_ceiling_from_split_half(res_rho)
        raw_rsb, raw_corr_ceiling, raw_r2_ceiling = reliability_ceiling_from_split_half(raw_rho)

        rows.append({
            "iteration": it,
            "n_subjects_A": int(cols_a.size),
            "n_subjects_B": int(cols_b.size),
            "n_valid_items": int(valid_items.sum()),
            "skipped": False,

            "raw_calorie_split_spearman": raw_rho,
            "raw_calorie_split_pearson": raw_pearson,
            "raw_calorie_spearman_brown": raw_rsb,
            "raw_calorie_corr_ceiling": raw_corr_ceiling,
            "raw_calorie_r2_ceiling": raw_r2_ceiling,

            "resclip_split_spearman": res_rho,
            "resclip_split_pearson": res_pearson,
            "resclip_spearman_brown": res_rsb,
            "resclip_corr_ceiling": res_corr_ceiling,
            "resclip_r2_ceiling": res_r2_ceiling,

            "split_A_clip_cv_r2": metrics_a["cv_r2"],
            "split_B_clip_cv_r2": metrics_b["cv_r2"],
            "split_A_clip_spearman": metrics_a["spearman_rho"],
            "split_B_clip_spearman": metrics_b["spearman_rho"],
        })

        if it % 100 == 0:
            print(f"  completed {it}/{N_RELIABILITY_SPLITS} splits")

    iter_df = pd.DataFrame(rows)

    valid = iter_df[~iter_df["skipped"].astype(bool)].copy()

    def summarize_col(col):
        vals = valid[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        lo, hi = ci95(vals)
        return {
            f"{col}_n": int(vals.size),
            f"{col}_mean": float(np.nanmean(vals)) if vals.size else np.nan,
            f"{col}_sem": sem(vals),
            f"{col}_ci95_lo": lo,
            f"{col}_ci95_hi": hi,
            f"{col}_median": float(np.nanmedian(vals)) if vals.size else np.nan,
            f"{col}_fisher_mean": fisher_mean_r(vals) if "spearman" in col or "pearson" in col else np.nan,
        }

    summary = {
        "n_rating_subject_columns_valid": int(valid_cols.size),
        "n_iterations_requested": int(N_RELIABILITY_SPLITS),
        "n_iterations_valid": int(len(valid)),
        "n_iterations_skipped": int(iter_df["skipped"].sum()),
        "clip_feature_source": "CLIP_50D",
        "ridge_alphas": json.dumps(ALPHAS.tolist()),
        "outer_cv_n_splits": int(N_SPLITS_ORIGINAL),
        "outer_cv_shuffle": True,
        "outer_cv_random_state": int(RANDOM_STATE),
    }

    for col in [
        "raw_calorie_split_spearman",
        "raw_calorie_spearman_brown",
        "raw_calorie_corr_ceiling",
        "raw_calorie_r2_ceiling",
        "resclip_split_spearman",
        "resclip_spearman_brown",
        "resclip_corr_ceiling",
        "resclip_r2_ceiling",
        "split_A_clip_cv_r2",
        "split_B_clip_cv_r2",
    ]:
        summary.update(summarize_col(col))

    summary_df = pd.DataFrame([summary])

    save_table(
        iter_df,
        os.path.join(OUTDIR, "resclip_step0_reliability_iterations"),
    )
    save_table(
        summary_df,
        os.path.join(OUTDIR, "resclip_step0_reliability_summary"),
    )

    print("\nStep 0 key result:")
    print(
        summary_df[
            [
                "resclip_split_spearman_mean",
                "resclip_spearman_brown_mean",
                "resclip_corr_ceiling_mean",
                "resclip_r2_ceiling_mean",
            ]
        ].round(4).to_string(index=False)
    )

    return iter_df, summary_df


# =============================================================================
# [9] STEP 1 — RICHER CLIP RECOVERY OF FINAL RESCLIP
# =============================================================================

def build_step1_model_registry(clip_512_available):
    models = OrderedDict()

    models["CLIP50_linear_ridge"] = {
        "label": "Original 50-D CLIP ridge",
        "feature_key": "clip_50",
        "estimator_factory": make_ridge,
        "n_permutations": N_PERMUTATIONS_LINEAR,
    }

    if RUN_KERNEL_RIDGE:
        models["CLIP50_rbf_kernel_ridge"] = {
            "label": "Original 50-D CLIP RBF kernel ridge",
            "feature_key": "clip_50",
            "estimator_factory": make_kernel_ridge_rbf,
            "n_permutations": N_PERMUTATIONS_KERNEL,
        }

    if clip_512_available:
        models["CLIP512_linear_ridge"] = {
            "label": "Full 512-D CLIP ridge",
            "feature_key": "clip_512",
            "estimator_factory": make_ridge,
            "n_permutations": N_PERMUTATIONS_LINEAR,
        }

        models["CLIP512_discarded_PC51plus_ridge"] = {
            "label": "Full 512-D CLIP discarded PCs ridge",
            "feature_key": "clip_512",
            "estimator_factory": make_discarded_pc_ridge,
            "n_permutations": N_PERMUTATIONS_LINEAR,
        }

    return models


def permutation_test_recovery(
    X,
    y,
    estimator_factory,
    observed_r2,
    observed_spearman,
    n_permutations,
    model_name,
):
    if not RUN_PERMUTATIONS or n_permutations <= 0:
        return pd.DataFrame(), np.nan, np.nan

    rng = np.random.default_rng(RANDOM_STATE + 1000 + abs(hash(model_name)) % 100000)
    rows = []

    valid = np.isfinite(y)
    y_valid_idx = np.where(valid)[0]

    for p_i in range(1, n_permutations + 1):
        y_perm = np.asarray(y, dtype=np.float64).copy()
        y_perm[y_valid_idx] = rng.permutation(y_perm[y_valid_idx])

        try:
            pred_perm, _, metrics_perm = cv_predict_manual(
                X=X,
                y=y_perm,
                estimator_factory=estimator_factory,
                n_splits=N_OUTER_SPLITS,
                random_state=RANDOM_STATE,
            )

            rows.append({
                "model": model_name,
                "permutation": p_i,
                "cv_r2": metrics_perm["cv_r2"],
                "spearman_rho": metrics_perm["spearman_rho"],
                "pearson_r": metrics_perm["pearson_r"],
            })

        except Exception as e:
            rows.append({
                "model": model_name,
                "permutation": p_i,
                "cv_r2": np.nan,
                "spearman_rho": np.nan,
                "pearson_r": np.nan,
                "error": str(e),
            })

        if p_i % 100 == 0:
            print(f"    permutation {p_i}/{n_permutations}")

    perm_df = pd.DataFrame(rows)

    perm_r2 = perm_df["cv_r2"].to_numpy(dtype=float)
    perm_s = perm_df["spearman_rho"].to_numpy(dtype=float)

    p_r2 = float((np.sum(perm_r2 >= observed_r2) + 1) / (np.isfinite(perm_r2).sum() + 1))
    p_s = float((np.sum(perm_s >= observed_spearman) + 1) / (np.isfinite(perm_s).sum() + 1))

    return perm_df, p_r2, p_s


def run_step1_clip_recovery(inputs, step0_summary_df):
    print("\n[3] Step 1 — Predict final ResCLIP from richer CLIP features")

    final_res = inputs["final_res"].astype(np.float64)

    # Fixed target: the existing ResCLIP vector used downstream.
    # Z-scoring only changes scale, not rank/correlation interpretation.
    y = zscore_nan(final_res)

    clip_512_available = inputs["clip_512"] is not None
    registry = build_step1_model_registry(clip_512_available)

    # Ceiling from Step 0.
    if step0_summary_df is not None and not step0_summary_df.empty:
        corr_ceiling = float(step0_summary_df["resclip_corr_ceiling_mean"].iloc[0])
        r2_ceiling = float(step0_summary_df["resclip_r2_ceiling_mean"].iloc[0])
    else:
        corr_ceiling = np.nan
        r2_ceiling = np.nan

    summary_rows = []
    prediction_df = pd.DataFrame({
        "stimulus_index_0based": np.arange(N_CONDITIONS, dtype=int),
        "stimulus_index_1based": np.arange(1, N_CONDITIONS + 1, dtype=int),
        "resclip_target_z": y,
        "resclip_target_raw": final_res,
    })

    # Add stimulus metadata, if present.
    if inputs["ordered_stimuli"] is not None:
        stim_df = inputs["ordered_stimuli"].copy()
        if len(stim_df) == N_CONDITIONS:
            for col in stim_df.columns:
                prediction_df[f"stimulus_{col}"] = stim_df[col].values

    fold_param_rows = []
    perm_dfs = []

    for model_name, spec in registry.items():
        print(f"\n  Model: {model_name} — {spec['label']}")

        X = inputs[spec["feature_key"]]
        estimator_factory = spec["estimator_factory"]

        try:
            pred, fold_df, metrics = cv_predict_manual(
                X=X,
                y=y,
                estimator_factory=estimator_factory,
                n_splits=N_OUTER_SPLITS,
                random_state=RANDOM_STATE,
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            summary_rows.append({
                "model": model_name,
                "label": spec["label"],
                "feature_key": spec["feature_key"],
                "n_features": int(X.shape[1]),
                "error": str(e),
            })
            continue

        prediction_df[f"pred_{model_name}"] = pred
        prediction_df[f"residual_after_{model_name}"] = y - pred

        fold_df.insert(0, "model", model_name)
        fold_df.insert(1, "label", spec["label"])
        fold_param_rows.append(fold_df)

        print(
            f"    CV R²={metrics['cv_r2']:+.4f}, "
            f"Pearson r={metrics['pearson_r']:+.4f}, "
            f"Spearman rho={metrics['spearman_rho']:+.4f}"
        )

        if RUN_PERMUTATIONS:
            print(f"    Permutation test, n={spec['n_permutations']}")
            perm_df, p_r2, p_s = permutation_test_recovery(
                X=X,
                y=y,
                estimator_factory=estimator_factory,
                observed_r2=metrics["cv_r2"],
                observed_spearman=metrics["spearman_rho"],
                n_permutations=spec["n_permutations"],
                model_name=model_name,
            )
            if not perm_df.empty:
                perm_dfs.append(perm_df)
        else:
            p_r2, p_s = np.nan, np.nan

        r_fraction = (
            metrics["pearson_r"] / corr_ceiling
            if np.isfinite(metrics["pearson_r"]) and np.isfinite(corr_ceiling) and corr_ceiling > 0
            else np.nan
        )
        rho_fraction = (
            metrics["spearman_rho"] / corr_ceiling
            if np.isfinite(metrics["spearman_rho"]) and np.isfinite(corr_ceiling) and corr_ceiling > 0
            else np.nan
        )
        r2_fraction = (
            max(metrics["cv_r2"], 0.0) / r2_ceiling
            if np.isfinite(metrics["cv_r2"]) and np.isfinite(r2_ceiling) and r2_ceiling > 0
            else np.nan
        )

        summary_rows.append({
            "model": model_name,
            "label": spec["label"],
            "feature_key": spec["feature_key"],
            "n_features": int(X.shape[1]),
            "target": "final calorie_res_cv_CLIP.npy, z-scored",
            **metrics,
            "p_perm_cv_r2": p_r2,
            "p_perm_spearman": p_s,
            "resclip_corr_ceiling_from_step0": corr_ceiling,
            "resclip_r2_ceiling_from_step0": r2_ceiling,
            "pearson_fraction_of_corr_ceiling": r_fraction,
            "spearman_fraction_of_corr_ceiling": rho_fraction,
            "positive_cv_r2_fraction_of_r2_ceiling": r2_fraction,
            "n_permutations": int(spec["n_permutations"]) if RUN_PERMUTATIONS else 0,
        })

    summary_df = pd.DataFrame(summary_rows)
    fold_params_df = (
        pd.concat(fold_param_rows, ignore_index=True)
        if fold_param_rows
        else pd.DataFrame()
    )
    perm_df_all = (
        pd.concat(perm_dfs, ignore_index=True)
        if perm_dfs
        else pd.DataFrame()
    )

    save_table(
        summary_df,
        os.path.join(OUTDIR, "resclip_step1_clip_recovery_summary"),
    )
    save_table(
        prediction_df,
        os.path.join(OUTDIR, "resclip_step1_predictions_by_item"),
    )

    if not fold_params_df.empty:
        save_table(
            fold_params_df,
            os.path.join(OUTDIR, "resclip_step1_outer_fold_params"),
        )

    if not perm_df_all.empty:
        save_table(
            perm_df_all,
            os.path.join(OUTDIR, "resclip_step1_permutation_distribution"),
        )

    print("\nStep 1 summary:")
    cols = [
        "model", "cv_r2", "pearson_r", "spearman_rho",
        "p_perm_cv_r2", "p_perm_spearman",
        "positive_cv_r2_fraction_of_r2_ceiling",
    ]
    print(summary_df[cols].round(4).to_string(index=False))

    return summary_df, prediction_df, fold_params_df, perm_df_all


# =============================================================================
# [10] PLOTS
# =============================================================================

def plot_step0_reliability(iter_df, summary_df):
    valid = iter_df[~iter_df["skipped"].astype(bool)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.hist(valid["resclip_split_spearman"].dropna().values, bins=30, alpha=0.85)
    mean_r = float(summary_df["resclip_split_spearman_mean"].iloc[0])
    ax.axvline(mean_r, linestyle="--", linewidth=1.5)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("Split-half Spearman rho")
    ax.set_ylabel("Iterations")
    ax.set_title("ResCLIP split-half reliability")

    ax = axes[1]
    vals = [
        float(summary_df["raw_calorie_spearman_brown_mean"].iloc[0]),
        float(summary_df["resclip_spearman_brown_mean"].iloc[0]),
        float(summary_df["resclip_corr_ceiling_mean"].iloc[0]),
    ]
    labels = [
        "Raw calorie\nSB reliability",
        "ResCLIP\nSB reliability",
        "ResCLIP\ncorr ceiling",
    ]
    ax.bar(np.arange(len(vals)), vals)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Reliability / ceiling")
    ax.set_title("Reliability summary")

    savefig(fig, "resclip_step0_reliability")
    plt.close(fig)


def plot_step1_recovery(summary_df):
    if summary_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    plot_df = summary_df.copy()
    plot_df = plot_df[np.isfinite(plot_df["cv_r2"])].copy()

    x = np.arange(len(plot_df))

    ax = axes[0]
    ax.bar(x, plot_df["cv_r2"].values)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"].values, rotation=30, ha="right")
    ax.set_ylabel("Cross-validated R²")
    ax.set_title("Prediction of final ResCLIP target")

    ax = axes[1]
    ax.bar(x, plot_df["positive_cv_r2_fraction_of_r2_ceiling"].values)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"].values, rotation=30, ha="right")
    ax.set_ylabel("Fraction of Step-0 R² ceiling")
    ax.set_title("Recovery relative to residual reliability")

    savefig(fig, "resclip_step1_clip_recovery")
    plt.close(fig)


# =============================================================================
# [11] METADATA
# =============================================================================

def write_metadata(inputs, step0_summary_df, step1_summary_df):
    metadata = {
        "script_name": "resclip_reliability_and_clip_recovery_diagnostic.py",
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": (
            "Step 0: estimate reliability of CalorieResCLIP by recomputing "
            "the CLIP residual in random split halves of behavioral rating "
            "subjects. Step 1: hold the final CalorieResCLIP target fixed and "
            "test whether alternative/richer CLIP representations recover it."
        ),
        "paths": {
            "REPO_ROOT": REPO_ROOT,
            "FEAT_DIR": FEAT_DIR,
            "BAND_DIR": BAND_DIR,
            "ORIGINAL_DIAG_DIR": ORIGINAL_DIAG_DIR,
            "OUTDIR": OUTDIR,
            "FIGDIR": FIGDIR,
            "Calorie_persubject": PATHS["calorie_persubject"],
            "CLIP_50": inputs["clip_50_path"],
            "CLIP_full512": inputs["clip_512_path"],
            "final_ResCLIP": PATHS["calorie_res_final"],
        },
        "original_decomposition_matching": {
            "ridge_alphas": ALPHAS.tolist(),
            "n_splits": N_SPLITS_ORIGINAL,
            "cv": "KFold(n_splits=5, shuffle=True, random_state=42)",
            "preprocessing": "StandardScaler(with_mean=True, with_std=True)",
            "model": "RidgeCV(alphas=ALPHAS)",
        },
        "step0": {
            "n_reliability_splits": N_RELIABILITY_SPLITS,
            "min_valid_items_per_rating_subject": MIN_VALID_ITEMS_PER_RATING_SUBJECT,
            "min_valid_items_per_split_target": MIN_VALID_ITEMS_PER_SPLIT_TARGET,
            "summary": (
                step0_summary_df.to_dict(orient="records")[0]
                if step0_summary_df is not None and not step0_summary_df.empty
                else {}
            ),
        },
        "step1": {
            "n_outer_splits": N_OUTER_SPLITS,
            "n_permutations_linear": N_PERMUTATIONS_LINEAR,
            "n_permutations_kernel": N_PERMUTATIONS_KERNEL,
            "run_kernel_ridge": RUN_KERNEL_RIDGE,
            "summary_models": (
                step1_summary_df.to_dict(orient="records")
                if step1_summary_df is not None and not step1_summary_df.empty
                else []
            ),
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "outputs": {
            "step0_iterations": os.path.join(OUTDIR, "resclip_step0_reliability_iterations.csv"),
            "step0_summary": os.path.join(OUTDIR, "resclip_step0_reliability_summary.csv"),
            "step1_summary": os.path.join(OUTDIR, "resclip_step1_clip_recovery_summary.csv"),
            "step1_predictions": os.path.join(OUTDIR, "resclip_step1_predictions_by_item.csv"),
            "step1_fold_params": os.path.join(OUTDIR, "resclip_step1_outer_fold_params.csv"),
            "step1_permutations": os.path.join(OUTDIR, "resclip_step1_permutation_distribution.csv"),
            "metadata": os.path.join(OUTDIR, "resclip_reliability_and_clip_recovery_diagnostic_METADATA.md"),
        },
    }

    json_path = os.path.join(
        OUTDIR,
        "resclip_reliability_and_clip_recovery_diagnostic_METADATA.json",
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  saved: {json_path}")

    md_path = os.path.join(
        OUTDIR,
        "resclip_reliability_and_clip_recovery_diagnostic_METADATA.md",
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ResCLIP reliability and richer-CLIP recovery diagnostic\n\n")
        f.write(f"Created: {metadata['created']}\n\n")
        f.write("## Purpose\n\n")
        f.write(metadata["purpose"] + "\n\n")

        f.write("## Inputs\n\n")
        for k, v in metadata["paths"].items():
            f.write(f"- `{k}`: `{v}`\n")

        f.write("\n## Original decomposition matching\n\n")
        f.write("- CLIP target decomposition follows the logic of `plot_residual_calorie_diagnostic.py`.\n")
        f.write("- Predictor preprocessing: `StandardScaler(with_mean=True, with_std=True)`.\n")
        f.write("- Model: `RidgeCV(alphas=np.logspace(-4, 4, 25))`.\n")
        f.write("- CV: `KFold(n_splits=5, shuffle=True, random_state=42)`.\n")

        f.write("\n## Step 0 reliability summary\n\n")
        if step0_summary_df is not None and not step0_summary_df.empty:
            row = step0_summary_df.iloc[0]
            f.write(f"- ResCLIP split-half Spearman mean: `{row.get('resclip_split_spearman_mean', np.nan):.6f}`\n")
            f.write(f"- ResCLIP Spearman-Brown reliability mean: `{row.get('resclip_spearman_brown_mean', np.nan):.6f}`\n")
            f.write(f"- ResCLIP correlation-scale ceiling mean: `{row.get('resclip_corr_ceiling_mean', np.nan):.6f}`\n")
            f.write(f"- ResCLIP R² ceiling mean: `{row.get('resclip_r2_ceiling_mean', np.nan):.6f}`\n")

        f.write("\n## Step 1 recovery summary\n\n")
        if step1_summary_df is not None and not step1_summary_df.empty:
            for _, row in step1_summary_df.iterrows():
                f.write(
                    f"- `{row['model']}`: CV R²=`{row.get('cv_r2', np.nan):.6f}`, "
                    f"Pearson r=`{row.get('pearson_r', np.nan):.6f}`, "
                    f"Spearman rho=`{row.get('spearman_rho', np.nan):.6f}`, "
                    f"p_perm_R²=`{row.get('p_perm_cv_r2', np.nan):.6f}`\n"
                )

    print(f"  saved: {md_path}")


# =============================================================================
# [12] MAIN
# =============================================================================

def main():
    inputs = load_inputs()

    step0_iter_df, step0_summary_df = run_step0_reliability(
        calorie_sub=inputs["calorie_sub"],
        clip_50=inputs["clip_50"],
    )

    step1_summary_df, step1_pred_df, step1_fold_df, step1_perm_df = run_step1_clip_recovery(
        inputs=inputs,
        step0_summary_df=step0_summary_df,
    )

    print("\n[4] Saving figures")
    plot_step0_reliability(step0_iter_df, step0_summary_df)
    plot_step1_recovery(step1_summary_df)

    print("\n[5] Writing metadata")
    write_metadata(
        inputs=inputs,
        step0_summary_df=step0_summary_df,
        step1_summary_df=step1_summary_df,
    )

    print(f"\nDone. Outputs saved in:\n{OUTDIR}")


if __name__ == "__main__":
    main()
#%%
# =============================================================================
# [APPENDIX] STEP 2 + STEP 3 — ResCLIP anchors and R1 bridge
# =============================================================================
r"""
Append this block after the existing:

    if __name__ == "__main__":
        main()

It will run after Step 0/1 and save additional diagnostics into the same OUTDIR.

Step 2
------
Characterize the final ResCLIP target against:
    A. item-level behavioral / annotation / CLIP-text-axis anchors
    B. RDM-level behavioral/model anchors from the R1 RSA model_rdm_vectors.csv

Step 3
------
Bridge current ResCLIP to the R1 calorie-profile residual:
    ResCLIP scalar -> absolute-difference RDM
    compared against R1 Calorie_profile_residual RDM

The R1 comparison is not tautological:
    - ResCLIP is an item-level scalar residual after CLIP prediction.
    - Calorie_profile_residual is an RDM-level subject-profile residual after
      removing the mean calorie RDM.
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr, rankdata


# =============================================================================
# [A1] STEP 2/3 CONFIG
# =============================================================================

STEP23_N_PERM = 10000
STEP23_RANDOM_STATE = 42

_STEP23_MAINDIR = globals().get("REPO_ROOT", REPO_ROOT)
_STEP23_FEAT_DIR = globals().get(
    "FEAT_DIR",
    os.path.join(_STEP23_MAINDIR, "feature_models"),
)
_STEP23_OUTDIR = globals().get(
    "OUTDIR",
    os.path.join(_STEP23_FEAT_DIR, "resclip_reliability_and_recovery_diagnostics"),
)
_STEP23_FIGDIR = globals().get(
    "FIGDIR",
    os.path.join(_STEP23_OUTDIR, "figures"),
)

os.makedirs(_STEP23_OUTDIR, exist_ok=True)
os.makedirs(_STEP23_FIGDIR, exist_ok=True)

# Final ResCLIP target used in Step 1.
_STEP23_RESCLIP_PATH = globals().get("PATHS", {}).get(
    "calorie_res_final",
    os.path.join(
        _STEP23_FEAT_DIR,
        "perceived_calorie_prediction_diagnostics",
        "calorie_res_cv_CLIP.npy",
    ),
)

_STEP23_PREDDIR = os.path.join(
    REPO_ROOT,
    "reproduced_outputs",
    "diagnostics",
    "predCLIP_axis_characterization_v2",
)
_STEP23_PREDCLIP_FULL_CSV = os.path.join(
    _STEP23_PREDDIR,
    "stimulus_scores_annotations_axes_residuals_FULL.csv",
)

# R1 model RDM vectors.
# The first path should be your current R1/disagreement-control output.
R1_MODEL_RDM_VECTORS_CANDIDATES = [
    os.path.join(
        REPO_ROOT,
        "resources",
        "derived_inputs",
        "parent_rsa",
        "model_rdm_vectors.csv",
    ),
]

# You can override manually if needed:
R1_MODEL_RDM_VECTORS_CSV = None


# =============================================================================
# [A2] STEP 2/3 HELPERS
# =============================================================================

def _step23_find_first_existing(paths):
    for p in paths:
        if p is not None and os.path.isfile(p):
            return p
    return None


def _step23_zscore(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-12:
        return np.full_like(x, np.nan, dtype=np.float64)
    return (x - m) / s


def _step23_upper_tri(mat):
    mat = np.asarray(mat)
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]


def _step23_scalar_absdiff_rdm(x):
    x = _step23_zscore(x)
    D = np.abs(x[:, None] - x[None, :])
    return _step23_upper_tri(D).astype(np.float64)


def _step23_safe_corr(x, y, method="spearman"):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return np.nan

    xg = x[good]
    yg = y[good]

    if np.nanstd(xg) < 1e-12 or np.nanstd(yg) < 1e-12:
        return np.nan

    if method == "pearson":
        return float(pearsonr(xg, yg)[0])
    if method == "spearman":
        return float(spearmanr(xg, yg).correlation)

    raise ValueError("method must be 'pearson' or 'spearman'")


def _step23_rank_residualize(y, controls):
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if controls is None or len(controls) == 0:
        return rankdata(y).astype(np.float64)

    X = np.column_stack([
        np.asarray(c, dtype=np.float64).reshape(-1)
        for c in controls
    ])

    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    out = np.full(y.shape, np.nan, dtype=np.float64)

    if good.sum() < X.shape[1] + 3:
        return out

    yr = rankdata(y[good]).reshape(-1, 1)
    Xr = np.column_stack([
        rankdata(X[good, j])
        for j in range(X.shape[1])
    ])

    # OLS residualization with intercept.
    Xb = np.column_stack([Xr, np.ones(Xr.shape[0])])
    beta, *_ = np.linalg.lstsq(Xb, yr, rcond=None)
    pred = Xb @ beta
    out[good] = (yr - pred).reshape(-1)

    return out


def _step23_partial_spearman(y, x, controls):
    y_res = _step23_rank_residualize(y, controls)
    x_res = _step23_rank_residualize(x, controls)

    good = np.isfinite(y_res) & np.isfinite(x_res)
    if good.sum() < 3:
        return np.nan

    if np.nanstd(y_res[good]) < 1e-12 or np.nanstd(x_res[good]) < 1e-12:
        return np.nan

    return float(np.corrcoef(y_res[good], x_res[good])[0, 1])


def _step23_perm_p_item_scalar(
    resclip_item,
    anchor,
    mode="item",
    controls=None,
    n_perm=STEP23_N_PERM,
    seed=STEP23_RANDOM_STATE,
):
    """
    Permutation p-value by shuffling item labels of ResCLIP.

    mode='item':
        anchor is an item-level vector.
    mode='rdm':
        anchor is an RDM vector.
    mode='partial_rdm':
        anchor is an RDM vector and controls are RDM vectors.
    """
    rng = np.random.default_rng(seed)
    resclip_item = np.asarray(resclip_item, dtype=np.float64).reshape(-1)

    if mode == "item":
        obs = _step23_safe_corr(resclip_item, anchor, method="spearman")

        null = np.full(n_perm, np.nan, dtype=np.float64)
        for i in range(n_perm):
            perm = rng.permutation(resclip_item)
            null[i] = _step23_safe_corr(perm, anchor, method="spearman")

    elif mode == "rdm":
        res_rdm = _step23_scalar_absdiff_rdm(resclip_item)
        obs = _step23_safe_corr(res_rdm, anchor, method="spearman")

        null = np.full(n_perm, np.nan, dtype=np.float64)
        for i in range(n_perm):
            perm_rdm = _step23_scalar_absdiff_rdm(rng.permutation(resclip_item))
            null[i] = _step23_safe_corr(perm_rdm, anchor, method="spearman")

    elif mode == "partial_rdm":
        res_rdm = _step23_scalar_absdiff_rdm(resclip_item)
        obs = _step23_partial_spearman(res_rdm, anchor, controls)

        null = np.full(n_perm, np.nan, dtype=np.float64)
        for i in range(n_perm):
            perm_rdm = _step23_scalar_absdiff_rdm(rng.permutation(resclip_item))
            null[i] = _step23_partial_spearman(perm_rdm, anchor, controls)

    else:
        raise ValueError("mode must be 'item', 'rdm', or 'partial_rdm'")

    good = np.isfinite(null)
    if not np.isfinite(obs) or good.sum() == 0:
        return obs, np.nan, np.nan, null

    p_greater = float((np.sum(null[good] >= obs) + 1) / (good.sum() + 1))
    p_two = float((np.sum(np.abs(null[good]) >= abs(obs)) + 1) / (good.sum() + 1))

    return obs, p_greater, p_two, null


def _step23_bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=np.float64)
    q = np.full(pvals.shape, np.nan, dtype=np.float64)

    good = np.isfinite(pvals)
    if good.sum() == 0:
        return q

    p = pvals[good]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)

    q_ranked = ranked * m / (np.arange(1, m + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)

    q_good = np.empty_like(q_ranked)
    q_good[order] = q_ranked
    q[good] = q_good

    return q


def _step23_save_table(df, base_path_no_ext):
    csv_path = base_path_no_ext + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"  saved: {csv_path}")

    excel_csv_path = base_path_no_ext + ".excel.csv"
    df.to_csv(
        excel_csv_path,
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        float_format="%.8g",
    )
    print(f"  saved: {excel_csv_path}")

    try:
        xlsx_path = base_path_no_ext + ".xlsx"
        df.to_excel(xlsx_path, index=False)
        print(f"  saved: {xlsx_path}")
    except Exception as e:
        print(f"  NOTE: could not save xlsx: {e}")


def _step23_savefig(fig, name):
    png = os.path.join(_STEP23_FIGDIR, name + ".png")
    pdf = os.path.join(_STEP23_FIGDIR, name + ".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"  saved: {png}")
    print(f"  saved: {pdf}")


# =============================================================================
# [A3] LOAD ITEM-LEVEL ANCHORS FOR STEP 2A
# =============================================================================

def _step23_load_item_anchors(resclip_item):
    """
    Item-level anchors:
        - from predCLIP_axis_characterization_v2 full table if available
        - from basic feature_model .npy files as fallback/addition
    """
    anchors = {}

    # Always include the target for audit, but it is skipped from correlations.
    anchors["ResCLIP_target_z"] = _step23_zscore(resclip_item)

    if os.path.isfile(_STEP23_PREDCLIP_FULL_CSV):
        full_df = pd.read_csv(_STEP23_PREDCLIP_FULL_CSV)
        if len(full_df) == len(resclip_item):
            for col in full_df.columns:
                if col.lower() in {"index", "condition"}:
                    continue
                if col in {"ResCLIP_z", "ResCLIP_target_z"}:
                    continue

                vals = pd.to_numeric(full_df[col], errors="coerce").to_numpy(dtype=np.float64)
                if np.isfinite(vals).sum() >= 10 and np.nanstd(vals) > 1e-12:
                    anchors[f"predclip_full__{col}"] = vals

            print(f"  Loaded item anchors from: {_STEP23_PREDCLIP_FULL_CSV}")
        else:
            print(
                f"  WARNING: predCLIP full table has {len(full_df)} rows, "
                f"expected {len(resclip_item)}; skipping."
            )
    else:
        print(f"  NOTE: predCLIP full table not found; skipping text/annotation anchors.")

    # Basic behavioral / category feature files.
    basic_files = {
        "Calorie_group": os.path.join(_STEP23_FEAT_DIR, "Calorie.npy"),
        "Health_group": os.path.join(_STEP23_FEAT_DIR, "Health.npy"),
        "Palatability_group": os.path.join(_STEP23_FEAT_DIR, "Palatability.npy"),
        "Familiarity_group": os.path.join(_STEP23_FEAT_DIR, "Familiarity.npy"),
        "CalorieObjective": os.path.join(_STEP23_FEAT_DIR, "CalorieObjective.npy"),
        "SavorySweet": os.path.join(_STEP23_FEAT_DIR, "SavorySweet.npy"),
        # Historically these files lived under FEAT_DIR because the
        # perceived-calorie diagnostic also wrote there. In the release
        # repository that diagnostic has its own reproduced_outputs folder.
        "PredCLIP_cv": os.path.join(
            REPO_ROOT,
            "reproduced_outputs",
            "diagnostics",
            "perceived_calorie_prediction_diagnostics",
            "calorie_pred_cv_CLIP.npy",
        ),
        "RawCalorie_cvtarget": os.path.join(
            REPO_ROOT,
            "reproduced_outputs",
            "diagnostics",
            "perceived_calorie_prediction_diagnostics",
            "calorie_group_raw_z.npy",
        ),
    }

    for name, path in basic_files.items():
        if not os.path.isfile(path):
            continue
        vals = np.load(path).astype(np.float64).squeeze()
        if vals.shape[0] == len(resclip_item):
            anchors[f"feature_file__{name}"] = vals.reshape(-1)

    return anchors


# =============================================================================
# [A4] STEP 2A — ITEM-LEVEL RESCLIP ANCHORS
# =============================================================================

def _step23_run_item_anchor_correlations(resclip_item):
    print("\n[STEP 2A] Item-level ResCLIP anchor correlations")

    anchors = _step23_load_item_anchors(resclip_item)
    rows = []

    for name, vals in anchors.items():
        if name == "ResCLIP_target_z":
            continue

        vals = np.asarray(vals, dtype=np.float64).reshape(-1)
        if vals.shape[0] != len(resclip_item):
            continue

        rho, p_greater, p_two, _null = _step23_perm_p_item_scalar(
            resclip_item=resclip_item,
            anchor=vals,
            mode="item",
            controls=None,
            n_perm=STEP23_N_PERM,
            seed=STEP23_RANDOM_STATE + 101,
        )

        rows.append({
            "analysis": "step2_item_level_anchor",
            "anchor": name,
            "n_items": int(np.isfinite(resclip_item).sum()),
            "spearman_rho": rho,
            "pearson_r": _step23_safe_corr(resclip_item, vals, method="pearson"),
            "p_perm_greater": p_greater,
            "p_perm_two_sided": p_two,
            "n_permutations": int(STEP23_N_PERM),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  WARNING: no item-level anchors available.")
        return df

    df["q_fdr_bh_two_sided"] = _step23_bh_fdr(df["p_perm_two_sided"].to_numpy(dtype=float))
    df["abs_spearman_rho"] = np.abs(df["spearman_rho"].to_numpy(dtype=float))
    df = df.sort_values("abs_spearman_rho", ascending=False).reset_index(drop=True)

    _step23_save_table(
        df,
        os.path.join(_STEP23_OUTDIR, "resclip_step2A_item_anchor_correlations"),
    )

    print("\nTop item-level ResCLIP anchors:")
    print(
        df[["anchor", "spearman_rho", "p_perm_two_sided", "q_fdr_bh_two_sided"]]
        .head(20)
        .round(4)
        .to_string(index=False)
    )

    # Plot top 20.
    top = df.head(20).copy()
    fig, ax = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)
    y = np.arange(len(top))[::-1]
    ax.barh(y, top["spearman_rho"].values[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(top["anchor"].values[::-1], fontsize=8)
    ax.set_xlabel("Spearman rho with ResCLIP item scores")
    ax.set_title("Step 2A — ResCLIP item-level anchors")
    _step23_savefig(fig, "resclip_step2A_item_anchor_correlations_top20")
    plt.close(fig)

    return df


# =============================================================================
# [A5] LOAD R1 MODEL RDM VECTORS
# =============================================================================

def _step23_load_r1_model_rdms():
    path = R1_MODEL_RDM_VECTORS_CSV
    if path is None:
        path = _step23_find_first_existing(R1_MODEL_RDM_VECTORS_CANDIDATES)

    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(
            "Could not find R1 model_rdm_vectors.csv. Tried:\n  "
            + "\n  ".join(R1_MODEL_RDM_VECTORS_CANDIDATES)
        )

    rdm_df = pd.read_csv(path)
    print(f"  Loaded R1 model RDM vectors: {path}")
    print(f"  Shape: {rdm_df.shape}")

    return path, rdm_df


# =============================================================================
# [A6] STEP 2B — RDM-LEVEL ANCHORS
# =============================================================================

def _step23_run_rdm_anchor_correlations(resclip_item, r1_rdm_df):
    print("\n[STEP 2B] RDM-level ResCLIP anchor correlations")

    resclip_rdm = _step23_scalar_absdiff_rdm(resclip_item)

    # Prefer interpretable anchors first, but run all finite columns.
    preferred_order = [
        "Calorie_mean",
        "Calorie_disagreement_sd",
        "Health_mean",
        "Palatability_mean",
        "CalorieObjective",
        "SavorySweet",
        "Health_profile_residual",
        "Palatability_profile_residual",
        "Calorie",
        "Health",
        "Palatability",
        "Gabor",
        "Color",
        "CORnet V4",
        "CORnet IT",
        "Calorie_profile_residual",
    ]

    cols_present = list(r1_rdm_df.columns)
    cols_ordered = [c for c in preferred_order if c in cols_present]
    cols_ordered += [c for c in cols_present if c not in cols_ordered]

    rows = []

    for col in cols_ordered:
        vals = pd.to_numeric(r1_rdm_df[col], errors="coerce").to_numpy(dtype=np.float64)
        if vals.shape[0] != resclip_rdm.shape[0]:
            continue
        if np.isfinite(vals).sum() < 10 or np.nanstd(vals) < 1e-12:
            continue

        rho, p_greater, p_two, _null = _step23_perm_p_item_scalar(
            resclip_item=resclip_item,
            anchor=vals,
            mode="rdm",
            controls=None,
            n_perm=STEP23_N_PERM,
            seed=STEP23_RANDOM_STATE + 202,
        )

        rows.append({
            "analysis": "step2_rdm_level_anchor",
            "anchor_rdm": col,
            "n_pairs": int(np.isfinite(resclip_rdm).sum()),
            "spearman_rho": rho,
            "pearson_r": _step23_safe_corr(resclip_rdm, vals, method="pearson"),
            "p_perm_greater_itemlabel": p_greater,
            "p_perm_two_sided_itemlabel": p_two,
            "n_permutations": int(STEP23_N_PERM),
            "is_direct_r1_bridge": bool(col == "Calorie_profile_residual"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  WARNING: no RDM-level anchors available.")
        return df

    df["q_fdr_bh_two_sided"] = _step23_bh_fdr(
        df["p_perm_two_sided_itemlabel"].to_numpy(dtype=float)
    )
    df["abs_spearman_rho"] = np.abs(df["spearman_rho"].to_numpy(dtype=float))
    df = df.sort_values("abs_spearman_rho", ascending=False).reset_index(drop=True)

    _step23_save_table(
        df,
        os.path.join(_STEP23_OUTDIR, "resclip_step2B_rdm_anchor_correlations"),
    )

    print("\nTop RDM-level ResCLIP anchors:")
    print(
        df[["anchor_rdm", "spearman_rho", "p_perm_two_sided_itemlabel", "q_fdr_bh_two_sided"]]
        .head(20)
        .round(4)
        .to_string(index=False)
    )

    top = df.head(20).copy()
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)
    y = np.arange(len(top))[::-1]
    ax.barh(y, top["spearman_rho"].values[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(top["anchor_rdm"].values[::-1], fontsize=8)
    ax.set_xlabel("Spearman rho with ResCLIP RDM")
    ax.set_title("Step 2B — ResCLIP RDM-level anchors")
    _step23_savefig(fig, "resclip_step2B_rdm_anchor_correlations_top20")
    plt.close(fig)

    return df


# =============================================================================
# [A7] STEP 3 — R1 RESIDUAL BRIDGE
# =============================================================================

def _step23_run_r1_bridge(resclip_item, r1_rdm_df):
    print("\n[STEP 3] R1 bridge: ResCLIP RDM vs Calorie_profile_residual")

    required = "Calorie_profile_residual"
    if required not in r1_rdm_df.columns:
        raise KeyError(
            f"R1 model_rdm_vectors.csv does not contain {required}. "
            f"Available columns: {list(r1_rdm_df.columns)}"
        )

    resclip_rdm = _step23_scalar_absdiff_rdm(resclip_item)
    r1_calres = pd.to_numeric(
        r1_rdm_df["Calorie_profile_residual"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    bridge_specs = [
        ("zero_order", []),
        ("control_Calorie_mean", ["Calorie_mean"]),
        ("control_Calorie_disagreement_sd", ["Calorie_disagreement_sd"]),
        ("control_Calorie_mean_plus_disagreement", ["Calorie_mean", "Calorie_disagreement_sd"]),
        ("control_Health_profile_residual", ["Health_profile_residual"]),
        ("control_Palatability_profile_residual", ["Palatability_profile_residual"]),
        ("control_visual_Gabor_Color_CORnet", ["Gabor", "Color", "CORnet V4", "CORnet IT"]),
        (
            "control_mean_disagreement_visual",
            ["Calorie_mean", "Calorie_disagreement_sd", "Gabor", "Color", "CORnet V4", "CORnet IT"],
        ),
    ]

    rows = []

    for test_name, control_names in bridge_specs:
        missing = [c for c in control_names if c not in r1_rdm_df.columns]
        if missing:
            rows.append({
                "analysis": "step3_r1_bridge",
                "test": test_name,
                "controls": "+".join(control_names),
                "skipped": True,
                "skip_reason": "missing controls: " + ", ".join(missing),
            })
            continue

        controls = [
            pd.to_numeric(r1_rdm_df[c], errors="coerce").to_numpy(dtype=np.float64)
            for c in control_names
        ]

        if len(controls) == 0:
            rho, p_greater, p_two, _null = _step23_perm_p_item_scalar(
                resclip_item=resclip_item,
                anchor=r1_calres,
                mode="rdm",
                controls=None,
                n_perm=STEP23_N_PERM,
                seed=STEP23_RANDOM_STATE + 303,
            )
            pear = _step23_safe_corr(resclip_rdm, r1_calres, method="pearson")
            corr_type = "zero_order_spearman"
        else:
            rho, p_greater, p_two, _null = _step23_perm_p_item_scalar(
                resclip_item=resclip_item,
                anchor=r1_calres,
                mode="partial_rdm",
                controls=controls,
                n_perm=STEP23_N_PERM,
                seed=STEP23_RANDOM_STATE + 303,
            )
            pear = np.nan
            corr_type = "rank_residualized_partial_spearman"

        rows.append({
            "analysis": "step3_r1_bridge",
            "test": test_name,
            "target_current": "ResCLIP_absdiff_RDM",
            "target_r1": "Calorie_profile_residual",
            "controls": "+".join(control_names),
            "skipped": False,
            "correlation_type": corr_type,
            "n_pairs": int(np.isfinite(resclip_rdm).sum()),
            "spearman_or_partial_rho": rho,
            "pearson_r_zero_order_only": pear,
            "p_perm_greater_itemlabel": p_greater,
            "p_perm_two_sided_itemlabel": p_two,
            "n_permutations": int(STEP23_N_PERM),
        })

    df = pd.DataFrame(rows)
    valid = ~df.get("skipped", False).astype(bool)
    df.loc[valid, "q_fdr_bh_two_sided"] = _step23_bh_fdr(
        df.loc[valid, "p_perm_two_sided_itemlabel"].to_numpy(dtype=float)
    )

    _step23_save_table(
        df,
        os.path.join(_STEP23_OUTDIR, "resclip_step3_r1_residual_bridge"),
    )

    print("\nR1 bridge results:")
    show_cols = [
        "test",
        "controls",
        "spearman_or_partial_rho",
        "p_perm_two_sided_itemlabel",
        "q_fdr_bh_two_sided",
    ]
    print(df.loc[valid, show_cols].round(4).to_string(index=False))

    # Plot.
    plot_df = df.loc[valid].copy()
    fig, ax = plt.subplots(figsize=(9.5, 4.9), constrained_layout=True)
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["spearman_or_partial_rho"].to_numpy(dtype=float))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["test"].values, rotation=35, ha="right")
    ax.set_ylabel("Spearman / partial Spearman rho")
    ax.set_title("Step 3 — Current ResCLIP RDM vs R1 calorie-profile residual")
    _step23_savefig(fig, "resclip_step3_r1_residual_bridge")
    plt.close(fig)

    return df


# =============================================================================
# [A8] DESCRIPTIVE TOP-PAIR OVERLAP
# =============================================================================

def _step23_top_pair_descriptives(resclip_item, r1_rdm_df):
    if "Calorie_profile_residual" not in r1_rdm_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    resclip_rdm = _step23_scalar_absdiff_rdm(resclip_item)
    r1_calres = pd.to_numeric(
        r1_rdm_df["Calorie_profile_residual"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    n = len(resclip_item)
    iu = np.triu_indices(n, k=1)

    pair_df = pd.DataFrame({
        "pair_index": np.arange(len(resclip_rdm), dtype=int),
        "stim_i_0based": iu[0],
        "stim_j_0based": iu[1],
        "stim_i_1based": iu[0] + 1,
        "stim_j_1based": iu[1] + 1,
        "ResCLIP_RDM_absdiff": resclip_rdm,
        "R1_Calorie_profile_residual": r1_calres,
        "abs_ResCLIP_RDM_absdiff": np.abs(resclip_rdm),
        "abs_R1_Calorie_profile_residual": np.abs(r1_calres),
    })

    pair_df["rank_abs_ResCLIP"] = pair_df["abs_ResCLIP_RDM_absdiff"].rank(
        ascending=False,
        method="first",
    )
    pair_df["rank_abs_R1_CalRes"] = pair_df["abs_R1_Calorie_profile_residual"].rank(
        ascending=False,
        method="first",
    )

    _step23_save_table(
        pair_df.sort_values("rank_abs_ResCLIP"),
        os.path.join(_STEP23_OUTDIR, "resclip_step3_pairwise_bridge_values"),
    )

    rows = []
    for k in [25, 50, 100, 150, 250, 500]:
        top_res = set(pair_df.nsmallest(k, "rank_abs_ResCLIP")["pair_index"].astype(int))
        top_r1 = set(pair_df.nsmallest(k, "rank_abs_R1_CalRes")["pair_index"].astype(int))
        overlap = len(top_res.intersection(top_r1))
        rows.append({
            "top_k_pairs": int(k),
            "n_overlap": int(overlap),
            "fraction_of_resclip_topk_overlapping_r1_topk": float(overlap / k),
        })

    overlap_df = pd.DataFrame(rows)
    _step23_save_table(
        overlap_df,
        os.path.join(_STEP23_OUTDIR, "resclip_step3_top_pair_overlap"),
    )

    return pair_df, overlap_df


# =============================================================================
# [A9] STEP 2/3 METADATA
# =============================================================================

def _step23_write_metadata(r1_path, item_df, rdm_df, bridge_df):
    meta = {
        "script_section": "APPENDED_STEP2_STEP3",
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": (
            "Characterize final CalorieResCLIP against item-level anchors, "
            "RDM-level behavioral/model anchors, and the R1 Calorie_profile_residual RDM."
        ),
        "inputs": {
            "final_resclip": _STEP23_RESCLIP_PATH,
            "predclip_axis_full_table": _STEP23_PREDCLIP_FULL_CSV,
            "r1_model_rdm_vectors_csv": r1_path,
        },
        "outputs": {
            "step2A_item_anchors": os.path.join(_STEP23_OUTDIR, "resclip_step2A_item_anchor_correlations.csv"),
            "step2B_rdm_anchors": os.path.join(_STEP23_OUTDIR, "resclip_step2B_rdm_anchor_correlations.csv"),
            "step3_r1_bridge": os.path.join(_STEP23_OUTDIR, "resclip_step3_r1_residual_bridge.csv"),
            "pairwise_bridge": os.path.join(_STEP23_OUTDIR, "resclip_step3_pairwise_bridge_values.csv"),
            "top_pair_overlap": os.path.join(_STEP23_OUTDIR, "resclip_step3_top_pair_overlap.csv"),
        },
        "n_permutations": int(STEP23_N_PERM),
        "permutation_scheme": (
            "item-label permutation of the ResCLIP scalar vector; for RDM tests, "
            "the permuted scalar is converted to an absolute-difference RDM."
        ),
        "notes": {
            "step3_direct_test": (
                "ResCLIP_absdiff_RDM vs R1 Calorie_profile_residual. "
                "This compares an item-level CLIP residual converted to an RDM "
                "against an RDM-level subject-profile calorie residual."
            ),
            "partial_tests": (
                "Partial tests use rank-transform plus linear residualization, "
                "matching the residualized Spearman logic used in the R1 RSA script."
            ),
        },
    }

    path_json = os.path.join(_STEP23_OUTDIR, "resclip_step2_step3_METADATA.json")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved: {path_json}")

    path_md = os.path.join(_STEP23_OUTDIR, "resclip_step2_step3_METADATA.md")
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("# ResCLIP Step 2/3 diagnostics\n\n")
        f.write(f"Created: {meta['created']}\n\n")
        f.write("## Purpose\n\n")
        f.write(meta["purpose"] + "\n\n")
        f.write("## Inputs\n\n")
        for k, v in meta["inputs"].items():
            f.write(f"- `{k}`: `{v}`\n")
        f.write("\n## Outputs\n\n")
        for k, v in meta["outputs"].items():
            f.write(f"- `{k}`: `{v}`\n")
        f.write("\n## Main interpretation target\n\n")
        f.write(
            "The key Step 3 test is the zero-order and controlled correlation "
            "between `ResCLIP_absdiff_RDM` and the R1 `Calorie_profile_residual` RDM.\n"
        )
    print(f"  saved: {path_md}")


# =============================================================================
# [A10] RUN STEP 2/3
# =============================================================================

def run_resclip_step2_step3_anchor_and_r1_bridge():
    print("\n" + "=" * 78)
    print("APPENDED ANALYSIS — STEP 2 + STEP 3")
    print("=" * 78)

    if not os.path.isfile(_STEP23_RESCLIP_PATH):
        raise FileNotFoundError(f"Missing final ResCLIP target:\n  {_STEP23_RESCLIP_PATH}")

    resclip_item = np.load(_STEP23_RESCLIP_PATH).astype(np.float64).squeeze()
    resclip_item = _step23_zscore(resclip_item)

    if resclip_item.shape[0] != 96:
        raise ValueError(f"Expected ResCLIP length 96, got {resclip_item.shape}")

    r1_path, r1_rdm_df = _step23_load_r1_model_rdms()

    item_df = _step23_run_item_anchor_correlations(resclip_item)
    rdm_df = _step23_run_rdm_anchor_correlations(resclip_item, r1_rdm_df)
    bridge_df = _step23_run_r1_bridge(resclip_item, r1_rdm_df)
    _pair_df, _overlap_df = _step23_top_pair_descriptives(resclip_item, r1_rdm_df)

    _step23_write_metadata(
        r1_path=r1_path,
        item_df=item_df,
        rdm_df=rdm_df,
        bridge_df=bridge_df,
    )

    print("\nStep 2/3 complete.")
    print(f"Outputs saved in:\n  {_STEP23_OUTDIR}")


if __name__ == "__main__":
    run_resclip_step2_step3_anchor_and_r1_bridge()