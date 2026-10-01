# -*- coding: utf-8 -*-
"""
Created on Mon May  4 11:09:23 2026

@author: G.Marrazzo
"""

# -*- coding: utf-8 -*-
"""
Perceived-calorie prediction / residualization diagnostics
========================================================================

Purpose
-------
Before running the neural encoding model, this script checks whether perceived
calorie ratings are predictable from visual/semantic feature spaces.

It creates diagnostic versions of:

    Calorie_raw
    Calorie_pred_from_LowVis
    Calorie_pred_from_HighVis
    Calorie_pred_from_LowHigh
    Calorie_res_from_LowHigh

If a separate CLIP feature file exists, it also tests:

    Calorie_pred_from_CLIP
    Calorie_res_from_CLIP

Important
---------
This script is for diagnostics only.

For the final neural encoding model, predicted/residual calorie predictors
must be recomputed inside each neural outer CV fold to avoid leakage.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.decomposition import PCA

print("Imports OK.")

# =============================================================================
# [1] PATHS
# =============================================================================

# RELEASE PATH ADAPTATION ONLY.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

FEAT_DIR = os.path.join(REPO_ROOT, "reproduced_outputs", "feature_extraction")
BAND_DIR = os.path.join(REPO_ROOT, "reproduced_outputs", "feature_bands")

OUTDIR = os.path.join(
    REPO_ROOT,
    "reproduced_outputs",
    "diagnostics",
    "perceived_calorie_prediction_diagnostics",
)
os.makedirs(OUTDIR, exist_ok=True)

PILOT_SUBJECTS = [
    104, 105, 109, 112, 115, 117, 118, 122, 124, 125, 126,
    129, 132, 134, 137, 139, 140, 141, 143, 144, 145, 146, 149,
    150, 151,
]

N_CONDITIONS = 96
RANDOM_STATE = 42

ALPHAS = np.logspace(-4, 4, 25)

# =============================================================================
# [2] LOAD FEATURES
# =============================================================================

def load_required(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)


def load_optional(path):
    if os.path.isfile(path):
        return np.load(path).astype(np.float32)
    return None


LowVis = load_required(os.path.join(BAND_DIR, "band_LowVis.npy"))
HighVis = load_required(os.path.join(BAND_DIR, "band_HighVis.npy"))

# Optional: use this if you have a separate CLIP feature matrix.
# If not present, the script will simply skip CLIP-only diagnostics.
CLIP = load_optional(os.path.join(FEAT_DIR, "CLIP.npy"))
if CLIP is None:
    CLIP = load_optional(os.path.join(BAND_DIR, "band_CLIP.npy"))

Calorie_sub = load_required(os.path.join(FEAT_DIR, "Calorie_persubject.npy"))

assert LowVis.shape[0] == N_CONDITIONS
assert HighVis.shape[0] == N_CONDITIONS
assert Calorie_sub.shape[0] == N_CONDITIONS

if CLIP is not None:
    assert CLIP.shape[0] == N_CONDITIONS

print("\nFeature shapes:")
print(f"  LowVis  : {LowVis.shape}")
print(f"  HighVis : {HighVis.shape}")
print(f"  CLIP    : {None if CLIP is None else CLIP.shape}")
print(f"  Calorie : {Calorie_sub.shape}")

# Group-average perceived calorie across subjects.
calorie_group = np.nanmean(Calorie_sub, axis=1).astype(np.float32)

# z-score target for diagnostic comparability.
calorie_group_z = (
    (calorie_group - np.nanmean(calorie_group))
    / (np.nanstd(calorie_group) + 1e-8)
).astype(np.float32)

# =============================================================================
# [3] MODEL HELPERS
# =============================================================================

def make_ridge():
    return make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        RidgeCV(alphas=ALPHAS),
    )


def cv_predict(X, y, n_splits=5):
    """
    Cross-validated prediction of y from X.
    Returns predicted y, residual y, and summary metrics.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    pred = cross_val_predict(
        make_ridge(),
        X,
        y,
        cv=cv,
        method="predict",
    ).astype(np.float32)

    res = (y - pred).astype(np.float32)

    ss_res = np.nansum((y - pred) ** 2)
    ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    pr = pearsonr(y, pred)[0]
    sr = spearmanr(y, pred).correlation

    return pred, res, {
        "cv_r2": float(r2),
        "pearson_r": float(pr),
        "spearman_rho": float(sr),
        "residual_var": float(np.nanvar(res)),
        "target_var": float(np.nanvar(y)),
        "residual_var_fraction": float(np.nanvar(res) / (np.nanvar(y) + 1e-8)),
    }


def full_residualize(X, y):
    """
    Full-data residualization for diagnostic RDM/PCA only.
    Do not use this output directly for neural CV encoding.
    """
    Xz = StandardScaler().fit_transform(X)
    yz = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-8)

    Xb = np.column_stack([Xz, np.ones(Xz.shape[0])])
    beta, *_ = np.linalg.lstsq(Xb, yz, rcond=None)
    yhat = Xb @ beta
    yres = yz - yhat

    return yhat.astype(np.float32), yres.astype(np.float32)


def rdm_vec_1d(y):
    """
    RDM for a scalar image-level variable.
    Uses absolute pairwise differences after z-scoring.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-8)

    D = np.abs(y[:, None] - y[None, :])
    iu = np.triu_indices_from(D, k=1)
    return D[iu]


def rdm_vec_features(X):
    X = StandardScaler().fit_transform(X)
    C = np.corrcoef(X)
    D = 1 - C
    iu = np.triu_indices_from(D, k=1)
    return D[iu]


def rdm_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3:
        return np.nan

    return float(np.corrcoef(a[good], b[good])[0, 1])

# =============================================================================
# [4] RUN GROUP-LEVEL CALORIE PREDICTION DIAGNOSTICS
# =============================================================================

predictor_sets = {
    "LowVis": LowVis,
    "HighVis": HighVis,
    "LowHigh": np.hstack([LowVis, HighVis]).astype(np.float32),
}

if CLIP is not None:
    predictor_sets["CLIP"] = CLIP
    predictor_sets["LowHighCLIP"] = np.hstack([LowVis, HighVis, CLIP]).astype(np.float32)

rows = []
predictions = {}

print("\nCross-validated prediction of group-average perceived calorie:")

for name, X in predictor_sets.items():
    pred, res, metrics = cv_predict(X, calorie_group_z, n_splits=5)
    predictions[name] = {
        "pred": pred,
        "res": res,
    }

    row = {"predictor": name, **metrics}
    rows.append(row)

    print(
        f"  {name:12s}: "
        f"CV R²={metrics['cv_r2']:+.3f}, "
        f"Pearson r={metrics['pearson_r']:+.3f}, "
        f"Spearman rho={metrics['spearman_rho']:+.3f}, "
        f"residual var frac={metrics['residual_var_fraction']:.3f}"
    )

summary_df = pd.DataFrame(rows)
summary_df.to_csv(os.path.join(OUTDIR, "calorie_prediction_cv_summary.csv"), index=False)

# Save diagnostic group-level predicted/residual arrays.
np.save(os.path.join(OUTDIR, "calorie_group_raw_z.npy"), calorie_group_z)

for name, vals in predictions.items():
    np.save(os.path.join(OUTDIR, f"calorie_pred_cv_{name}.npy"), vals["pred"])
    np.save(os.path.join(OUTDIR, f"calorie_res_cv_{name}.npy"), vals["res"])

# =============================================================================
# [5] SUBJECT-SPECIFIC DIAGNOSTICS
# =============================================================================

sub_rows = []

for si, sid in enumerate(PILOT_SUBJECTS):
    if si >= Calorie_sub.shape[1]:
        continue

    y = Calorie_sub[:, si].astype(np.float32)

    if not np.all(np.isfinite(y)):
        continue

    yz = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-8)

    for name, X in predictor_sets.items():
        pred, res, metrics = cv_predict(X, yz, n_splits=5)
        sub_rows.append({
            "subject": sid,
            "predictor": name,
            **metrics,
        })

sub_df = pd.DataFrame(sub_rows)
sub_df.to_csv(os.path.join(OUTDIR, "calorie_prediction_subject_specific_cv_summary.csv"), index=False)

print("\nSubject-specific prediction summary:")
print(
    sub_df.groupby("predictor")
    .agg(
        mean_cv_r2=("cv_r2", "mean"),
        sem_cv_r2=("cv_r2", lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.isfinite(x).sum())),
        mean_pearson_r=("pearson_r", "mean"),
        mean_residual_var_fraction=("residual_var_fraction", "mean"),
    )
    .round(3)
)

# =============================================================================
# [6] RDM CORRELATION DIAGNOSTICS
# =============================================================================

rdm_rows = []

raw_vec = rdm_vec_1d(calorie_group_z)

for name, X in predictor_sets.items():
    X_vec = rdm_vec_features(X)

    # Full-data residualization is only for descriptive RDM diagnostics.
    yhat_full, yres_full = full_residualize(X, calorie_group_z)

    pred_vec = rdm_vec_1d(yhat_full)
    res_vec = rdm_vec_1d(yres_full)

    rdm_rows.append({
        "predictor": name,
        "corr_RDM_rawCalorie_predictorFeatures": rdm_corr(raw_vec, X_vec),
        "corr_RDM_predCalorie_predictorFeatures": rdm_corr(pred_vec, X_vec),
        "corr_RDM_resCalorie_predictorFeatures": rdm_corr(res_vec, X_vec),
        "corr_RDM_rawCalorie_predCalorie": rdm_corr(raw_vec, pred_vec),
        "corr_RDM_rawCalorie_resCalorie": rdm_corr(raw_vec, res_vec),
        "corr_RDM_predCalorie_resCalorie": rdm_corr(pred_vec, res_vec),
    })

rdm_df = pd.DataFrame(rdm_rows)
rdm_df.to_csv(os.path.join(OUTDIR, "calorie_rdm_diagnostics.csv"), index=False)

print("\nRDM diagnostics:")
print(rdm_df.round(3).to_string(index=False))

# =============================================================================
# [7] PLOTS
# =============================================================================

best_name = summary_df.sort_values("cv_r2", ascending=False).iloc[0]["predictor"]
best_pred = predictions[best_name]["pred"]
best_res = predictions[best_name]["res"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Panel 1: CV prediction metrics
ax = axes[0, 0]
x = np.arange(len(summary_df))
ax.bar(x, summary_df["cv_r2"].values)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(summary_df["predictor"], rotation=30, ha="right")
ax.set_ylabel("CV R²")
ax.set_title("Perceived calorie prediction\nfrom feature spaces")

# Panel 2: predicted vs raw
ax = axes[0, 1]
ax.scatter(calorie_group_z, best_pred, s=35, alpha=0.8)
r = pearsonr(calorie_group_z, best_pred)[0]
rho = spearmanr(calorie_group_z, best_pred).correlation
ax.set_xlabel("Raw perceived calorie, z")
ax.set_ylabel(f"Predicted calorie from {best_name}")
ax.set_title(f"Best predictor: {best_name}\nr={r:.3f}, rho={rho:.3f}")

# Panel 3: residual vs raw
ax = axes[0, 2]
ax.scatter(calorie_group_z, best_res, s=35, alpha=0.8)
r = pearsonr(calorie_group_z, best_res)[0]
ax.set_xlabel("Raw perceived calorie, z")
ax.set_ylabel("Residual calorie")
ax.set_title(f"Residual from {best_name}\nr(raw,res)={r:.3f}")

# Panel 4: residual histogram
ax = axes[1, 0]
ax.hist(best_res, bins=20, alpha=0.85)
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Residual calorie")
ax.set_ylabel("Images")
ax.set_title("Residual calorie distribution")

# Panel 5: subject-specific CV R²
ax = axes[1, 1]
pred_order = summary_df["predictor"].tolist()
data = [
    sub_df.loc[sub_df["predictor"] == p, "cv_r2"].values
    for p in pred_order
]
ax.boxplot(data, labels=pred_order, showfliers=False)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticklabels(pred_order, rotation=30, ha="right")
ax.set_ylabel("Subject-specific CV R²")
ax.set_title("Prediction of individual calorie ratings")

# Panel 6: RDM correlations
ax = axes[1, 2]
plot_cols = [
    "corr_RDM_rawCalorie_predictorFeatures",
    "corr_RDM_resCalorie_predictorFeatures",
]
M = rdm_df.set_index("predictor")[plot_cols]
im = ax.imshow(M.values, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(np.arange(len(plot_cols)))
ax.set_xticklabels(["Raw vs features", "Residual vs features"], rotation=30, ha="right")
ax.set_yticks(np.arange(len(M.index)))
ax.set_yticklabels(M.index)
ax.set_title("RDM correlations")
plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()

out_png = os.path.join(OUTDIR, "perceived_calorie_prediction_diagnostics.png")
plt.savefig(out_png, dpi=250, bbox_inches="tight")
plt.close()

print(f"\nSaved figure:\n{out_png}")
print(f"Saved diagnostics to:\n{OUTDIR}")