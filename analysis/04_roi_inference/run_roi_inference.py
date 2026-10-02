# -*- coding: utf-8 -*-
r"""
Encoding analysis — ROI inference for nested calorie/semantic models
========================================================================

Script filename
---------------
roi_inference_model_comparison_calorie_finale_NC_legacyload.py

Traceability
------------
This version loads the split-half noise ceiling from the legacy encoding_model
cache created by the earlier ROI-summary script:

    reproduced_outputs/roi_inference/noise_ceiling_cache/sub-XXX/
        noise_ceiling_r_splithalf100_seed42.npy

That file is already NC(r) = sqrt(max(r_SB, 0)), and is therefore comparable
to held-out r_joint.

Main story
----------
1. The visual baseline model (M0) works.
2. Adding calorie information (M1 raw, M2 PredCLIP) improves model fit over M0.
3. M1 ≈ M2: the benefit of raw calorie ratings is largely recovered by PredCLIP.
4. The gain from M2 over M0 increases along the visual hierarchy.

Main figure
-----------
main_fig1_core_story:
    A. M0 visual baseline across EarlyVisual, IntermediateVisual, HighLevelVTC
       with noise ceiling band.
    B. M0 / M1 / M2 grouped bars across the visual hierarchy
       with noise ceiling band — shows M1 ≈ M2 > M0.
    C. Delta r_joint (M2-M0 bars, M1-M0 line overlay) across the hierarchy
       — lines overlap, showing PredCLIP captures the same gain as raw calorie.

Supplementary figures
---------------------
supp_fig1_predclip_vs_resclip_split:
    PredCLIP vs ResCLIP split contributions across ROIs in M4

supp_fig2_hierarchy_split_contributions:
    A. Absolute M4 split contributions across the visual hierarchy
       (LowVis, HighVis, PredCLIP, ResCLIP)
    B. Relative importance across the visual hierarchy
       (stacked 100% bars; descriptive, positive contributions only)

Statistics
----------
- Subject is the inferential unit.
- ROI value per subject = mean vertexwise held-out r_joint within ROI.
- One-sample tests of r_joint > 0 use sign-flipping on Fisher-z transformed r.
- Paired model comparisons use sign-flipping on within-subject Fisher-z differences.
- Split-model one-sample tests use sign-flipping on r_split values.
- Split-model paired PredCLIP > ResCLIP uses sign-flipping on within-subject
  paired r_split differences.
- Visual hierarchy:
      EarlyVisual = 0
      IntermediateVisual = 1
      HighLevelVTC = 2

Important plotting rule
-----------------------
Inferential markers in figures use FDR-corrected significance only:
- bar stars: q_fdr_bh < .05
- brackets: q_fdr_bh < .05

Noise ceiling
-------------
Split-half reliability (Spearman-Brown corrected to full N repetitions) loaded
from GLMsingle output per subject. Shown as a shaded band (mean ± 1 SEM across
subjects) per ROI in Panels A and B. Provides an upper bound on achievable
encoding model performance given the noise level of the data.

Outputs
-------
CSV:
    subjects_included.csv
    models_available.csv
    roi_model_r_joint_subject_values.csv
    roi_model_r_joint_inference.csv
    roi_model_comparisons.csv
    hierarchy_model_values.csv
    hierarchy_model_gain_values.csv
    hierarchy_model_gain_slopes.csv
    hierarchy_model_gain_slope_tests.csv
    hierarchy_model_gain_pairwise.csv
    plot_annotation_audit.csv
    split_model_roi_subject_values.csv
    split_model_band_inference.csv
    split_model_contrast_inference.csv
    split_model_hierarchy_relative_importance.csv
    noise_ceiling_roi_values.csv          ← new

Figures:
    main_fig1_core_story.png/.pdf
    supp_fig1_predclip_vs_resclip_split.png/.pdf
    supp_fig2_hierarchy_split_contributions.png/.pdf
"""

# =============================================================================
# [0] IMPORTS
# =============================================================================

import os
import argparse
from collections import OrderedDict
from itertools import product as itertools_product

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

print("Imports OK.")


# =============================================================================
# [1] CONFIG
# =============================================================================

# RELEASE PATH ADAPTATION ONLY.
# Statistical and numerical inference code below is unchanged from the final
# executed historical ROI-inference script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

MAINDIR = REPO_ROOT

OUTDIR = os.path.join(REPO_ROOT, "reproduced_outputs", "roi_inference")
FIGDIR = os.path.join(OUTDIR, "figures")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

ATLAS_DIR = os.path.join(REPO_ROOT, "resources", "atlas")
HCP_DLABEL = os.path.join(
    ATLAS_DIR,
    "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors"
    ".32k_fs_LR.dlabel.nii",
)

PILOT_SUBJECTS = [
    104, 105, 109, 112, 115, 117, 118, 122, 124, 125, 126,
    129, 132, 134, 137, 139, 140, 141, 143, 144, 145, 146, 149,
    150, 151,
]

N_GRAYORD  = 91282
N_CORTICAL = 59412
N_CONDITIONS = 96
ALPHA              = 0.05
RANDOM_STATE       = 42
MAX_EXACT_SUBS     = 20
N_MONTE_CARLO_PERMS = 50000
MIN_ROI_VERTICES   = 25

SAVE_PNG      = True
SAVE_PDF      = True
DPI           = 300
SAVE_SUPP_FIG = True

# Figure display policy
MAINFIG_USE_FDR_ONLY = True
SUPPFIG_USE_FDR_ONLY = True

# point styling
POINT_SIZE  = 28
POINT_ALPHA = 0.70

# hierarchy-only summary figure
HIERARCHY_SPLIT_ROIS = ["EarlyVisual", "IntermediateVisual", "HighLevelVTC"]

# ── Noise ceiling config ──────────────────────────────────────────────────────
# The noise ceiling is descriptive only; it is not used for model fitting or
# primary statistical inference.
#
# Choose exactly one mode:
#
#   "historical"
#       Load the bundled reference NC(r) maps supplied under historical_outputs/.
#       This is the default for exact manuscript/figure reproduction.
#       Missing historical files are treated as an error.
#
#   "cached"
#       Load NC(r) maps only from the local reproduced-output cache below.
#       If a subject has no cached map, NC is omitted for that subject.
#       Nothing is recomputed.
#
#   "recompute"
#       Recompute NC(r) for every subject from the public GLMsingle trial betas,
#       overwrite/update the local cache, and use the recomputed maps.
#
# Historical definition:
#     r_half = corr(beta_half_1, beta_half_2) across conditions
#     r_SB   = 2*r_half / (1+r_half)
#     NC(r)  = sqrt(max(r_SB, 0))
#
# NC(r) is therefore in the same correlation metric as r_joint.

def _parse_nc_mode_from_cli():
    """
    Select noise-ceiling behavior from the command line.

    Supported forms
    ---------------
    python run_roi_inference.py --historical-nc
    python run_roi_inference.py --cached-nc
    python run_roi_inference.py --recompute-nc

    Single-dash aliases are also accepted:
    -historical-nc, -cached-nc, -recompute-nc

    Alternatively:
    python run_roi_inference.py --nc-mode historical
    python run_roi_inference.py --nc-mode cached
    python run_roi_inference.py --nc-mode recompute

    If no NC option is supplied, historical mode is used.
    """
    parser = argparse.ArgumentParser(
        description="ROI inference and model comparison for the encoding analysis."
    )
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--historical-nc", "-historical-nc",
        dest="nc_mode",
        action="store_const",
        const="historical",
        help="Load bundled reference noise-ceiling maps from historical_outputs/ (default).",
    )
    group.add_argument(
        "--cached-nc", "-cached-nc",
        dest="nc_mode",
        action="store_const",
        const="cached",
        help="Use only existing reproduced NC cache; do not recompute missing maps.",
    )
    group.add_argument(
        "--recompute-nc", "-recompute-nc",
        dest="nc_mode",
        action="store_const",
        const="recompute",
        help="Recompute all noise-ceiling maps from GLMsingle trial betas.",
    )
    group.add_argument(
        "--nc-mode",
        dest="nc_mode",
        choices=("historical", "cached", "recompute"),
        help="Explicitly select the NC mode.",
    )

    args = parser.parse_args()
    return args.nc_mode or "historical"


NC_MODE = _parse_nc_mode_from_cli()

GLMSINGLE_DIR = os.path.join(REPO_ROOT, "data", "glmsingle")
HISTORICAL_OUTPUTS_DIR = os.path.join(REPO_ROOT, "historical_outputs")
NC_CACHE_DIR = os.path.join(OUTDIR, "noise_ceiling_cache")

N_SPLIT_HALF_ITER = 100
NOISE_RANDOM_STATE = 42
NC_CACHE_TAG = f"splithalf{N_SPLIT_HALF_ITER}_seed{NOISE_RANDOM_STATE}"

# Preferred NC(r) filename. This is already correlation-scale NC(r), not raw
# r_half and not raw Spearman-Brown reliability.
LEGACY_NC_FILENAME = f"noise_ceiling_r_{NC_CACHE_TAG}.npy"
LEGACY_NC_RSB_FILENAME = f"noise_ceiling_rsb_raw_{NC_CACHE_TAG}.npy"
LEGACY_NC_RHALF_FILENAME = f"noise_ceiling_rhalf_{NC_CACHE_TAG}.npy"

# Retained for compatibility when reading an existing reproduced cache.
NC_FNAME_CANDIDATES = [
    LEGACY_NC_FILENAME,
    "noise_ceiling_r_splithalf100_seed42.npy",
    "reliability_splithalf_sb.npy",
    "reliability_splithalf_SB.npy",
    "sub-{sub_id}_reliability_splithalf_sb.npy",
    "sub-{sub_id}_reliability_splithalf_SB.npy",
    "reliability_splithalf.npy",
    "sub-{sub_id}_reliability_splithalf.npy",
]

# Noise-ceiling recomputation can tolerate incomplete GLMsingle designs.
NC_ALLOW_INCOMPLETE_DESIGN = True
NC_MIN_REPS_PER_CONDITION = 2
NC_WARN_IF_VALID_CONDITIONS_BELOW = 90

if NC_MODE not in {"historical", "cached", "recompute"}:
    raise ValueError(
        f"Invalid NC_MODE={NC_MODE!r}. "
        "Choose 'historical', 'cached', or 'recompute'."
    )

NC_COLOR = "#333333"   # dark gray for noise ceiling band
NC_ALPHA = 0.13        # transparency of shaded band
NC_MIN_SUBJECTS = 1



# =============================================================================
# [1.1] MODEL REGISTRY
# =============================================================================

MODEL_ROOT = os.path.join(REPO_ROOT, "reproduced_outputs", "encoding_models")

MODELS = OrderedDict([
    (
        "M0_visual",
        {
            "label":       "M0 Visual",
            "short_label": "Visual",
            "long_label":  "LowVis + HighVis",
            "dir":         os.path.join(MODEL_ROOT, "M0_visual"),
            "color":       "#2CA02C",
        },
    ),
    (
        "M1_visual_rawcal",
        {
            "label":       "M1 +RawCal",
            "short_label": "+RawCal",
            "long_label":  "LowVis + HighVis + raw perceived calorie",
            "dir":         os.path.join(MODEL_ROOT, "M1_visual_rawcal"),
            "color":       "#4C78A8",
        },
    ),
    (
        "M2_visual_predclip",
        {
            "label":       "M2 +PredCLIP",
            "short_label": "+PredCLIP",
            "long_label":  "LowVis + HighVis + CLIP-predicted calorie",
            "dir":         os.path.join(MODEL_ROOT, "M2_visual_predclip"),
            "color":       "#D65F5F",
        },
    ),
    (
        "M3_visual_resclip",
        {
            "label":       "M3 +ResCLIP",
            "short_label": "+ResCLIP",
            "long_label":  "LowVis + HighVis + CLIP-residual calorie",
            "dir":         os.path.join(MODEL_ROOT, "M3_visual_resclip"),
            "color":       "#8C564B",
        },
    ),
    (
        "M4_visual_pred_resclip",
        {
            "label":       "M4 +Pred+Res",
            "short_label": "+Pred+Res",
            "long_label":  "LowVis + HighVis + CLIP-predicted + CLIP-residual calorie",
            "dir":         os.path.join(MODEL_ROOT, "M4_visual_pred_resclip"),
            "color":       "#E15759",
        },
    ),
])

REQUIRE_ALL_MODELS = True

# =============================================================================
# [1.1b] CLIP-SEPARATED BASELINE DIAGNOSTIC MODEL REGISTRY
# =============================================================================

RUN_CLIPBAND_DIAGNOSTIC = True
REQUIRE_ALL_DIAGNOSTIC_MODELS = True

DIAGNOSTIC_MODELS = OrderedDict([
    (
        "D0_visual_noclip",
        {
            "label":       "D0 no CLIP",
            "short_label": "NoCLIP",
            "long_label":  "LowVis + HighVisNoCLIP",
            "dir":         os.path.join(MODEL_ROOT, "D0_visual_noclip"),
            "color":       "#7F7F7F",
        },
    ),
    (
        "D1_visual_clipband",
        {
            "label":       "D1 CLIP band",
            "short_label": "+CLIPband",
            "long_label":  "LowVis + HighVisNoCLIP + CLIPown",
            "dir":         os.path.join(MODEL_ROOT, "D1_visual_clipband"),
            "color":       "#9467BD",
        },
    ),
    (
        "D2_visual_clipband_predclip",
        {
            "label":       "D2 CLIP band + PredCLIP",
            "short_label": "+CLIPband+PredCLIP",
            "long_label":  "LowVis + HighVisNoCLIP + CLIPown + CaloriePredCLIP",
            "dir":         os.path.join(MODEL_ROOT, "D2_visual_clipband_predclip"),
            "color":       "#D65F5F",
        },
    ),
])

# =============================================================================
# [1.2] MODEL COMPARISONS
# =============================================================================

MODEL_COMPARISONS = OrderedDict([
    ("M1_rawcal_gt_M0_visual",    ("M1_visual_rawcal",       "M0_visual",            "greater")),
    ("M2_predclip_gt_M0_visual",  ("M2_visual_predclip",     "M0_visual",            "greater")),
    ("M3_resclip_gt_M0_visual",   ("M3_visual_resclip",      "M0_visual",            "greater")),
    ("M4_pred_res_gt_M0_visual",  ("M4_visual_pred_resclip", "M0_visual",            "greater")),
    ("M2_predclip_gt_M1_rawcal",  ("M2_visual_predclip",     "M1_visual_rawcal",     "greater")),
    ("M2_predclip_gt_M3_resclip", ("M2_visual_predclip",     "M3_visual_resclip",    "greater")),
    ("M4_pred_res_gt_M2_predclip",("M4_visual_pred_resclip", "M2_visual_predclip",   "greater")),
    ("M4_pred_res_gt_M1_rawcal",  ("M4_visual_pred_resclip", "M1_visual_rawcal",     "greater")),
])

COMPARISON_LABELS = OrderedDict([
    ("M1_rawcal_gt_M0_visual",     "+RawCal > Visual"),
    ("M2_predclip_gt_M0_visual",   "+PredCLIP > Visual"),
    ("M3_resclip_gt_M0_visual",    "+ResCLIP > Visual"),
    ("M4_pred_res_gt_M0_visual",   "+Pred+Res > Visual"),
    ("M2_predclip_gt_M1_rawcal",   "+PredCLIP > +RawCal"),
    ("M2_predclip_gt_M3_resclip",  "+PredCLIP > +ResCLIP"),
    ("M4_pred_res_gt_M2_predclip", "+Pred+Res > +PredCLIP"),
    ("M4_pred_res_gt_M1_rawcal",   "+Pred+Res > +RawCal"),
])

# Diagnostic comparisons are deliberately separate from MODEL_COMPARISONS.
# This keeps primary M0-M4 inference and FDR unchanged.

DIAGNOSTIC_MODEL_COMPARISONS = OrderedDict([
    (
        "D1_clipband_gt_D0_noclip",
        ("D1_visual_clipband", "D0_visual_noclip", "greater"),
    ),
    (
        "D2_predclip_gt_D1_clipband",
        ("D2_visual_clipband_predclip", "D1_visual_clipband", "greater"),
    ),
    (
        "D2_predclip_gt_D0_noclip",
        ("D2_visual_clipband_predclip", "D0_visual_noclip", "greater"),
    ),
])

COMPARISON_LABELS.update(OrderedDict([
    ("D1_clipband_gt_D0_noclip", "+CLIPband > NoCLIP"),
    ("D2_predclip_gt_D1_clipband", "+PredCLIP > CLIPband"),
    ("D2_predclip_gt_D0_noclip", "+CLIPband+PredCLIP > NoCLIP"),
]))

MAIN_DIAGNOSTIC_COMPARISON = "D2_predclip_gt_D1_clipband"

MAIN_GAIN_COMPARISON = "M2_predclip_gt_M0_visual"
M1_GAIN_COMPARISON   = "M1_rawcal_gt_M0_visual"


# =============================================================================
# [1.3] OPTIONAL SPLIT-MODEL INFERENCE
# =============================================================================

SPLIT_MODEL_KEY = "M4_visual_pred_resclip"

SPLIT_FEATURE_SPACES = [
    "LowVis",
    "HighVis",
    "CaloriePredCLIP",
    "CalorieResCLIP",
]

SPLIT_BAND_LABELS = OrderedDict([
    ("LowVis",           "LowVis"),
    ("HighVis",          "HighVis"),
    ("CaloriePredCLIP",  "PredCLIP"),
    ("CalorieResCLIP",   "ResCLIP"),
])

SPLIT_BAND_COLORS = OrderedDict([
    ("LowVis",           "#4878CF"),
    ("HighVis",          "#6ACC65"),
    ("CaloriePredCLIP",  "#D65F5F"),
    ("CalorieResCLIP",   "#8C564B"),
])

SPLIT_CONTRASTS = OrderedDict([
    ("PredCLIP_gt_ResCLIP",
     ("r_split_CaloriePredCLIP", "r_split_CalorieResCLIP", "greater")),
])

SPLIT_CONTRAST_LABELS = OrderedDict([
    ("PredCLIP_gt_ResCLIP", "PredCLIP > ResCLIP"),
])


# =============================================================================
# [1.4] ROIs
# =============================================================================

# ROI_GROUPS = OrderedDict([
#     ("EarlyVisual",        ["V1", "V2", "V3", "V4"]),
#     ("IntermediateVisual", ["V8", "PIT", "LO1", "LO2", "LO3"]),
#     ("HighLevelVTC", ["FFC", "VVC", "TE1p", "TE2p"]),
#     ("PHC",                ["PHA1", "PHA2", "PHA3"]),
#     ("OFCFrontal",         ["47s", "47m", "13l", "11l", "OFC", "pOFC"]),
# ])

# ROI_LABELS = OrderedDict([
#     ("EarlyVisual",        "Early visual"),
#     ("IntermediateVisual", "Intermediate visual"),
#     ("HighLevelVTC",            "HighLevelVTC"),
#     ("PHC",                "PHC"),
#     ("OFCFrontal",         "OFC/frontal"),
# ])

ROI_GROUPS = OrderedDict([
    ("EarlyVisual",        ["V1", "V2", "V3", "V4"]),
    ("IntermediateVisual", ["V8", "PIT", "LO1", "LO2", "LO3"]),
    ("HighLevelVTC", ["FFC", "VVC", "TE1p", "TE2p"]),
  
])

ROI_LABELS = OrderedDict([
    ("EarlyVisual",        "Early visual"),
    ("IntermediateVisual", "Intermediate visual"),
    ("HighLevelVTC",            "HighLevelVTC"),
    
])
VISUAL_HIERARCHY = OrderedDict([
    ("EarlyVisual",        0.0),
    ("IntermediateVisual", 1.0),
    ("HighLevelVTC",            2.0),
])


# =============================================================================
# [2] STATS HELPERS
# =============================================================================

def fisher_z(r):
    return np.arctanh(np.clip(np.asarray(r, dtype=np.float64), -0.999999, 0.999999))


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
    m     = float(np.mean(x))
    s     = sem(x)
    tcrit = stats.t.ppf(0.975, df=x.size - 1)
    return float(m - tcrit * s), float(m + tcrit * s)


def cohens_dz(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    sd = np.std(x, ddof=1)
    if sd < 1e-12:
        return np.nan
    return float(np.mean(x) / sd)


def signflip_1d(values, alternative="greater", seed=RANDOM_STATE):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size

    if n == 0:
        return np.nan

    obs = float(np.mean(x))

    if n <= MAX_EXACT_SUBS:
        n_combs = 2 ** n
        exceed  = 0
        for signs_tuple in itertools_product([-1.0, 1.0], repeat=n):
            signs = np.asarray(signs_tuple, dtype=np.float64)
            null  = float(np.mean(signs * x))
            if alternative == "greater":
                exceed += null >= obs
            elif alternative == "less":
                exceed += null <= obs
            elif alternative == "two-sided":
                exceed += abs(null) >= abs(obs)
            else:
                raise ValueError("alternative must be greater, less, or two-sided")
        return max(exceed / n_combs, 1.0 / n_combs)

    rng    = np.random.default_rng(seed)
    batch  = 5000
    exceed = 0
    done   = 0
    while done < N_MONTE_CARLO_PERMS:
        b     = min(batch, N_MONTE_CARLO_PERMS - done)
        signs = rng.choice([-1.0, 1.0], size=(b, n)).astype(np.float64)
        null  = np.mean(signs * x[None, :], axis=1)
        if alternative == "greater":
            exceed += int(np.sum(null >= obs))
        elif alternative == "less":
            exceed += int(np.sum(null <= obs))
        elif alternative == "two-sided":
            exceed += int(np.sum(np.abs(null) >= abs(obs)))
        else:
            raise ValueError("alternative must be greater, less, or two-sided")
        done += b
    return float((exceed + 1) / (N_MONTE_CARLO_PERMS + 1))


def add_fdr(df, p_col="p", alpha=ALPHA, group_col=None):
    df = df.copy()
    df["q_fdr_bh"]  = np.nan
    df["sig_fdr_05"] = False

    if len(df) == 0 or p_col not in df.columns:
        return df

    if group_col is None:
        valid = np.isfinite(df[p_col].to_numpy(dtype=float))
        if valid.sum() > 0:
            idx = df.index[valid]
            q   = multipletests(df.loc[idx, p_col].values, method="fdr_bh")[1]
            df.loc[idx, "q_fdr_bh"]  = q
            df.loc[idx, "sig_fdr_05"] = q < alpha
    else:
        for _, subdf in df.groupby(group_col, sort=False):
            valid = np.isfinite(subdf[p_col].to_numpy(dtype=float))
            if valid.sum() > 0:
                idx = subdf.index[valid]
                q   = multipletests(df.loc[idx, p_col].values, method="fdr_bh")[1]
                df.loc[idx, "q_fdr_bh"]  = q
                df.loc[idx, "sig_fdr_05"] = q < alpha
    return df


def safe_float(x):
    try:
        x = float(x)
    except Exception:
        return np.nan
    return x if np.isfinite(x) else np.nan


def p_to_stars(p=None, q=None):
    val = q if q is not None and np.isfinite(q) else p
    if val is None or not np.isfinite(val):
        return ""
    if val < 0.001:
        return "***"
    if val < 0.01:
        return "**"
    if val < 0.05:
        return "*"
    return ""


def should_mark(row, use_fdr_only=True):
    if row is None:
        return False
    if use_fdr_only:
        q = safe_float(row.get("q_fdr_bh", np.nan))
        return np.isfinite(q) and q < ALPHA
    p = safe_float(row.get("p", np.nan))
    return np.isfinite(p) and p < ALPHA


def mark_text(row, use_fdr_only=True):
    if row is None:
        return ""
    if use_fdr_only:
        return p_to_stars(safe_float(row.get("q_fdr_bh", np.nan)))
    return p_to_stars(safe_float(row.get("p", np.nan)))


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
# [3] ATLAS HELPERS
# =============================================================================

def load_hcp_parcels(dlabel_path):
    print("\n[3] Loading HCP MMP parcellation")

    dlabel = nib.load(dlabel_path)
    data   = dlabel.get_fdata(dtype=np.float32).squeeze()

    if data.ndim != 1:
        data = data.reshape(-1)

    if data.shape[0] == N_CORTICAL:
        label_map = data.astype(np.float32)
    elif data.shape[0] >= N_CORTICAL:
        label_map = data[:N_CORTICAL].astype(np.float32)
    else:
        raise ValueError(f"Unexpected dlabel length {data.shape[0]}")

    label_axis = dlabel.header.get_axis(0)
    label_dict = label_axis.label[0]

    name_to_values = {}
    for val, (full_name, _rgba) in label_dict.items():
        val = int(val)
        if val == 0:
            continue
        base = full_name.replace("_ROI", "")
        if base.startswith("L_") or base.startswith("R_"):
            base = base[2:]
        if base.endswith("_L") or base.endswith("_R"):
            base = base[:-2]
        name_to_values.setdefault(base, []).append(val)

    requested = set()
    for parcels in ROI_GROUPS.values():
        requested.update(parcels)

    parcel_masks = {}
    for parcel in sorted(requested):
        vals = name_to_values.get(parcel, [])
        if len(vals) == 0:
            print(f"  WARNING: parcel not found: {parcel}")
            parcel_masks[parcel] = np.zeros(N_CORTICAL, dtype=bool)
        else:
            parcel_masks[parcel] = np.isin(label_map, vals)

    roi_masks = OrderedDict()
    for roi_name, parcels in ROI_GROUPS.items():
        mask = np.zeros(N_CORTICAL, dtype=bool)
        for p in parcels:
            mask |= parcel_masks.get(p, np.zeros(N_CORTICAL, dtype=bool))
        roi_masks[roi_name] = mask
        print(f"  ROI {roi_name:20s}: {int(mask.sum()):6d} vertices")

    return parcel_masks, roi_masks


# =============================================================================
# [3.5] NOISE CEILING LOADING
# =============================================================================

def _subject_strings(sid):
    """
    Return standardized subject strings.

    Accepts:
        104
        "104"
        "sub-104"

    Returns:
        sub_str = "sub-104"
        sub_id  = "104"
    """
    sid_str = str(sid)
    sub_str = sid_str if sid_str.startswith("sub-") else f"sub-{sid_str}"
    sub_id = sub_str.replace("sub-", "")
    return sub_str, sub_id


def _legacy_nc_paths(sid):
    """
    Return the expected legacy noise-ceiling cache paths for one subject.
    """
    sub_str, _ = _subject_strings(sid)
    sub_dir = os.path.join(NC_CACHE_DIR, sub_str)

    return {
        "nc_r": os.path.join(sub_dir, LEGACY_NC_FILENAME),
        "r_sb_raw": os.path.join(sub_dir, LEGACY_NC_RSB_FILENAME),
        "r_half": os.path.join(sub_dir, LEGACY_NC_RHALF_FILENAME),
    }


def _find_historical_nc_file(sid):
    """
    Locate the bundled reference historical NC(r) file for one subject.

    The exact internal layout of historical_outputs/ is not assumed. The
    resolver searches recursively for the canonical filename inside a
    directory named sub-XXX, and never searches outside historical_outputs/.
    """
    sub_str, _ = _subject_strings(sid)

    if not os.path.isdir(HISTORICAL_OUTPUTS_DIR):
        return None

    matches = []
    for root, _dirs, files in os.walk(HISTORICAL_OUTPUTS_DIR):
        if os.path.basename(root) != sub_str:
            continue
        if LEGACY_NC_FILENAME in files:
            matches.append(os.path.join(root, LEGACY_NC_FILENAME))

    matches = sorted(set(matches))

    if len(matches) == 0:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"{sub_str}: multiple historical NC files found:\n  "
            + "\n  ".join(matches)
        )
    return matches[0]


def _find_cached_nc_file(sid):
    """
    Find an existing NC map only in the reproduced-output cache.

    No recomputation is triggered here.
    """
    sub_str, sub_id = _subject_strings(sid)

    for pattern in NC_FNAME_CANDIDATES:
        p = os.path.join(
            NC_CACHE_DIR,
            sub_str,
            pattern.format(sub_id=sub_id, sub_str=sub_str),
        )
        if os.path.isfile(p):
            return p

    return None


def _find_nc_file(sid):
    """
    Resolve an NC source according to NC_MODE.

    historical -> bundled reference historical_outputs only
    cached     -> reproduced cache only
    recompute  -> handled explicitly by load_noise_ceiling()
    """
    if NC_MODE == "historical":
        return _find_historical_nc_file(sid)
    if NC_MODE == "cached":
        return _find_cached_nc_file(sid)
    if NC_MODE == "recompute":
        return None
    raise RuntimeError(f"Unhandled NC_MODE={NC_MODE!r}")


def _load_bool_mask(path):
    """
    Load a boolean mask safely.
    """
    arr = np.load(path)
    arr = np.asarray(arr).squeeze()
    return arr.astype(bool)


def pearson_r_columns_nc(A, B):
    """
    Pearson correlation between matching columns of A and B.

    Used only when NC_MODE='recompute'.
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    A = A - np.nanmean(A, axis=0, keepdims=True)
    B = B - np.nanmean(B, axis=0, keepdims=True)

    num = np.nansum(A * B, axis=0)
    den = np.sqrt(np.nansum(A ** 2, axis=0) * np.nansum(B ** 2, axis=0))

    out = np.full(A.shape[1], np.nan, dtype=np.float32)
    good = den > 1e-12
    out[good] = (num[good] / den[good]).astype(np.float32)

    return out


def fisher_mean_r_nc(rs):
    """
    Fisher-z average of correlation arrays.

    Used only when NC_MODE='recompute'.
    """
    rs = np.asarray(rs, dtype=np.float32)
    rs = np.clip(rs, -0.999999, 0.999999)
    z = np.arctanh(rs)
    return np.tanh(np.nanmean(z, axis=0)).astype(np.float32)


def expand_to_full_nc(compact, good_mask, fill=np.nan):
    """
    Expand compact GLMsingle grayordinate data to full 91k CIFTI space.

    Used only when NC_MODE='recompute'.
    """
    full = np.full(N_GRAYORD, fill, dtype=np.float32)
    full[good_mask] = np.asarray(compact, dtype=np.float32)
    return full


def compute_subject_noise_ceiling_if_missing(sid, force=False):
    """
    Compute full-grayordinate split-half NC(r) for one subject.

    This reproduces the legacy ROI-summary noise-ceiling calculation but is
    tolerant to incomplete trial designs:

        r_half = corr(beta_half_1, beta_half_2) across conditions
        r_SB   = 2*r_half/(1+r_half)
        NC(r)  = sqrt(max(r_SB, 0))

    Incomplete-design policy
    ------------------------
    - betas_trials.npy and stimorder.npy must have the same number of rows.
    - stimorder can be 0-based [0..95] or 1-based [1..96].
    - conditions with >= NC_MIN_REPS_PER_CONDITION trial betas are included.
    - conditions with <  NC_MIN_REPS_PER_CONDITION trial betas are excluded
      from the split-half correlation for that subject.
    - if NC_ALLOW_INCOMPLETE_DESIGN=False, any deviation from exactly
      96 conditions x 14 repetitions raises an error.

    Output files are saved to:
        NC_CACHE_DIR/sub-XXX/
    """
    sub_str, sub_id = _subject_strings(sid)
    sid_int = int(sub_id)

    paths = _legacy_nc_paths(sid)
    if (not force) and all(os.path.isfile(p) for p in paths.values()):
        return paths

    sub_dir = os.path.join(GLMSINGLE_DIR, sub_str)

    betas_path = os.path.join(sub_dir, "betas_trials.npy")
    stim_path = os.path.join(sub_dir, "stimorder.npy")
    mask_path = os.path.join(sub_dir, "good_grayordinates_mask.npy")

    needed = [betas_path, stim_path, mask_path]
    missing = [p for p in needed if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"{sub_str}: cannot compute NC because required GLMsingle files "
            f"are missing:\n" + "\n".join(missing)
        )

    betas_trials = np.load(betas_path).astype(np.float32)
    stimorder_raw = np.load(stim_path).astype(int).reshape(-1)
    good_mask = np.load(mask_path).astype(bool).reshape(-1)

    if betas_trials.ndim != 2:
        raise ValueError(
            f"{sub_str}: betas_trials.npy must be 2D, got shape {betas_trials.shape}"
        )

    if stimorder_raw.shape[0] != betas_trials.shape[0]:
        raise ValueError(
            f"{sub_str}: stimorder length={stimorder_raw.shape[0]} does not match "
            f"betas rows={betas_trials.shape[0]}"
        )

    if good_mask.shape[0] != N_GRAYORD:
        raise ValueError(
            f"{sub_str}: good_grayordinates_mask has length {good_mask.shape[0]}, "
            f"expected {N_GRAYORD}"
        )

    if betas_trials.shape[1] != int(good_mask.sum()):
        raise ValueError(
            f"{sub_str}: betas columns={betas_trials.shape[1]} but "
            f"good_mask.sum()={int(good_mask.sum())}"
        )

    # Normalize stimorder to 0-based condition indices [0..95].
    stim_min = int(np.nanmin(stimorder_raw))
    stim_max = int(np.nanmax(stimorder_raw))

    if stim_min >= 0 and stim_max <= N_CONDITIONS - 1:
        stimorder = stimorder_raw.copy()
        stimorder_base = 0
    elif stim_min >= 1 and stim_max <= N_CONDITIONS:
        stimorder = stimorder_raw - 1
        stimorder_base = 1
    else:
        raise ValueError(
            f"{sub_str}: unexpected stimorder range {stim_min}..{stim_max}; "
            f"expected 0..{N_CONDITIONS - 1} or 1..{N_CONDITIONS}"
        )

    # Keep only rows with valid condition labels and at least some finite beta data.
    valid_label = (stimorder >= 0) & (stimorder < N_CONDITIONS)
    valid_beta_row = np.any(np.isfinite(betas_trials), axis=1)
    valid_row = valid_label & valid_beta_row

    if not np.all(valid_row):
        n_drop = int(np.sum(~valid_row))
        print(f"  {sub_str}: dropping {n_drop} invalid/empty trial rows before NC")
        betas_trials = betas_trials[valid_row, :]
        stimorder = stimorder[valid_row]

    expected_trials = N_CONDITIONS * 14
    by_condition_all = [np.where(stimorder == c)[0] for c in range(N_CONDITIONS)]
    counts = np.array([idx.size for idx in by_condition_all], dtype=int)

    full_design_ok = (
        betas_trials.shape[0] == expected_trials
        and np.all(counts == 14)
    )

    if not full_design_ok and not NC_ALLOW_INCOMPLETE_DESIGN:
        bad = np.where(counts != 14)[0]
        preview = [(int(c), int(counts[c])) for c in bad[:12]]
        raise ValueError(
            f"{sub_str}: incomplete design and NC_ALLOW_INCOMPLETE_DESIGN=False. "
            f"betas rows={betas_trials.shape[0]}, expected={expected_trials}; "
            f"bad condition counts preview={preview}"
        )

    valid_conditions = np.array(
        [c for c, idx in enumerate(by_condition_all) if idx.size >= NC_MIN_REPS_PER_CONDITION],
        dtype=int,
    )
    excluded_conditions = np.array(
        [c for c, idx in enumerate(by_condition_all) if idx.size < NC_MIN_REPS_PER_CONDITION],
        dtype=int,
    )

    if valid_conditions.size < 3:
        raise ValueError(
            f"{sub_str}: only {valid_conditions.size} conditions have at least "
            f"{NC_MIN_REPS_PER_CONDITION} repetitions; cannot compute a stable "
            f"condition-wise split-half correlation."
        )

    if valid_conditions.size < NC_WARN_IF_VALID_CONDITIONS_BELOW:
        print(
            f"  WARNING: {sub_str}: NC uses only {valid_conditions.size}/"
            f"{N_CONDITIONS} conditions with >= {NC_MIN_REPS_PER_CONDITION} reps."
        )
    elif not full_design_ok:
        print(
            f"  {sub_str}: incomplete design tolerated; NC uses "
            f"{valid_conditions.size}/{N_CONDITIONS} conditions."
        )

    by_condition = [by_condition_all[c] for c in valid_conditions]

    rng = np.random.default_rng(NOISE_RANDOM_STATE + sid_int)
    n_good = betas_trials.shape[1]
    n_cond_nc = len(by_condition)
    r_halves = []

    for _ in range(N_SPLIT_HALF_ITER):
        h1 = np.zeros((n_cond_nc, n_good), dtype=np.float32)
        h2 = np.zeros((n_cond_nc, n_good), dtype=np.float32)

        for row_i, idx in enumerate(by_condition):
            idx = rng.permutation(idx)
            n1 = idx.size // 2

            # idx.size >= 2 by construction, so both halves have at least one trial.
            h1[row_i] = np.nanmean(betas_trials[idx[:n1]], axis=0).astype(np.float32)
            h2[row_i] = np.nanmean(betas_trials[idx[n1:]], axis=0).astype(np.float32)

        r_halves.append(pearson_r_columns_nc(h1, h2))

    r_half = fisher_mean_r_nc(np.stack(r_halves, axis=0))

    denom = 1.0 + r_half
    r_sb_raw = np.full_like(r_half, np.nan, dtype=np.float32)
    good = np.abs(denom) > 1e-8
    r_sb_raw[good] = (2.0 * r_half[good] / denom[good]).astype(np.float32)

    r_sb_pos = np.clip(r_sb_raw, 0.0, 1.0)
    nc_r = np.sqrt(r_sb_pos).astype(np.float32)

    out = {
        "nc_r": expand_to_full_nc(nc_r, good_mask),
        "r_sb_raw": expand_to_full_nc(r_sb_raw, good_mask),
        "r_half": expand_to_full_nc(r_half, good_mask),
    }

    legacy_sub_dir = os.path.join(NC_CACHE_DIR, sub_str)
    os.makedirs(legacy_sub_dir, exist_ok=True)

    for key, path in paths.items():
        np.save(path, out[key])

    # Save an audit table so incomplete-design NC computations are traceable.
    counts_df = pd.DataFrame({
        "condition_0based": np.arange(N_CONDITIONS, dtype=int),
        "n_trial_betas": counts.astype(int),
        "included_in_nc": counts >= NC_MIN_REPS_PER_CONDITION,
    })
    counts_df["condition_1based"] = counts_df["condition_0based"] + 1
    counts_df.to_csv(
        os.path.join(legacy_sub_dir, f"noise_ceiling_condition_counts_{NC_CACHE_TAG}.csv"),
        index=False,
    )

    summary_path = os.path.join(legacy_sub_dir, f"noise_ceiling_recompute_summary_{NC_CACHE_TAG}.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(f"subject: {sub_str}\n")
        fh.write(f"stimorder_base_detected: {stimorder_base}\n")
        fh.write(f"n_trial_betas_used: {betas_trials.shape[0]}\n")
        fh.write(f"expected_full_design_trials: {expected_trials}\n")
        fh.write(f"full_design_ok: {full_design_ok}\n")
        fh.write(f"n_conditions_total: {N_CONDITIONS}\n")
        fh.write(f"n_conditions_included_in_nc: {valid_conditions.size}\n")
        fh.write(f"min_reps_required_per_condition: {NC_MIN_REPS_PER_CONDITION}\n")
        fh.write(f"excluded_conditions_0based: {excluded_conditions.tolist()}\n")
        fh.write(f"excluded_conditions_1based: {(excluded_conditions + 1).tolist()}\n")
        fh.write(f"n_split_half_iter: {N_SPLIT_HALF_ITER}\n")
        fh.write(f"noise_random_state_subject: {NOISE_RANDOM_STATE + sid_int}\n")
        fh.write("formula: NC(r)=sqrt(max(2*r_half/(1+r_half),0))\n")

    print(
        f"  computed and cached NC for {sub_str}: "
        f"{valid_conditions.size}/{N_CONDITIONS} conditions included | {paths['nc_r']}"
    )

    return paths



def _find_glmsingle_good_mask(sub_str):
    """
    Find the GLMsingle good-grayordinates mask.

    This is only used for fallback compact files. The preferred legacy NC file
    is full 91k and does not need expansion.
    """
    candidates = [
        os.path.join(GLMSINGLE_DIR, sub_str, "good_grayordinates_mask.npy"),
        os.path.join(GLMSINGLE_DIR, sub_str, "good_mask.npy"),
    ]

    for p in candidates:
        if os.path.isfile(p):
            return p

    return None


def _find_encoding_combined_mask(sub_str):
    """
    Find an encoding-model combined mask as a final fallback.

    This should usually not be needed for the preferred legacy NC files.
    """
    candidates = []

    for mk in [
        "M0_visual",
        "M1_visual_rawcal",
        "M2_visual_predclip",
        "M3_visual_resclip",
        "M4_visual_pred_resclip",
    ]:
        candidates.append(os.path.join(MODEL_ROOT, mk, sub_str, "combined_mask.npy"))

    for p in candidates:
        if os.path.isfile(p):
            return p

    return None


def _expand_with_mask(arr, mask, sub_str, mask_name):
    """
    Expand a compact array using either a full-91k or cortical mask.

    Returns a cortical-length vector.
    """
    arr = np.asarray(arr).squeeze().astype(np.float32)
    mask = np.asarray(mask).squeeze().astype(bool)

    if mask.shape[0] == N_GRAYORD:
        if int(mask.sum()) == arr.size:
            full = np.full(N_GRAYORD, np.nan, dtype=np.float32)
            full[mask] = arr
            return full[:N_CORTICAL], f"compact_expanded_with_{mask_name}_full91k"

        cortical_mask = mask[:N_CORTICAL]
        if int(cortical_mask.sum()) == arr.size:
            cortical = np.full(N_CORTICAL, np.nan, dtype=np.float32)
            cortical[cortical_mask] = arr
            return cortical, f"compact_expanded_with_{mask_name}_cortical_subset"

    if mask.shape[0] == N_CORTICAL:
        if int(mask.sum()) == arr.size:
            cortical = np.full(N_CORTICAL, np.nan, dtype=np.float32)
            cortical[mask] = arr
            return cortical, f"compact_expanded_with_{mask_name}_cortical"

    raise ValueError(
        f"{sub_str}: compact NC size {arr.size} does not match "
        f"{mask_name} mask shape={mask.shape}, true={int(mask.sum())}"
    )


def _load_nc_as_cortical(fpath, sid):
    """
    Load a noise-ceiling map and return a cortical-length vector.

    Preferred case:
        noise_ceiling_r_splithalf100_seed42.npy is a full 91k vector saved
        by the old ROI-summary script. We simply trim it to the first 59,412
        cortical grayordinates.

    Fallback cases:
        - cortical 59,412 vector
        - compact vector expanded with GLMsingle good mask
        - compact vector expanded with encoding combined mask
    """
    sub_str, _ = _subject_strings(sid)

    arr = np.load(fpath).astype(np.float32)
    arr = np.asarray(arr).squeeze()

    if arr.ndim != 1:
        arr = arr.reshape(-1)

    n = arr.size
    base = os.path.basename(fpath)

    # Preferred legacy file: full 91k.
    if n == N_GRAYORD:
        method = "legacy_noise_ceiling_r_full91k"
        if base != LEGACY_NC_FILENAME:
            method = "full91k_fallback"
        return arr[:N_CORTICAL].astype(np.float32), method, n

    # Cortical-only vector.
    if n == N_CORTICAL:
        return arr.astype(np.float32), "full_cortical_59412", n

    # Fallback compact GLMsingle vector.
    good_mask_path = _find_glmsingle_good_mask(sub_str)
    if good_mask_path is not None:
        good_mask = _load_bool_mask(good_mask_path)
        try:
            cortical, method = _expand_with_mask(
                arr=arr,
                mask=good_mask,
                sub_str=sub_str,
                mask_name="glmsingle_good_grayordinates_mask",
            )
            return cortical, method, n
        except ValueError as e:
            print(f"  NOTE: {e}")

    # Fallback encoding compact vector.
    combined_mask_path = _find_encoding_combined_mask(sub_str)
    if combined_mask_path is not None:
        combined_mask = _load_bool_mask(combined_mask_path)
        try:
            cortical, method = _expand_with_mask(
                arr=arr,
                mask=combined_mask,
                sub_str=sub_str,
                mask_name="encoding_combined_mask",
            )
            return cortical, method, n
        except ValueError as e:
            print(f"  NOTE: {e}")

    raise ValueError(
        f"{sub_str}: could not resolve NC array shape {arr.shape}. "
        f"Expected {N_GRAYORD}, {N_CORTICAL}, GLMsingle compact good-mask size, "
        f"or encoding compact combined-mask size."
    )


def load_noise_ceiling(subjects, roi_masks):
    """
    Load or recompute split-half noise-ceiling maps according to NC_MODE.

    historical
        Load bundled reference NC(r) maps from historical_outputs/. Missing files are an
        error because this mode is intended for exact historical reproduction.

    cached
        Load only existing maps from the reproduced NC cache. Missing or
        unreadable maps are skipped; nothing is recomputed.

    recompute
        Recompute every subject's NC(r) from GLMsingle trial betas, overwrite
        the local cache, and load the newly generated map.
    """
    print("\n[3.5] Noise ceiling")
    print(f"  NC mode              : {NC_MODE}")

    if NC_MODE == "historical":
        print(f"  Historical root      : {HISTORICAL_OUTPUTS_DIR}")
    else:
        print(f"  Reproduced NC cache  : {NC_CACHE_DIR}")

    print(f"  NC filename          : {LEGACY_NC_FILENAME}")

    rows = []
    missing_or_skipped = []
    loaded_subjects = []

    for sid in subjects:
        sub_str, _ = _subject_strings(sid)

        if NC_MODE == "recompute":
            try:
                paths = compute_subject_noise_ceiling_if_missing(sid, force=True)
                fpath = paths["nc_r"]
            except Exception as e:
                raise RuntimeError(
                    f"{sub_str}: NC recomputation failed in NC_MODE='recompute'"
                ) from e
        else:
            fpath = _find_nc_file(sid)

        if fpath is None:
            if NC_MODE == "historical":
                raise FileNotFoundError(
                    f"{sub_str}: historical NC file not found under:\n"
                    f"  {HISTORICAL_OUTPUTS_DIR}\n"
                    f"Expected filename: {LEGACY_NC_FILENAME}"
                )

            # cached mode: explicitly omit NC when no cache is available.
            missing_or_skipped.append(sub_str)
            continue

        try:
            cortical, expansion_method, nc_array_size = _load_nc_as_cortical(fpath, sid)
        except Exception as e:
            if NC_MODE == "historical":
                raise RuntimeError(
                    f"{sub_str}: historical NC file could not be loaded:\n  {fpath}"
                ) from e

            if NC_MODE == "recompute":
                raise RuntimeError(
                    f"{sub_str}: freshly recomputed NC file could not be loaded:\n  {fpath}"
                ) from e

            # cached mode: do not silently recompute.
            print(f"  WARNING: {sub_str} cached NC could not be loaded; omitting NC.")
            print(f"           {fpath}")
            print(f"           {e}")
            missing_or_skipped.append(sub_str)
            continue

        loaded_subjects.append(sub_str)

        for roi_name, roi_mask in roi_masks.items():
            finite = roi_mask & np.isfinite(cortical)

            if int(finite.sum()) < MIN_ROI_VERTICES:
                continue

            rows.append({
                "subject": sub_str,
                "roi": roi_name,
                "nc_r": float(np.nanmean(cortical[finite])),
                "n_vertices": int(finite.sum()),
                "nc_source_file": fpath,
                "nc_array_size": int(nc_array_size),
                "nc_expansion_method": expansion_method,
                # Keep the historical column name for backward-compatible
                # output tables. True means the canonical NC(r) filename was used.
                "nc_is_preferred_legacy_file": (
                    os.path.basename(fpath) == LEGACY_NC_FILENAME
                ),
            })

    print(f"  Loaded NC data for {len(set(loaded_subjects))}/{len(subjects)} subjects.")

    if missing_or_skipped:
        print(
            f"  NC omitted for {len(missing_or_skipped)} subjects: "
            f"{missing_or_skipped[:5]}{'...' if len(missing_or_skipped) > 5 else ''}"
        )

    nc_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=[
            "subject", "roi", "nc_r", "n_vertices", "nc_source_file",
            "nc_array_size", "nc_expansion_method", "nc_is_preferred_legacy_file"
        ]
    )

    if nc_df.empty:
        print("  No NC data loaded — noise ceiling omitted from figures.")

    nc_roi = {}

    for roi_name in ROI_GROUPS.keys():
        vals = nc_df.loc[nc_df["roi"] == roi_name, "nc_r"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            nc_roi[roi_name] = None
            print(f"  {roi_name:22s}: NC — no data, omitted from figures")
            continue

        if vals.size < NC_MIN_SUBJECTS:
            nc_roi[roi_name] = None
            print(
                f"  {roi_name:22s}: NC — only {vals.size} subjects "
                f"(< NC_MIN_SUBJECTS={NC_MIN_SUBJECTS}), omitted"
            )
            continue

        m = float(np.mean(vals))
        s = sem(vals)
        ci_lo, ci_hi = ci95(vals)

        # Figure band: mean +/- SEM.
        # nc_roi[roi_name] = {
        #     "mean": m,
        #     "sem": s,
        #     "lo": float(m - s),
        #     "hi": float(m + s),
        #     "ci95_lo": ci_lo,
        #     "ci95_hi": ci_hi,
        #     "n": int(vals.size),
        # }
        
        # Figure band: 95% CI around the group mean.
        nc_roi[roi_name] = {
            "mean": m,
            "sem": s,
            "lo": float(ci_lo),
            "hi": float(ci_hi),
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "n": int(vals.size),
        }
        print(
            f"  {roi_name:22s}: NC mean r = {m:.4f} "
            f"± SEM {s:.4f} (n={vals.size})"
        )

    return nc_df, nc_roi

def summarize_nc_for_subject_subset(nc_df, subjects_subset):
    """
    Reuse the already-loaded main noise-ceiling table and summarize it for
    a specific subject subset.

    This avoids accidentally loading a different NC quantity for diagnostic
    figures. The NC is always the same NC(r) quantity used in the main figure.
    """
    subjects_subset = set(subjects_subset)

    subdf = nc_df[nc_df["subject"].isin(subjects_subset)].copy()

    nc_roi = {}

    for roi_name in ROI_GROUPS.keys():
        vals = subdf.loc[subdf["roi"] == roi_name, "nc_r"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            nc_roi[roi_name] = None
            continue

        m = float(np.mean(vals))
        s = sem(vals)
        ci_lo, ci_hi = ci95(vals)

        # nc_roi[roi_name] = {
        #     "mean": m,
        #     "sem": s,
        #     "lo": float(m - s),
        #     "hi": float(m + s),
        #     "ci95_lo": ci_lo,
        #     "ci95_hi": ci_hi,
        #     "n": int(vals.size),
        # }
        nc_roi[roi_name] = {
            "mean": m,
            "sem": s,
            "lo": float(ci_lo),
            "hi": float(ci_hi),
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "n": int(vals.size),
        }

    return subdf, nc_roi

def write_traceability_metadata(subjects, nc_df, available_models=None):
    """
    Write a lightweight metadata JSON file so the generated outputs can be
    traced back to this script and its inputs.
    """
    import json
    import platform
    from datetime import datetime

    metadata = {
        "script_name": "roi_inference_model_comparison_calorie_finale_NC_legacyload.py",
        "script_version": "v2.3_nc_source_modes",
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": (
            "ROI inference for nested calorie/semantic encoding models with "
            "configurable historical, cached, or recomputed split-half noise ceilings."
        ),
        "noise_ceiling_definition": {
            "preferred_file": LEGACY_NC_FILENAME,
            "nc_mode": NC_MODE,
            "historical_outputs_directory": HISTORICAL_OUTPUTS_DIR,
            "reproduced_cache_directory": NC_CACHE_DIR,
            "formula": "NC(r) = sqrt(max(r_SB, 0)); r_SB = 2*r_half/(1+r_half)",
            "n_split_half_iter": N_SPLIT_HALF_ITER,
            "noise_random_state": NOISE_RANDOM_STATE,
            "allow_incomplete_design": NC_ALLOW_INCOMPLETE_DESIGN,
            "min_reps_per_condition": NC_MIN_REPS_PER_CONDITION,
        },
        "paths": {
            "MAINDIR": MAINDIR,
            "OUTDIR": OUTDIR,
            "FIGDIR": FIGDIR,
            "MODEL_ROOT": MODEL_ROOT,
            "GLMSINGLE_DIR": GLMSINGLE_DIR,
            "HISTORICAL_OUTPUTS_DIR": HISTORICAL_OUTPUTS_DIR,
            "NC_CACHE_DIR": NC_CACHE_DIR,
            "HCP_DLABEL": HCP_DLABEL,
        },
        "subjects": list(map(str, subjects)),
        "n_subjects": int(len(subjects)),
        "roi_groups": {k: list(v) for k, v in ROI_GROUPS.items()},
        "models": {
            mk: {
                "label": spec.get("label"),
                "dir": spec.get("dir"),
                "n_subjects_available": len(spec.get("subjects", [])),
            }
            for mk, spec in (available_models or {}).items()
        },
        "noise_ceiling_loaded": {
            "n_rows": int(len(nc_df)),
            "n_subjects_loaded": int(nc_df["subject"].nunique()) if not nc_df.empty else 0,
            "expansion_methods": (
                sorted(nc_df["nc_expansion_method"].dropna().unique().tolist())
                if not nc_df.empty and "nc_expansion_method" in nc_df.columns
                else []
            ),
            "n_preferred_legacy_rows": (
                int(nc_df["nc_is_preferred_legacy_file"].sum())
                if not nc_df.empty and "nc_is_preferred_legacy_file" in nc_df.columns
                else 0
            ),
            "output_csv": os.path.join(OUTDIR, "noise_ceiling_roi_values.csv"),
        },
        "outputs": {
            "subjects_included": os.path.join(OUTDIR, "subjects_included.csv"),
            "models_available": os.path.join(OUTDIR, "models_available.csv"),
            "roi_model_values": os.path.join(OUTDIR, "roi_model_r_joint_subject_values.csv"),
            "roi_model_inference": os.path.join(OUTDIR, "roi_model_r_joint_inference.csv"),
            "roi_model_comparisons": os.path.join(OUTDIR, "roi_model_comparisons.csv"),
            "hierarchy_model_values": os.path.join(OUTDIR, "hierarchy_model_values.csv"),
            "hierarchy_model_gain_values": os.path.join(OUTDIR, "hierarchy_model_gain_values.csv"),
            "main_figure_png": os.path.join(FIGDIR, "main_fig1_core_story.png"),
            "main_figure_pdf": os.path.join(FIGDIR, "main_fig1_core_story.pdf"),
            "metadata_json": os.path.join(OUTDIR, "roi_inference_model_comparison_calorie_finale_NC_legacyload_metadata.json"),
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "nibabel": nib.__version__,
        },
    }

    out_path = os.path.join(
        OUTDIR,
        "roi_inference_model_comparison_calorie_finale_NC_legacyload_metadata.json",
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  saved metadata: {out_path}")



# =============================================================================
# [4] MODEL DISCOVERY + LOADING
# =============================================================================

def subject_folder_name(sid):
    if isinstance(sid, str) and sid.startswith("sub-"):
        return sid
    return f"sub-{sid}"


def list_subjects_for_model(model_key, model_registry=MODELS):
    model_dir = model_registry[model_key]["dir"]
    if not os.path.isdir(model_dir):
        return []
    allowed = {subject_folder_name(s) for s in PILOT_SUBJECTS}
    subs = []
    for d in sorted(os.listdir(model_dir)):
        if d not in allowed:
            continue
        rj_path = os.path.join(model_dir, d, "r_joint.npy")
        if os.path.isfile(rj_path):
            subs.append(d)
    return subs

def resolve_available_models(
    model_registry=MODELS,
    require_all=REQUIRE_ALL_MODELS,
    registry_name="primary",
):
    print(f"\n[4] Resolving available models: {registry_name}")
    available = OrderedDict()
    missing = []

    for mk, spec in model_registry.items():
        subs = list_subjects_for_model(mk, model_registry=model_registry)
        if len(subs) == 0:
            missing.append(mk)
            print(f"  MISSING {mk:35s}: {spec['dir']}")
        else:
            available[mk] = {**spec, "subjects": subs}
            print(f"  FOUND   {mk:35s}: {len(subs):2d} subjects | {spec['dir']}")

    if missing and require_all:
        raise RuntimeError(
            f"Some {registry_name} models were missing. Missing: "
            + ", ".join(missing)
        )

    if len(available) < 2:
        raise RuntimeError(f"Need at least two available {registry_name} models.")

    return available
def common_subjects_across_models(available_models):
    subject_sets = [set(spec["subjects"]) for spec in available_models.values()]
    common       = sorted(set.intersection(*subject_sets))
    print(f"\nCommon subjects across available models ({len(common)}):")
    print(common)
    if len(common) == 0:
        raise RuntimeError("No common subjects across available models.")
    return common

def load_r_joint_map(model_key, sub, model_registry=MODELS):
    path = os.path.join(model_registry[model_key]["dir"], sub, "r_joint.npy")
    arr = np.load(path).astype(np.float32)
    if arr.shape[0] != N_GRAYORD:
        raise ValueError(f"{path} has shape {arr.shape}; expected ({N_GRAYORD},)")
    return arr[:N_CORTICAL]

def load_r_split_map(model_key, sub, fs):
    path = os.path.join(MODELS[model_key]["dir"], sub, f"r_split_{fs}.npy")
    arr  = np.load(path).astype(np.float32)
    if arr.shape[0] != N_GRAYORD:
        raise ValueError(f"{path} has shape {arr.shape}; expected ({N_GRAYORD},)")
    return arr[:N_CORTICAL]


def load_all_r_joint(available_models, subjects, model_registry=MODELS):
    print("\n[5] Loading r_joint maps")
    r_joint = {mk: {} for mk in available_models.keys()}
    for mk in available_models.keys():
        for sub in subjects:
            r_joint[mk][sub] = load_r_joint_map(
                mk, sub, model_registry=model_registry
            )
        print(f"  {mk:35s}: loaded {len(subjects)} subjects")
    return r_joint
# =============================================================================
# [5] ROI MODEL EXTRACTION + INFERENCE
# =============================================================================

def extract_roi_model_values(subjects, roi_masks, available_models, r_joint):
    print("\n[6] Extracting ROI model values")
    rows = []
    for sub in subjects:
        for roi_name, roi_mask in roi_masks.items():
            finite = roi_mask.copy()
            for mk in available_models.keys():
                finite &= np.isfinite(r_joint[mk][sub])
            n_vertices = int(finite.sum())
            if n_vertices < MIN_ROI_VERTICES:
                print(f"  WARNING: {sub} {roi_name}: only {n_vertices} finite vertices")
                continue
            for mk, spec in available_models.items():
                val = float(np.nanmean(r_joint[mk][sub][finite]))
                rows.append({
                    "subject":          sub,
                    "roi":              roi_name,
                    "roi_label":        ROI_LABELS.get(roi_name, roi_name),
                    "model":            mk,
                    "model_label":      spec["label"],
                    "model_short_label": spec["short_label"],
                    "n_vertices":       n_vertices,
                    "r_joint":          val,
                    "z_joint":          float(fisher_z(val)),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No ROI model rows extracted.")
    return df


def roi_model_r_joint_inference(model_df):
    rows = []
    for (roi, mk), subdf in model_df.groupby(["roi", "model"], sort=False):
        vals_r = subdf["r_joint"].to_numpy(dtype=float)
        vals_z = subdf["z_joint"].to_numpy(dtype=float)
        p      = signflip_1d(vals_z, alternative="greater", seed=RANDOM_STATE + 1)
        lo, hi = ci95(vals_r)
        rows.append({
            "roi":              roi,
            "roi_label":        subdf["roi_label"].iloc[0],
            "model":            mk,
            "model_label":      subdf["model_label"].iloc[0],
            "n_subjects":       int(np.isfinite(vals_z).sum()),
            "mean_r_joint":     float(np.nanmean(vals_r)),
            "sem_r_joint":      sem(vals_r),
            "ci95_lo_r_joint":  lo,
            "ci95_hi_r_joint":  hi,
            "mean_z_joint":     float(np.nanmean(vals_z)),
            "cohens_dz_z":      cohens_dz(vals_z),
            "p":                p,
            "sig_unc_05":       p < ALPHA,
        })
    return add_fdr(pd.DataFrame(rows), p_col="p")


def available_comparisons(available_models, comparison_registry=MODEL_COMPARISONS):
    comps = OrderedDict()
    for cname, (ma, mb, alt) in comparison_registry.items():
        if ma in available_models and mb in available_models:
            comps[cname] = (ma, mb, alt)
        else:
            print(f"  Skipping comparison {cname}: missing {ma} or {mb}")
    return comps

def roi_model_comparisons(model_df, comps):
    print("\n[7] ROI paired model comparisons")
    rows = []
    for roi, roi_df in model_df.groupby("roi", sort=False):
        for cname, (ma, mb, alt) in comps.items():
            wide = roi_df[roi_df["model"].isin([ma, mb])].pivot(
                index="subject", columns="model", values=["r_joint", "z_joint"],
            )
            if ("z_joint", ma) not in wide.columns or ("z_joint", mb) not in wide.columns:
                continue
            za = wide[("z_joint", ma)].to_numpy(dtype=float)
            zb = wide[("z_joint", mb)].to_numpy(dtype=float)
            ra = wide[("r_joint", ma)].to_numpy(dtype=float)
            rb = wide[("r_joint", mb)].to_numpy(dtype=float)
            good  = np.isfinite(za) & np.isfinite(zb) & np.isfinite(ra) & np.isfinite(rb)
            dz    = za[good] - zb[good]
            dr    = ra[good] - rb[good]
            p     = signflip_1d(dz, alternative=alt, seed=RANDOM_STATE + 2)
            lo, hi = ci95(dr)
            rows.append({
                "roi":              roi,
                "roi_label":        ROI_LABELS.get(roi, roi),
                "comparison":       cname,
                "comparison_label": COMPARISON_LABELS.get(cname, cname),
                "model_A":          ma,
                "model_B":          mb,
                "alternative":      alt,
                "n_subjects":       int(good.sum()),
                "mean_r_A":         float(np.nanmean(ra[good])) if good.sum() else np.nan,
                "mean_r_B":         float(np.nanmean(rb[good])) if good.sum() else np.nan,
                "mean_delta_r":     float(np.nanmean(dr))       if good.sum() else np.nan,
                "sem_delta_r":      sem(dr),
                "ci95_lo_delta_r":  lo,
                "ci95_hi_delta_r":  hi,
                "mean_delta_z":     float(np.nanmean(dz))       if good.sum() else np.nan,
                "cohens_dz":        cohens_dz(dz),
                "p":                p,
                "sig_unc_05":       p < ALPHA if np.isfinite(p) else False,
            })
    return add_fdr(pd.DataFrame(rows), p_col="p")


# =============================================================================
# [6] HIERARCHY ANALYSIS
# =============================================================================

def extract_hierarchy_model_values(model_df):
    rows = []
    for sub, subdf in model_df.groupby("subject"):
        for roi_name, level in VISUAL_HIERARCHY.items():
            tmp = subdf[subdf["roi"] == roi_name].copy()
            if tmp.empty:
                continue
            tmp["hierarchy_level"] = float(level)
            rows.append(tmp)
    if not rows:
        raise RuntimeError("No hierarchy rows extracted.")
    return pd.concat(rows, ignore_index=True)


def compute_hierarchy_model_gain_values(hdf, comps):
    rows = []
    for sub in sorted(hdf["subject"].unique()):
        sdf = hdf[hdf["subject"] == sub]
        for roi_name, level in VISUAL_HIERARCHY.items():
            rdf = sdf[sdf["roi"] == roi_name]
            for cname, (ma, mb, alt) in comps.items():
                ra = rdf.loc[rdf["model"] == ma, "r_joint"]
                rb = rdf.loc[rdf["model"] == mb, "r_joint"]
                za = rdf.loc[rdf["model"] == ma, "z_joint"]
                zb = rdf.loc[rdf["model"] == mb, "z_joint"]
                if len(ra) != 1 or len(rb) != 1 or len(za) != 1 or len(zb) != 1:
                    continue
                rows.append({
                    "subject":          sub,
                    "roi":              roi_name,
                    "roi_label":        ROI_LABELS.get(roi_name, roi_name),
                    "hierarchy_level":  float(level),
                    "comparison":       cname,
                    "comparison_label": COMPARISON_LABELS.get(cname, cname),
                    "model_A":          ma,
                    "model_B":          mb,
                    "alternative":      alt,
                    "delta_r":          float(ra.iloc[0] - rb.iloc[0]),
                    "delta_z":          float(za.iloc[0] - zb.iloc[0]),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No hierarchy gain rows computed.")
    return out


def hierarchy_gain_slopes(gain_df, use_comparisons=None):
    rows  = []
    comps = use_comparisons if use_comparisons is not None else sorted(gain_df["comparison"].unique())
    for cname in comps:
        cdf = gain_df[gain_df["comparison"] == cname]
        if cdf.empty:
            continue
        for sub, sdf in cdf.groupby("subject"):
            dat = sdf[["hierarchy_level", "delta_z", "delta_r"]].dropna()
            if len(dat) < len(VISUAL_HIERARCHY):
                continue
            lr_z = stats.linregress(dat["hierarchy_level"].to_numpy(dtype=float),
                                    dat["delta_z"].to_numpy(dtype=float))
            lr_r = stats.linregress(dat["hierarchy_level"].to_numpy(dtype=float),
                                    dat["delta_r"].to_numpy(dtype=float))
            rows.append({
                "subject":                  sub,
                "comparison":               cname,
                "comparison_label":         COMPARISON_LABELS.get(cname, cname),
                "model_A":                  cdf["model_A"].iloc[0],
                "model_B":                  cdf["model_B"].iloc[0],
                "slope_delta_z_per_step":   float(lr_z.slope),
                "slope_delta_r_per_step":   float(lr_r.slope),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No hierarchy gain slopes computed.")
    return out


def hierarchy_gain_slope_tests(slope_df, comps):
    rows = []
    for cname, subdf in slope_df.groupby("comparison", sort=False):
        alt      = comps[cname][2]
        slopes_z = subdf["slope_delta_z_per_step"].to_numpy(dtype=float)
        slopes_r = subdf["slope_delta_r_per_step"].to_numpy(dtype=float)

        p_greater = signflip_1d(slopes_z, alternative="greater",   seed=RANDOM_STATE + 30)
        p_less    = signflip_1d(slopes_z, alternative="less",      seed=RANDOM_STATE + 31)
        p_two     = signflip_1d(slopes_z, alternative="two-sided", seed=RANDOM_STATE + 32)

        p_expected = p_greater if alt == "greater" else (p_less if alt == "less" else p_two)

        rows.append({
            "comparison":                   cname,
            "comparison_label":             COMPARISON_LABELS.get(cname, cname),
            "model_A":                      comps[cname][0],
            "model_B":                      comps[cname][1],
            "n_subjects":                   int(np.isfinite(slopes_z).sum()),
            "mean_slope_delta_z_per_step":  float(np.nanmean(slopes_z)),
            "mean_slope_delta_r_per_step":  float(np.nanmean(slopes_r)),
            "cohens_dz_slope_z":            cohens_dz(slopes_z),
            "p_expected":                   p_expected,
            "sig_unc_05_expected":          p_expected < ALPHA if np.isfinite(p_expected) else False,
        })
    return add_fdr(pd.DataFrame(rows), p_col="p_expected")


def hierarchy_gain_pairwise_tests(gain_df, comps):
    pair_defs = OrderedDict([
        ("Intermediate_gt_Early",    ("IntermediateVisual", "EarlyVisual")),
        ("HighLevelVTC_gt_Early",         ("HighLevelVTC",            "EarlyVisual")),
        ("HighLevelVTC_gt_Intermediate",  ("HighLevelVTC",            "IntermediateVisual")),
    ])

    rows = []
    for cname, cdf in gain_df.groupby("comparison", sort=False):
        alt = comps[cname][2]
        for pair_name, (roi_a, roi_b) in pair_defs.items():
            vals_a, vals_b, raw_a, raw_b = [], [], [], []
            for sub, sdf in cdf.groupby("subject"):
                a  = sdf.loc[sdf["roi"] == roi_a, "delta_z"]
                b  = sdf.loc[sdf["roi"] == roi_b, "delta_z"]
                ar = sdf.loc[sdf["roi"] == roi_a, "delta_r"]
                br = sdf.loc[sdf["roi"] == roi_b, "delta_r"]
                if len(a) != 1 or len(b) != 1 or len(ar) != 1 or len(br) != 1:
                    continue
                vals_a.append(float(a.iloc[0]))
                vals_b.append(float(b.iloc[0]))
                raw_a.append(float(ar.iloc[0]))
                raw_b.append(float(br.iloc[0]))

            vals_a = np.asarray(vals_a, dtype=float)
            vals_b = np.asarray(vals_b, dtype=float)
            raw_a  = np.asarray(raw_a,  dtype=float)
            raw_b  = np.asarray(raw_b,  dtype=float)
            delta  = vals_a - vals_b
            p      = signflip_1d(delta, alternative=alt, seed=RANDOM_STATE + 40)

            rows.append({
                "comparison":                    cname,
                "comparison_label":              COMPARISON_LABELS.get(cname, cname),
                "pair":                          pair_name,
                "roi_A":                         roi_a,
                "roi_B":                         roi_b,
                "alternative":                   alt,
                "n_subjects":                    len(delta),
                "mean_pair_delta_z_A_minus_B":   float(np.nanmean(delta))        if len(delta) else np.nan,
                "mean_pair_delta_r_A_minus_B":   float(np.nanmean(raw_a - raw_b)) if len(delta) else np.nan,
                "cohens_dz":                     cohens_dz(delta),
                "p":                             p,
                "sig_unc_05":                    p < ALPHA if np.isfinite(p) else False,
            })
    return add_fdr(pd.DataFrame(rows), p_col="p")


# =============================================================================
# [7] OPTIONAL SPLIT-MODEL INFERENCE
# =============================================================================

def split_model_available(available_models, subjects):
    if SPLIT_MODEL_KEY is None:
        return False
    if SPLIT_MODEL_KEY not in available_models:
        print(f"\nSplit-model inference disabled: {SPLIT_MODEL_KEY} unavailable.")
        return False
    for sub in subjects:
        for fs in SPLIT_FEATURE_SPACES:
            p = os.path.join(MODELS[SPLIT_MODEL_KEY]["dir"], sub, f"r_split_{fs}.npy")
            if not os.path.isfile(p):
                print(f"\nSplit-model inference disabled: missing {p}")
                return False
    return True


def extract_split_roi_values(subjects, roi_masks, r_joint):
    print(f"\n[8] Extracting split-model ROI values for {SPLIT_MODEL_KEY}")
    rows = []
    for sub in subjects:
        split_maps = {fs: load_r_split_map(SPLIT_MODEL_KEY, sub, fs) for fs in SPLIT_FEATURE_SPACES}
        rj = r_joint[SPLIT_MODEL_KEY][sub]
        for roi_name, roi_mask in roi_masks.items():
            finite = roi_mask & np.isfinite(rj)
            for fs in SPLIT_FEATURE_SPACES:
                finite &= np.isfinite(split_maps[fs])
            n_vertices = int(finite.sum())
            if n_vertices < MIN_ROI_VERTICES:
                continue
            row = {
                "subject":   sub,
                "roi":       roi_name,
                "roi_label": ROI_LABELS.get(roi_name, roi_name),
                "n_vertices": n_vertices,
                "r_joint":   float(np.nanmean(rj[finite])),
            }
            for fs in SPLIT_FEATURE_SPACES:
                row[f"r_split_{fs}"] = float(np.nanmean(split_maps[fs][finite]))
            row["r_split_Visual"]       = row["r_split_LowVis"] + row["r_split_HighVis"]
            row["r_split_CalorieTotal"] = (
                row["r_split_CaloriePredCLIP"] + row["r_split_CalorieResCLIP"]
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No split-model ROI rows extracted.")
    return out


def split_band_inference(split_df):
    rows = []
    for roi, subdf in split_df.groupby("roi", sort=False):
        for fs in SPLIT_FEATURE_SPACES:
            vals   = subdf[f"r_split_{fs}"].to_numpy(dtype=float)
            p      = signflip_1d(vals, alternative="greater", seed=RANDOM_STATE + 50)
            lo, hi = ci95(vals)
            rows.append({
                "roi":           roi,
                "roi_label":     ROI_LABELS.get(roi, roi),
                "band":          fs,
                "band_label":    SPLIT_BAND_LABELS.get(fs, fs),
                "mean_r_split":  float(np.nanmean(vals)),
                "sem_r_split":   sem(vals),
                "ci95_lo_r_split": lo,
                "ci95_hi_r_split": hi,
                "cohens_dz":     cohens_dz(vals),
                "p":             p,
                "sig_unc_05":    p < ALPHA if np.isfinite(p) else False,
            })
    return add_fdr(pd.DataFrame(rows), p_col="p")


def split_contrast_inference(split_df):
    rows = []
    for roi, subdf in split_df.groupby("roi", sort=False):
        for cname, (metric_a, metric_b, alt) in SPLIT_CONTRASTS.items():
            a      = subdf[metric_a].to_numpy(dtype=float)
            b      = subdf[metric_b].to_numpy(dtype=float)
            delta  = a - b
            p      = signflip_1d(delta, alternative=alt, seed=RANDOM_STATE + 51)
            lo, hi = ci95(delta)
            rows.append({
                "roi":             roi,
                "roi_label":       ROI_LABELS.get(roi, roi),
                "contrast":        cname,
                "contrast_label":  SPLIT_CONTRAST_LABELS.get(cname, cname),
                "mean_A":          float(np.nanmean(a)),
                "mean_B":          float(np.nanmean(b)),
                "mean_delta":      float(np.nanmean(delta)),
                "sem_delta":       sem(delta),
                "ci95_lo_delta":   lo,
                "ci95_hi_delta":   hi,
                "cohens_dz":       cohens_dz(delta),
                "p":               p,
                "sig_unc_05":      p < ALPHA if np.isfinite(p) else False,
            })
    return add_fdr(pd.DataFrame(rows), p_col="p")


def compute_split_relative_importance(split_df, rois=None):
    if rois is None:
        rois = list(split_df["roi"].unique())
    rows = []
    for roi in rois:
        sdf = split_df[split_df["roi"] == roi].copy()
        if sdf.empty:
            continue
        mat   = np.column_stack([sdf[f"r_split_{fs}"].to_numpy(dtype=float) for fs in SPLIT_FEATURE_SPACES])
        pos   = np.maximum(mat, 0.0)
        denom = np.sum(pos, axis=1)
        shares = np.full_like(pos, np.nan, dtype=float)
        valid  = denom > 0
        shares[valid] = pos[valid] / denom[valid, None]
        for j, fs in enumerate(SPLIT_FEATURE_SPACES):
            vals = shares[:, j]
            vals = vals[np.isfinite(vals)]
            lo, hi = ci95(vals)
            rows.append({
                "roi":              roi,
                "roi_label":        ROI_LABELS.get(roi, roi),
                "band":             fs,
                "band_label":       SPLIT_BAND_LABELS.get(fs, fs),
                "n_subjects_valid": int(vals.size),
                "mean_share":       float(np.nanmean(vals)) if vals.size else np.nan,
                "sem_share":        sem(vals),
                "ci95_lo_share":    lo,
                "ci95_hi_share":    hi,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No relative-importance rows computed.")
    return out


# =============================================================================
# [8] PLOT HELPERS
# =============================================================================

def set_plot_defaults():
    plt.rcParams.update({
        "font.size":        10,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  10,
        "figure.titlesize": 15,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })


def roi_label(roi):
    return ROI_LABELS.get(roi, roi)


def hierarchy_rois():
    return list(VISUAL_HIERARCHY.keys())


def panel_label(ax, label):
    ax.text(-0.18, 1.08, label, transform=ax.transAxes,
            fontsize=20, fontweight="bold", ha="left", va="top")


def jitter_positions(n, width=0.08, seed=RANDOM_STATE):
    if n <= 1:
        return np.zeros(n)
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, size=n)


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    m      = float(np.mean(values))
    lo, hi = ci95(values)
    return m, lo, hi


def nice_ylim(values, include_zero=True, pad_fraction=0.18, min_span=0.04):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        lo, hi = -min_span, min_span
    else:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    span = max(hi - lo, min_span)
    pad  = pad_fraction * span
    return lo - pad, hi + pad


def draw_bracket(ax, x1, x2, y, h, txt):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            color="black", linewidth=1.1, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, txt, ha="center", va="bottom", fontsize=12)


def get_rjoint_row(rjoint_inf, roi, model):
    row = rjoint_inf[(rjoint_inf["roi"] == roi) & (rjoint_inf["model"] == model)]
    if len(row) != 1:
        return None
    return row.iloc[0].to_dict()


def get_comp_row(comp_df, roi, comparison):
    row = comp_df[(comp_df["roi"] == roi) & (comp_df["comparison"] == comparison)]
    if len(row) != 1:
        return None
    return row.iloc[0].to_dict()


def get_pair_row(pairwise_df, comparison, pair):
    row = pairwise_df[(pairwise_df["comparison"] == comparison) & (pairwise_df["pair"] == pair)]
    if len(row) != 1:
        return None
    return row.iloc[0].to_dict()


def get_model_values(model_df, roi, model):
    vals = model_df.loc[
        (model_df["roi"] == roi) & (model_df["model"] == model),
        "r_joint"
    ].to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def get_gain_values(gain_df, roi, comparison):
    vals = gain_df.loc[
        (gain_df["roi"] == roi) & (gain_df["comparison"] == comparison),
        "delta_r"
    ].to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def draw_nc_band(ax, i, nc_entry, x_half_width=0.45, color=NC_COLOR, alpha=NC_ALPHA):
    """
    Draw a noise ceiling shaded band for ROI at x-position i.

    The band spans [lo, hi] (mean ± 1 SEM across subjects) over the bar group.
    A horizontal line marks the mean.

    nc_entry : dict with keys 'mean', 'lo', 'hi' — or None to skip.
    """
    if nc_entry is None:
        return
    lo  = nc_entry["lo"]
    hi  = nc_entry["hi"]
    mn  = nc_entry["mean"]
    ax.fill_between(
        [i - x_half_width, i + x_half_width],
        [lo, lo], [hi, hi],
        color=color, alpha=alpha, linewidth=0, zorder=0,
    )
    ax.hlines(mn, i - x_half_width, i + x_half_width,
              colors=color, linewidths=1.2, linestyles="--", zorder=1, alpha=0.6)


# =============================================================================
# [9] MAIN FIGURE
# =============================================================================

def plot_main_fig_core_story(
    model_df, rjoint_inf, comp_df, gain_df, pairwise_df,
    nc_roi=None,
):
    """
    Main figure with noise ceiling and M1 added.

    Panel A  Visual baseline (M0) per ROI with noise ceiling band.
    Panel B  M0 / M1 / M2 three-bar groups per ROI with noise ceiling band.
             Shows M1 ≈ M2 > M0 directly.
    Panel C  Model gain Δr (M2-M0 bars + M1-M0 dashed line overlay).
             The two lines overlap, making the M1 ≈ M2 point visually explicit.
    """
    if nc_roi is None:
        nc_roi = {}

    rois        = hierarchy_rois()
    x           = np.arange(len(rois))
    m0          = "M0_visual"
    m1          = "M1_visual_rawcal"
    m2          = "M2_visual_predclip"
    gain_comp   = MAIN_GAIN_COMPARISON
    gain_comp_m1 = M1_GAIN_COMPARISON

    fig = plt.figure(figsize=(17.5, 5.8), constrained_layout=True)
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.55, 1.15])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    audit_rows = []

    # ── NC legend handle (single entry, drawn once) ───────────────────────────
    from matplotlib.patches import Patch
    nc_handle = Patch(facecolor=NC_COLOR, alpha=NC_ALPHA * 3,
                      edgecolor=NC_COLOR, linewidth=0.5, label="Noise ceiling")

    # =========================================================================
    # Panel A — Visual baseline with noise ceiling
    # =========================================================================
    panel_label(axA, "A")

    values_A = []
    for roi in rois:
        values_A.extend(get_model_values(model_df, roi, m0))
        nc = nc_roi.get(roi)
        if nc is not None:
            values_A.append(nc["hi"])
    yloA, yhiA = nice_ylim(values_A, include_zero=True, pad_fraction=0.22, min_span=0.05)

    for i, roi in enumerate(rois):
        # noise ceiling band
        draw_nc_band(axA, i, nc_roi.get(roi), x_half_width=0.40)

        vals     = get_model_values(model_df, roi, m0)
        m, lo, hi = mean_ci(vals)

        axA.bar(i, m, width=0.60, color=MODELS[m0]["color"],
                alpha=0.90, edgecolor="white", linewidth=0.8)
        axA.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none",
                     color="black", capsize=3, linewidth=1.1)

        jit = jitter_positions(len(vals), width=0.09, seed=RANDOM_STATE + i)
        axA.scatter(np.full(len(vals), i) + jit, vals,
                    s=POINT_SIZE, facecolors="white", edgecolors="0.25",
                    linewidth=0.9, alpha=POINT_ALPHA, zorder=3)

        row = get_rjoint_row(rjoint_inf, roi, m0)
        if should_mark(row, use_fdr_only=MAINFIG_USE_FDR_ONLY):
            txt = mark_text(row, use_fdr_only=MAINFIG_USE_FDR_ONLY)
            y   = hi + 0.04 * (yhiA - yloA)
            axA.text(i, y, txt, ha="center", va="bottom", fontsize=13)

        audit_rows.append({
            "panel": "A", "roi": roi, "effect": "M0_visual_gt_0",
            "mean": m, "ci_lo": lo, "ci_hi": hi,
            "p": safe_float(row["p"])       if row is not None else np.nan,
            "q_fdr_bh": safe_float(row["q_fdr_bh"]) if row is not None else np.nan,
            "marked_in_plot": should_mark(row, use_fdr_only=MAINFIG_USE_FDR_ONLY),
        })

    axA.axhline(0, color="black", linewidth=0.9)
    axA.set_xlim(-0.6, len(rois) - 0.4)
    axA.set_ylim(yloA, yhiA)
    axA.set_xticks(x)
    axA.set_xticklabels([roi_label(r) for r in rois], rotation=20, ha="right")
    axA.set_ylabel(r"Mean held-out $r_{joint}$")
    axA.set_title("Visual baseline encoding")
    axA.grid(axis="y", color="0.90", linewidth=0.8)

    # noise ceiling legend entry in Panel A
    if any(nc_roi.get(r) is not None for r in rois):
        axA.legend(handles=[nc_handle], frameon=False, loc="upper left",
                   fontsize=9)

    # =========================================================================
    # Panel B — M0 / M1 / M2 three-bar groups with noise ceiling
    # =========================================================================
    panel_label(axB, "B")

    width = 0.22
    off0  = -0.25   # M0
    off1  =  0.00   # M1
    off2  = +0.25   # M2

    values_B = []
    for roi in rois:
        for mk in [m0, m1, m2]:
            values_B.extend(get_model_values(model_df, roi, mk))
        nc = nc_roi.get(roi)
        if nc is not None:
            values_B.append(nc["hi"])
    yloB, yhiB = nice_ylim(values_B, include_zero=True, pad_fraction=0.30, min_span=0.06)

    for i, roi in enumerate(rois):
        # noise ceiling band spanning the whole bar group
        draw_nc_band(axB, i, nc_roi.get(roi), x_half_width=0.45)

        max_hi = -np.inf

        for mk, off in [(m0, off0), (m1, off1), (m2, off2)]:
            vals      = get_model_values(model_df, roi, mk)
            m, lo, hi = mean_ci(vals)
            xpos      = i + off
            color     = MODELS[mk]["color"]

            axB.bar(xpos, m, width=width, color=color, alpha=0.90,
                    edgecolor="white", linewidth=0.8,
                    label=MODELS[mk]["label"] if i == 0 else None)
            axB.errorbar(xpos, m, yerr=[[m - lo], [hi - m]], fmt="none",
                         color="black", capsize=3, linewidth=1.1)

            jit = jitter_positions(
                len(vals), width=0.038,
                seed=RANDOM_STATE + 10 * i + [m0, m1, m2].index(mk)
            )
            axB.scatter(np.full(len(vals), xpos) + jit, vals,
                        s=POINT_SIZE - 4, facecolors="white", edgecolors="0.25",
                        linewidth=0.7, alpha=POINT_ALPHA, zorder=3)

            row = get_rjoint_row(rjoint_inf, roi, mk)
            if should_mark(row, use_fdr_only=MAINFIG_USE_FDR_ONLY):
                txt = mark_text(row, use_fdr_only=MAINFIG_USE_FDR_ONLY)
                y   = hi + 0.035 * (yhiB - yloB)
                axB.text(xpos, y, txt, ha="center", va="bottom", fontsize=11)

            max_hi = max(max_hi, hi)

            audit_rows.append({
                "panel": "B", "roi": roi, "effect": f"{mk}_gt_0",
                "mean": m, "ci_lo": lo, "ci_hi": hi,
                "p": safe_float(row["p"])       if row is not None else np.nan,
                "q_fdr_bh": safe_float(row["q_fdr_bh"]) if row is not None else np.nan,
                "marked_in_plot": should_mark(row, use_fdr_only=MAINFIG_USE_FDR_ONLY),
            })

        # M1 > M0 bracket
        crow_m1 = get_comp_row(comp_df, roi, gain_comp_m1)
        if should_mark(crow_m1, use_fdr_only=MAINFIG_USE_FDR_ONLY):
            txt = mark_text(crow_m1, use_fdr_only=MAINFIG_USE_FDR_ONLY)
            y   = max_hi + 0.08 * (yhiB - yloB)
            h   = 0.020 * (yhiB - yloB)
            draw_bracket(axB, i + off0, i + off1, y, h, txt)
        
        # M2 > M0 bracket
        crow_m2 = get_comp_row(comp_df, roi, gain_comp)
        if should_mark(crow_m2, use_fdr_only=MAINFIG_USE_FDR_ONLY):
            txt = mark_text(crow_m2, use_fdr_only=MAINFIG_USE_FDR_ONLY)
            y   = max_hi + 0.16 * (yhiB - yloB)
            h   = 0.020 * (yhiB - yloB)
            draw_bracket(axB, i + off0, i + off2, y, h, txt)

    audit_rows.append({
        "panel": "B", "roi": roi, "effect": gain_comp_m1,
        "mean": safe_float(crow_m1["mean_delta_r"]) if crow_m1 is not None else np.nan,
        "ci_lo": safe_float(crow_m1["ci95_lo_delta_r"]) if crow_m1 is not None else np.nan,
        "ci_hi": safe_float(crow_m1["ci95_hi_delta_r"]) if crow_m1 is not None else np.nan,
        "p": safe_float(crow_m1["p"]) if crow_m1 is not None else np.nan,
        "q_fdr_bh": safe_float(crow_m1["q_fdr_bh"]) if crow_m1 is not None else np.nan,
        "marked_in_plot": should_mark(crow_m1, use_fdr_only=MAINFIG_USE_FDR_ONLY),
    })
    
    audit_rows.append({
        "panel": "B", "roi": roi, "effect": gain_comp,
        "mean": safe_float(crow_m2["mean_delta_r"]) if crow_m2 is not None else np.nan,
        "ci_lo": safe_float(crow_m2["ci95_lo_delta_r"]) if crow_m2 is not None else np.nan,
        "ci_hi": safe_float(crow_m2["ci95_hi_delta_r"]) if crow_m2 is not None else np.nan,
        "p": safe_float(crow_m2["p"]) if crow_m2 is not None else np.nan,
        "q_fdr_bh": safe_float(crow_m2["q_fdr_bh"]) if crow_m2 is not None else np.nan,
        "marked_in_plot": should_mark(crow_m2, use_fdr_only=MAINFIG_USE_FDR_ONLY),
    })
    axB.axhline(0, color="black", linewidth=0.9)
    axB.set_xlim(-0.6, len(rois) - 0.4)
    axB.set_ylim(yloB, yhiB)
    axB.set_xticks(x)
    axB.set_xticklabels([roi_label(r) for r in rois], rotation=20, ha="right")
    axB.set_ylabel(r"Mean held-out $r_{joint}$")
    axB.set_title("Calorie-related models improve visual encoding")
    axB.grid(axis="y", color="0.90", linewidth=0.8)

    handles, labels = axB.get_legend_handles_labels()
    if any(nc_roi.get(r) is not None for r in rois):
        handles.append(nc_handle)
        labels.append("Noise ceiling")
    axB.legend(handles, labels, frameon=False, loc="upper left", fontsize=9)

    # =========================================================================
    # Panel C — Gain (M2-M0 bars + M1-M0 dashed line overlay)
    #
    # The two lines will visually overlap because M1 ≈ M2, making the key
    # point — that PredCLIP recovers all of the calorie gain — immediately
    # apparent from the figure without needing a statistical test.
    # =========================================================================
    panel_label(axC, "C")

    values_C = []
    for roi in rois:
        values_C.extend(get_gain_values(gain_df, roi, gain_comp))
        values_C.extend(get_gain_values(gain_df, roi, gain_comp_m1))
    yloC, yhiC = nice_ylim(values_C, include_zero=True, pad_fraction=0.28, min_span=0.025)

    meansC_m2 = []
    meansC_m1 = []
    hisC      = []

    for i, roi in enumerate(rois):
        # M2-M0 bars (primary)
        vals_m2     = get_gain_values(gain_df, roi, gain_comp)
        m2_, lo, hi = mean_ci(vals_m2)
        meansC_m2.append(m2_)
        hisC.append(hi)

        axC.bar(i, m2_, width=0.58, color=MODELS[m2]["color"],
                alpha=0.85, edgecolor="white", linewidth=0.8,
                label="M2−M0" if i == 0 else None)
        axC.errorbar(i, m2_, yerr=[[m2_ - lo], [hi - m2_]], fmt="none",
                     color="black", capsize=3, linewidth=1.1)

        jit = jitter_positions(len(vals_m2), width=0.09, seed=RANDOM_STATE + 100 + i)
        axC.scatter(np.full(len(vals_m2), i) + jit, vals_m2,
                    s=POINT_SIZE, facecolors="white", edgecolors="0.25",
                    linewidth=0.8, alpha=POINT_ALPHA, zorder=3)

        crow = get_comp_row(comp_df, roi, gain_comp)
        if should_mark(crow, use_fdr_only=MAINFIG_USE_FDR_ONLY):
            txt = mark_text(crow, use_fdr_only=MAINFIG_USE_FDR_ONLY)
            y   = hi + 0.04 * (yhiC - yloC)
            axC.text(i, y, txt, ha="center", va="bottom", fontsize=13)

        # M1-M0 mean point
        vals_m1 = get_gain_values(gain_df, roi, gain_comp_m1)
        m1_     = float(np.nanmean(vals_m1)) if vals_m1.size else np.nan
        meansC_m1.append(m1_)

        audit_rows.append({
            "panel": "C", "roi": roi, "effect": gain_comp,
            "mean": m2_, "ci_lo": lo, "ci_hi": hi,
            "p": safe_float(crow["p"])       if crow is not None else np.nan,
            "q_fdr_bh": safe_float(crow["q_fdr_bh"]) if crow is not None else np.nan,
            "marked_in_plot": should_mark(crow, use_fdr_only=MAINFIG_USE_FDR_ONLY),
        })

    # M2-M0 connecting line
    axC.plot(x, meansC_m2, color="#9D3F3F", linewidth=2.2, alpha=0.95, zorder=4)

    # M1-M0 dashed line overlay — will visually overlap with M2-M0 line
    valid_m1 = [(i, v) for i, v in enumerate(meansC_m1) if np.isfinite(v)]
    if valid_m1:
        xs_m1 = [v[0] for v in valid_m1]
        ys_m1 = [v[1] for v in valid_m1]
        axC.plot(xs_m1, ys_m1,
                 color=MODELS[m1]["color"],
                 linewidth=2.2,
                 linestyle="--",
                 alpha=0.85,
                 zorder=5,
                 label="M1−M0")

    # HighLevelVTC > EarlyVisual bracket
    pair_row = get_pair_row(pairwise_df, gain_comp, "HighLevelVTC_gt_Early")
    if should_mark(pair_row, use_fdr_only=MAINFIG_USE_FDR_ONLY):
        txt = mark_text(pair_row, use_fdr_only=MAINFIG_USE_FDR_ONLY)
        y   = max(hisC[0], hisC[2]) + 0.10 * (yhiC - yloC)
        h   = 0.025 * (yhiC - yloC)
        draw_bracket(axC, 0, 2, y, h, txt)

    audit_rows.append({
        "panel": "C", "roi": "HighLevelVTC_vs_EarlyVisual",
        "effect": "HighLevelVTC_gt_Early_for_M2_minus_M0",
        "mean": safe_float(pair_row["mean_pair_delta_r_A_minus_B"]) if pair_row is not None else np.nan,
        "ci_lo": np.nan, "ci_hi": np.nan,
        "p": safe_float(pair_row["p"])       if pair_row is not None else np.nan,
        "q_fdr_bh": safe_float(pair_row["q_fdr_bh"]) if pair_row is not None else np.nan,
        "marked_in_plot": should_mark(pair_row, use_fdr_only=MAINFIG_USE_FDR_ONLY),
    })

    axC.axhline(0, color="black", linewidth=0.9)
    axC.set_xlim(-0.6, len(rois) - 0.4)
    axC.set_ylim(yloC, yhiC)
    axC.set_xticks(x)
    axC.set_xticklabels([roi_label(r) for r in rois], rotation=20, ha="right")
    axC.set_ylabel(r"Model gain, $\Delta r_{joint}$")
    axC.set_title("Calorie gain increases along the visual hierarchy")
    axC.grid(axis="y", color="0.90", linewidth=0.8)
    axC.legend(frameon=False, loc="upper left", fontsize=9)

    savefig(fig, "main_fig1_core_story")
    plt.close(fig)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(os.path.join(OUTDIR, "plot_annotation_audit.csv"), index=False)
    print(f"  saved: {os.path.join(OUTDIR, 'plot_annotation_audit.csv')}")


# =============================================================================
# [10] SUPPLEMENTARY FIGURE 1
# =============================================================================

def plot_supp_fig_predclip_vs_resclip_split(split_df, split_band_df, split_contrast_df):
    rois        = list(ROI_GROUPS.keys())
    x           = np.arange(len(rois))
    pred_metric = "r_split_CaloriePredCLIP"
    res_metric  = "r_split_CalorieResCLIP"

    fig, ax = plt.subplots(figsize=(11.5, 5.4), constrained_layout=True)

    width    = 0.28
    off_pred = -0.17
    off_res  = +0.17

    allvals = []
    allvals.extend(split_df[pred_metric].to_numpy(dtype=float))
    allvals.extend(split_df[res_metric].to_numpy(dtype=float))
    ylo, yhi = nice_ylim(allvals, include_zero=True, pad_fraction=0.30, min_span=0.03)

    for i, roi in enumerate(rois):
        max_hi = -np.inf
        for metric, off, key in [
            (pred_metric, off_pred, "CaloriePredCLIP"),
            (res_metric,  off_res,  "CalorieResCLIP"),
        ]:
            vals      = split_df.loc[split_df["roi"] == roi, metric].to_numpy(dtype=float)
            vals      = vals[np.isfinite(vals)]
            m, lo, hi = mean_ci(vals)
            xpos      = i + off

            ax.bar(xpos, m, width=width, color=SPLIT_BAND_COLORS[key],
                   alpha=0.88, edgecolor="white", linewidth=0.8,
                   label=SPLIT_BAND_LABELS[key] if i == 0 else None)
            ax.errorbar(xpos, m, yerr=[[m - lo], [hi - m]], fmt="none",
                        color="black", capsize=3, linewidth=1.1)

            jit = jitter_positions(
                len(vals), width=0.045,
                seed=RANDOM_STATE + 200 + 10 * i + (0 if key == "CaloriePredCLIP" else 1)
            )
            ax.scatter(np.full(len(vals), xpos) + jit, vals,
                       s=POINT_SIZE - 3, facecolors="white", edgecolors="0.25",
                       linewidth=0.8, alpha=POINT_ALPHA, zorder=3)

            brow = split_band_df[(split_band_df["roi"] == roi) & (split_band_df["band"] == key)]
            row  = brow.iloc[0].to_dict() if len(brow) == 1 else None
            if should_mark(row, use_fdr_only=SUPPFIG_USE_FDR_ONLY):
                txt = mark_text(row, use_fdr_only=SUPPFIG_USE_FDR_ONLY)
                y   = hi + 0.04 * (yhi - ylo)
                ax.text(xpos, y, txt, ha="center", va="bottom", fontsize=12)

            max_hi = max(max_hi, hi)

        crow = split_contrast_df[
            (split_contrast_df["roi"] == roi) &
            (split_contrast_df["contrast"] == "PredCLIP_gt_ResCLIP")
        ]
        row = crow.iloc[0].to_dict() if len(crow) == 1 else None
        if should_mark(row, use_fdr_only=SUPPFIG_USE_FDR_ONLY):
            txt = mark_text(row, use_fdr_only=SUPPFIG_USE_FDR_ONLY)
            y   = max_hi + 0.10 * (yhi - ylo)
            h   = 0.025 * (yhi - ylo)
            draw_bracket(ax, i + off_pred, i + off_res, y, h, txt)

    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xlim(-0.6, len(rois) - 0.4)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks(x)
    ax.set_xticklabels([roi_label(r) for r in rois], rotation=20, ha="right")
    ax.set_ylabel(r"M4 split contribution, $r_{split}$")
    ax.set_title("Supplementary: PredCLIP versus ResCLIP split contributions")
    ax.grid(axis="y", color="0.90", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")

    savefig(fig, "supp_fig1_predclip_vs_resclip_split")
    plt.close(fig)


# =============================================================================
# [11] SUPPLEMENTARY FIGURE 2
# =============================================================================

def nice_ylim_from_values(values, pad=0.12, min_span=0.001, include_zero=True):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        lo, hi = -min_span / 2, min_span / 2
    else:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    span = max(hi - lo, min_span)
    return lo - pad * span, hi + pad * span


def plot_supp_fig_hierarchy_split_contributions(
    split_df, split_band_df, split_contrast_df, relative_df
):
    if "HIERARCHY_SPLIT_ROIS" in globals():
        rois = [r for r in HIERARCHY_SPLIT_ROIS if r in set(split_df["roi"])]
    else:
        rois = [r for r in VISUAL_HIERARCHY.keys() if r in set(split_df["roi"])]

    if len(rois) == 0:
        raise RuntimeError("No hierarchy ROIs found in split_df.")

    bands = list(SPLIT_FEATURE_SPACES)
    x     = np.arange(len(rois), dtype=float)

    def _panel_label(ax, label):
        ax.text(-0.14, 1.04, label, transform=ax.transAxes,
                fontsize=17, fontweight="bold", ha="left", va="top", clip_on=False)

    def _mean_ci(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.nan, np.nan, np.nan
        m      = float(np.mean(vals))
        lo, hi = ci95(vals)
        return m, lo, hi

    def _fdr_stars(row):
        if row is None:
            return ""
        q = safe_float(row.get("q_fdr_bh", np.nan))
        if not np.isfinite(q) or q >= ALPHA:
            return ""
        return p_to_stars(q=q)

    def _draw_bracket(ax, x1, x2, y, h, text):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
                color="black", linewidth=1.0, clip_on=False)
        ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
                fontsize=11, clip_on=False)

    fig = plt.figure(figsize=(15.8, 6.6), constrained_layout=False)
    gs  = fig.add_gridspec(
        1, 2, width_ratios=[1.55, 1.0],
        left=0.07, right=0.985, bottom=0.21, top=0.78, wspace=0.32,
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    fig.suptitle("M4 split contributions across the visual hierarchy",
                 y=0.965, fontsize=14)

    _panel_label(axA, "A")

    offsets = np.linspace(-0.31, 0.31, len(bands))
    width   = 0.15

    allvals = []
    for roi in rois:
        for fs in bands:
            allvals.extend(
                split_df.loc[split_df["roi"] == roi, f"r_split_{fs}"].to_numpy(dtype=float)
            )
    allvals = np.asarray(allvals, dtype=float)[np.isfinite(np.asarray(allvals, dtype=float))]
    if allvals.size == 0:
        yloA, yhiA = -0.01, 0.01
    else:
        yloA, yhiA = nice_ylim_from_values(allvals, pad=0.28, min_span=0.035)
        yloA = min(yloA, -0.002)
        yhiA = max(yhiA, 0.012)

    yrangeA       = yhiA - yloA
    annotation_ymax = yhiA
    legend_handles  = []
    legend_labels   = []

    for i, roi in enumerate(rois):
        roi_bar_hi = -np.inf
        for j, fs in enumerate(bands):
            vals = split_df.loc[split_df["roi"] == roi, f"r_split_{fs}"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            m, lo, hi = _mean_ci(vals)
            xpos = x[i] + offsets[j]

            bar = axA.bar(xpos, m, width=width, color=SPLIT_BAND_COLORS[fs],
                          alpha=0.90, edgecolor="white", linewidth=0.8,
                          label=SPLIT_BAND_LABELS.get(fs, fs), zorder=2)
            if i == 0:
                legend_handles.append(bar[0])
                legend_labels.append(SPLIT_BAND_LABELS.get(fs, fs))

            if np.isfinite(m):
                axA.errorbar(xpos, m, yerr=[[m - lo], [hi - m]], fmt="none",
                             color="black", capsize=3, linewidth=1.0, zorder=4)

            if vals.size > 0:
                jit = jitter_positions(vals.size, width=0.035,
                                       seed=RANDOM_STATE + 500 + i * 20 + j)
                axA.scatter(np.full(vals.size, xpos) + jit, vals,
                            s=14, facecolors="white", edgecolors="0.28",
                            linewidth=0.7, alpha=0.72, zorder=5)

            brow      = split_band_df[(split_band_df["roi"] == roi) & (split_band_df["band"] == fs)]
            brow_dict = brow.iloc[0].to_dict() if len(brow) == 1 else None
            stars     = _fdr_stars(brow_dict)

            if stars and np.isfinite(hi):
                star_y = hi + 0.045 * yrangeA
                axA.text(xpos, star_y, stars, ha="center", va="bottom",
                         fontsize=10.5, clip_on=False)
                annotation_ymax = max(annotation_ymax, star_y + 0.02 * yrangeA)

            if np.isfinite(hi):
                roi_bar_hi = max(roi_bar_hi, hi)

        crow = split_contrast_df[
            (split_contrast_df["roi"] == roi) &
            (split_contrast_df["contrast"] == "PredCLIP_gt_ResCLIP")
        ]
        crow_dict = crow.iloc[0].to_dict() if len(crow) == 1 else None
        stars     = _fdr_stars(crow_dict)

        if stars and np.isfinite(roi_bar_hi):
            pred_x   = x[i] + offsets[bands.index("CaloriePredCLIP")]
            res_x    = x[i] + offsets[bands.index("CalorieResCLIP")]
            bracket_y = roi_bar_hi + 0.13 * yrangeA
            bracket_h = 0.035 * yrangeA
            _draw_bracket(axA, pred_x, res_x, bracket_y, bracket_h, stars)
            annotation_ymax = max(annotation_ymax, bracket_y + bracket_h + 0.04 * yrangeA)

    yhiA_final = max(yhiA, annotation_ymax + 0.02 * yrangeA)
    axA.axhline(0, color="black", linewidth=0.9, zorder=1)
    axA.set_xlim(-0.60, len(rois) - 0.40)
    axA.set_ylim(yloA, yhiA_final)
    axA.set_xticks(x)
    axA.set_xticklabels([ROI_LABELS.get(r, r) for r in rois], rotation=18, ha="right")
    axA.set_ylabel(r"M4 split contribution, $r_{split}$")
    axA.set_title("Absolute split contributions", pad=8)
    axA.grid(axis="y", color="0.90", linewidth=0.7, zorder=0)

    # Panel B: relative importance
    _panel_label(axB, "B")
    bottoms = np.zeros(len(rois), dtype=float)

    for fs in bands:
        vals = []
        for roi in rois:
            row = relative_df[(relative_df["roi"] == roi) & (relative_df["band"] == fs)]
            vals.append(float(row["mean_share"].iloc[0]) if len(row) == 1 else np.nan)
        vals = np.asarray(vals, dtype=float)

        axB.bar(x, vals, bottom=bottoms, width=0.62,
                color=SPLIT_BAND_COLORS[fs], edgecolor="white",
                linewidth=0.8, alpha=0.92, zorder=2)

        for i, v in enumerate(vals):
            if np.isfinite(v) and v >= 0.12:
                txt_color = "white" if v >= 0.18 else "black"
                axB.text(x[i], bottoms[i] + v / 2, f"{100 * v:.0f}%",
                         ha="center", va="center", fontsize=9,
                         color=txt_color, clip_on=True, zorder=4)

        bottoms = bottoms + np.nan_to_num(vals, nan=0.0)

    axB.set_xlim(-0.60, len(rois) - 0.40)
    axB.set_ylim(0, 1.0)
    axB.set_xticks(x)
    axB.set_xticklabels([ROI_LABELS.get(r, r) for r in rois], rotation=18, ha="right")
    axB.set_yticks(np.linspace(0, 1, 6))
    axB.set_yticklabels([f"{int(100 * t)}%" for t in np.linspace(0, 1, 6)])
    axB.set_ylabel("Share of positive split contribution")
    axB.set_title("Relative importance", pad=8)
    axB.grid(axis="y", color="0.90", linewidth=0.7, zorder=0)

    fig.legend(legend_handles, legend_labels,
               loc="upper center", bbox_to_anchor=(0.50, 0.895),
               ncol=len(bands), frameon=False,
               columnspacing=1.8, handlelength=1.6)

    fig.text(
        0.07, 0.075,
        "Panel A: bars show group mean ± 95% CI; dots are subjects; "
        "stars/brackets use BH-FDR q<.05. "
        "Panel B is descriptive: negative split contributions are clipped to 0 "
        "before subject-wise normalization.",
        ha="left", va="center", fontsize=8.5, color="0.35",
    )

    savefig(fig, "supp_fig2_hierarchy_split_contributions")
    plt.close(fig)

# =============================================================================
# [11b] CLIP-SEPARATED BASELINE DIAGNOSTIC
# =============================================================================

def plot_clipband_diagnostic_figure(
    diag_df,
    diag_rjoint_inf,
    diag_comp_df,
    diag_gain_df,
    diag_nc_roi=None,
):
    """
    Supplementary diagnostic figure.

    Panel A:
        D0 no CLIP, D1 CLIP band, D2 CLIP band + PredCLIP.
        Noise ceiling is drawn from diag_nc_roi.

    Panel B:
        D2 - D1 gain, testing whether PredCLIP adds beyond the CLIP-separated
        baseline. No noise ceiling is drawn here because this is a delta-r plot.
    """
    if diag_nc_roi is None:
        diag_nc_roi = {}

    rois = hierarchy_rois()
    x = np.arange(len(rois))

    d0 = "D0_visual_noclip"
    d1 = "D1_visual_clipband"
    d2 = "D2_visual_clipband_predclip"
    gain_comp = MAIN_DIAGNOSTIC_COMPARISON

    print("\nNC actually passed to diagnostic figure:")
    for roi in rois:
        entry = diag_nc_roi.get(roi)
        if entry is None:
            print(f"  {roi:22s}: None")
        else:
            print(
                f"  {roi:22s}: mean={entry['mean']:.4f}, "
                f"lo={entry['lo']:.4f}, hi={entry['hi']:.4f}, n={entry['n']}"
            )

    fig = plt.figure(figsize=(13.5, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    from matplotlib.patches import Patch
    nc_handle = Patch(
        facecolor=NC_COLOR,
        alpha=NC_ALPHA * 3,
        edgecolor=NC_COLOR,
        linewidth=0.5,
        label="Noise ceiling",
    )

    # -------------------------------------------------------------------------
    # Panel A — absolute prediction for D0/D1/D2
    # -------------------------------------------------------------------------
    panel_label(axA, "A")

    models = [d0, d1, d2]
    offsets = [-0.25, 0.0, 0.25]
    width = 0.22

    values_A = []
    for roi in rois:
        for mk in models:
            values_A.extend(get_model_values(diag_df, roi, mk))
        entry = diag_nc_roi.get(roi)
        if entry is not None:
            values_A.append(entry["hi"])

    yloA, yhiA = nice_ylim(values_A, include_zero=True, pad_fraction=0.26, min_span=0.06)

    for i, roi in enumerate(rois):
        entry = diag_nc_roi.get(roi)

        if entry is not None:
            # This catches the exact bug we were seeing.
            assert entry["mean"] > 0.10, (
                f"Suspicious NC for {roi}: {entry}. "
                "You are probably plotting the wrong NC object."
            )

        draw_nc_band(axA, i, entry, x_half_width=0.45)

        max_hi = -np.inf

        for mk, off in zip(models, offsets):
            vals = get_model_values(diag_df, roi, mk)
            m, lo, hi = mean_ci(vals)
            xpos = i + off

            axA.bar(
                xpos,
                m,
                width=width,
                color=DIAGNOSTIC_MODELS[mk]["color"],
                alpha=0.90,
                edgecolor="white",
                linewidth=0.8,
                label=DIAGNOSTIC_MODELS[mk]["label"] if i == 0 else None,
            )
            axA.errorbar(
                xpos,
                m,
                yerr=[[m - lo], [hi - m]],
                fmt="none",
                color="black",
                capsize=3,
                linewidth=1.1,
            )

            jit = jitter_positions(
                len(vals),
                width=0.038,
                seed=RANDOM_STATE + 700 + 10 * i + models.index(mk),
            )
            axA.scatter(
                np.full(len(vals), xpos) + jit,
                vals,
                s=POINT_SIZE - 4,
                facecolors="white",
                edgecolors="0.25",
                linewidth=0.7,
                alpha=POINT_ALPHA,
                zorder=3,
            )

            row = get_rjoint_row(diag_rjoint_inf, roi, mk)
            if should_mark(row, use_fdr_only=SUPPFIG_USE_FDR_ONLY):
                txt = mark_text(row, use_fdr_only=SUPPFIG_USE_FDR_ONLY)
                axA.text(
                    xpos,
                    hi + 0.035 * (yhiA - yloA),
                    txt,
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )

            max_hi = max(max_hi, hi)

        # D1 > D0 bracket
        row_d1 = get_comp_row(diag_comp_df, roi, "D1_clipband_gt_D0_noclip")
        if should_mark(row_d1, use_fdr_only=SUPPFIG_USE_FDR_ONLY):
            txt = mark_text(row_d1, use_fdr_only=SUPPFIG_USE_FDR_ONLY)
            y = max_hi + 0.08 * (yhiA - yloA)
            h = 0.020 * (yhiA - yloA)
            draw_bracket(axA, i + offsets[0], i + offsets[1], y, h, txt)

        # D2 > D1 bracket
        row_d2 = get_comp_row(diag_comp_df, roi, "D2_predclip_gt_D1_clipband")
        if should_mark(row_d2, use_fdr_only=SUPPFIG_USE_FDR_ONLY):
            txt = mark_text(row_d2, use_fdr_only=SUPPFIG_USE_FDR_ONLY)
            y = max_hi + 0.16 * (yhiA - yloA)
            h = 0.020 * (yhiA - yloA)
            draw_bracket(axA, i + offsets[1], i + offsets[2], y, h, txt)

    axA.axhline(0, color="black", linewidth=0.9)
    axA.set_xlim(-0.6, len(rois) - 0.4)
    axA.set_ylim(yloA, yhiA)
    axA.set_xticks(x)
    axA.set_xticklabels([roi_label(r) for r in rois], rotation=20, ha="right")
    axA.set_ylabel(r"Mean held-out $r_{joint}$")
    axA.set_title("CLIP-separated baseline diagnostic")
    axA.grid(axis="y", color="0.90", linewidth=0.8)

    handles, labels = axA.get_legend_handles_labels()
    if any(diag_nc_roi.get(r) is not None for r in rois):
        handles.append(nc_handle)
        labels.append("Noise ceiling")
    axA.legend(handles, labels, frameon=False, loc="upper left", fontsize=9)

    # -------------------------------------------------------------------------
    # Panel B — D2 - D1 gain
    # -------------------------------------------------------------------------
    panel_label(axB, "B")

    values_B = []
    for roi in rois:
        values_B.extend(get_gain_values(diag_gain_df, roi, gain_comp))

    yloB, yhiB = nice_ylim(values_B, include_zero=True, pad_fraction=0.30, min_span=0.025)

    for i, roi in enumerate(rois):
        vals = get_gain_values(diag_gain_df, roi, gain_comp)
        m, lo, hi = mean_ci(vals)

        axB.bar(
            i,
            m,
            width=0.55,
            color=DIAGNOSTIC_MODELS[d2]["color"],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
        )
        axB.errorbar(
            i,
            m,
            yerr=[[m - lo], [hi - m]],
            fmt="none",
            color="black",
            capsize=3,
            linewidth=1.1,
        )

        jit = jitter_positions(len(vals), width=0.09, seed=RANDOM_STATE + 900 + i)
        axB.scatter(
            np.full(len(vals), i) + jit,
            vals,
            s=POINT_SIZE,
            facecolors="white",
            edgecolors="0.25",
            linewidth=0.8,
            alpha=POINT_ALPHA,
            zorder=3,
        )

        row = get_comp_row(diag_comp_df, roi, gain_comp)
        if should_mark(row, use_fdr_only=SUPPFIG_USE_FDR_ONLY):
            txt = mark_text(row, use_fdr_only=SUPPFIG_USE_FDR_ONLY)
            axB.text(
                i,
                hi + 0.04 * (yhiB - yloB),
                txt,
                ha="center",
                va="bottom",
                fontsize=13,
            )

    axB.axhline(0, color="black", linewidth=0.9)
    axB.set_xlim(-0.6, len(rois) - 0.4)
    axB.set_ylim(yloB, yhiB)
    axB.set_xticks(x)
    axB.set_xticklabels([roi_label(r) for r in rois], rotation=20, ha="right")
    axB.set_ylabel(r"Diagnostic gain, $\Delta r_{joint}$")
    axB.set_title("PredCLIP beyond CLIP-separated baseline")
    axB.grid(axis="y", color="0.90", linewidth=0.8)

    # Force overwrite to prevent accidentally inspecting an old figure.
    for ext in ["png", "pdf"]:
        old = os.path.join(FIGDIR, f"supp_fig3_clipband_diagnostic.{ext}")
        if os.path.isfile(old):
            os.remove(old)
            print(f"  removed old diagnostic figure: {old}")

    savefig(fig, "supp_fig3_clipband_diagnostic")
    plt.close(fig)

def run_clipband_diagnostic(roi_masks, nc_df=None):
    """
    Add-on diagnostic only.

    This does not modify the primary M0-M4 analysis.
    It loads D0/D1/D2 from DIAGNOSTIC_MODELS and writes separate CSV outputs.

    Critical contrast:
        D2_visual_clipband_predclip > D1_visual_clipband

    Interpretation:
        Does CaloriePredCLIP still add beyond a baseline where CLIP has its
        own feature band and its own ridge regularisation?
    """
    print(f"\n{'#' * 80}")
    print("CLIP-SEPARATED BASELINE DIAGNOSTIC")
    print(f"{'#' * 80}")

    available_diag = resolve_available_models(
        model_registry=DIAGNOSTIC_MODELS,
        require_all=REQUIRE_ALL_DIAGNOSTIC_MODELS,
        registry_name="clipband diagnostic",
    )

    subjects_diag = common_subjects_across_models(available_diag)
    diag_nc_df = pd.DataFrame()
    diag_nc_roi = {}
    
    if nc_df is not None and not nc_df.empty:
        diag_nc_df, diag_nc_roi = summarize_nc_for_subject_subset(
            nc_df=nc_df,
            subjects_subset=subjects_diag,
        )
    
        diag_nc_df.to_csv(
            os.path.join(OUTDIR, "clipband_diagnostic_noise_ceiling_roi_values.csv"),
            index=False,
        )

    print("\nDiagnostic NC reused from main NC table:")
    for roi_name, entry in diag_nc_roi.items():
        if entry is None:
            print(f"  {roi_name:22s}: no NC")
        else:
            print(
                f"  {roi_name:22s}: mean={entry['mean']:.4f}, "
                f"SEM={entry['sem']:.4f}, n={entry['n']}"
            )

    pd.DataFrame({"subject": subjects_diag}).to_csv(
        os.path.join(OUTDIR, "clipband_diagnostic_subjects_included.csv"),
        index=False,
    )

    diag_model_table = pd.DataFrame([
        {
            "model": mk,
            "label": spec["label"],
            "short_label": spec["short_label"],
            "long_label": spec["long_label"],
            "dir": spec["dir"],
            "n_subjects_available": len(spec["subjects"]),
        }
        for mk, spec in available_diag.items()
    ])
    diag_model_table.to_csv(
        os.path.join(OUTDIR, "clipband_diagnostic_models_available.csv"),
        index=False,
    )

    r_joint_diag = load_all_r_joint(
        available_models=available_diag,
        subjects=subjects_diag,
        model_registry=DIAGNOSTIC_MODELS,
    )

    diag_df = extract_roi_model_values(
        subjects=subjects_diag,
        roi_masks=roi_masks,
        available_models=available_diag,
        r_joint=r_joint_diag,
    )
    diag_df.to_csv(
        os.path.join(OUTDIR, "clipband_diagnostic_roi_model_r_joint_subject_values.csv"),
        index=False,
    )

    diag_rjoint_inf = roi_model_r_joint_inference(diag_df)
    diag_rjoint_inf.to_csv(
        os.path.join(OUTDIR, "clipband_diagnostic_roi_model_r_joint_inference.csv"),
        index=False,
    )

    diag_comps = available_comparisons(
        available_models=available_diag,
        comparison_registry=DIAGNOSTIC_MODEL_COMPARISONS,
    )

    diag_comp_df = roi_model_comparisons(diag_df, diag_comps)
    diag_comp_df.to_csv(
        os.path.join(OUTDIR, "clipband_diagnostic_roi_model_comparisons.csv"),
        index=False,
    )

    diag_hdf = extract_hierarchy_model_values(diag_df)
    diag_hdf.to_csv(
        os.path.join(OUTDIR, "clipband_diagnostic_hierarchy_model_values.csv"),
        index=False,
    )

    diag_gain_df = compute_hierarchy_model_gain_values(diag_hdf, diag_comps)
    diag_gain_df.to_csv(
        os.path.join(OUTDIR, "clipband_diagnostic_hierarchy_model_gain_values.csv"),
        index=False,
    )

    diag_slope_df = hierarchy_gain_slopes(
        diag_gain_df,
        use_comparisons=[MAIN_DIAGNOSTIC_COMPARISON],
    )
    diag_slope_df.to_csv(
        os.path.join(OUTDIR, "clipband_diagnostic_hierarchy_model_gain_slopes.csv"),
        index=False,
    )

    diag_slope_tests = hierarchy_gain_slope_tests(diag_slope_df, diag_comps)
    diag_slope_tests.to_csv(
        os.path.join(OUTDIR, "clipband_diagnostic_hierarchy_model_gain_slope_tests.csv"),
        index=False,
    )

    diag_pairwise = hierarchy_gain_pairwise_tests(
        diag_gain_df[diag_gain_df["comparison"] == MAIN_DIAGNOSTIC_COMPARISON],
        diag_comps,
    )
    diag_pairwise.to_csv(
        os.path.join(OUTDIR, "clipband_diagnostic_hierarchy_model_gain_pairwise.csv"),
        index=False,
    )

    # Compact summary: model performance per ROI.
    summary_perf = (
        diag_rjoint_inf[
            [
                "roi", "roi_label", "model", "model_label", "n_subjects",
                "mean_r_joint", "sem_r_joint", "ci95_lo_r_joint",
                "ci95_hi_r_joint", "cohens_dz_z", "p", "q_fdr_bh",
                "sig_fdr_05",
            ]
        ]
        .sort_values(["roi", "model"])
        .copy()
    )
    summary_perf.to_csv(
        os.path.join(OUTDIR, "clipband_diagnostic_summary_model_performance.csv"),
        index=False,
    )

    # Compact summary: key diagnostic comparisons per ROI.
    summary_comps = (
        diag_comp_df[
            [
                "roi", "roi_label", "comparison", "comparison_label",
                "model_A", "model_B", "n_subjects",
                "mean_r_A", "mean_r_B", "mean_delta_r",
                "sem_delta_r", "ci95_lo_delta_r", "ci95_hi_delta_r",
                "cohens_dz", "p", "q_fdr_bh", "sig_fdr_05",
            ]
        ]
        .sort_values(["comparison", "roi"])
        .copy()
    )
    summary_comps.to_csv(
        os.path.join(OUTDIR, "clipband_diagnostic_summary_key_comparisons.csv"),
        index=False,
    )

    print("\nKey diagnostic comparison:")
    key = diag_comp_df[
        diag_comp_df["comparison"] == MAIN_DIAGNOSTIC_COMPARISON
    ].copy()
    
    if SAVE_SUPP_FIG:
        plot_clipband_diagnostic_figure(
            diag_df=diag_df,
            diag_rjoint_inf=diag_rjoint_inf,
            diag_comp_df=diag_comp_df,
            diag_gain_df=diag_gain_df,
            diag_nc_roi=diag_nc_roi,
        )

    if key.empty:
        print(f"  WARNING: {MAIN_DIAGNOSTIC_COMPARISON} not found.")
    else:
        cols = [
            "roi", "mean_r_A", "mean_r_B", "mean_delta_r",
            "ci95_lo_delta_r", "ci95_hi_delta_r", "p", "q_fdr_bh",
            "sig_fdr_05",
        ]
        print(key[cols].round(5).to_string(index=False))

    print(
        "\nSaved CLIP-band diagnostic outputs with prefix:\n"
        f"  {os.path.join(OUTDIR, 'clipband_diagnostic_')}"
    )

# =============================================================================
# [12] MAIN
# =============================================================================

def main():
    set_plot_defaults()

    _, roi_masks = load_hcp_parcels(HCP_DLABEL)

    available_models = resolve_available_models()
    subjects         = common_subjects_across_models(available_models)

    pd.DataFrame({"subject": subjects}).to_csv(
        os.path.join(OUTDIR, "subjects_included.csv"), index=False)

    model_table = pd.DataFrame([
        {
            "model":                 mk,
            "label":                 spec["label"],
            "short_label":           spec["short_label"],
            "long_label":            spec["long_label"],
            "dir":                   spec["dir"],
            "n_subjects_available":  len(spec["subjects"]),
        }
        for mk, spec in available_models.items()
    ])
    model_table.to_csv(os.path.join(OUTDIR, "models_available.csv"), index=False)

    # ── Noise ceiling ─────────────────────────────────────────────────────────
    nc_df, nc_roi = load_noise_ceiling(subjects, roi_masks)
    if not nc_df.empty:
        nc_df.to_csv(os.path.join(OUTDIR, "noise_ceiling_roi_values.csv"), index=False)

    # ── Model r_joint ─────────────────────────────────────────────────────────
    r_joint = load_all_r_joint(available_models, subjects)

    model_df = extract_roi_model_values(
        subjects=subjects, roi_masks=roi_masks,
        available_models=available_models, r_joint=r_joint,
    )
    model_df.to_csv(
        os.path.join(OUTDIR, "roi_model_r_joint_subject_values.csv"), index=False)

    rjoint_inf = roi_model_r_joint_inference(model_df)
    rjoint_inf.to_csv(
        os.path.join(OUTDIR, "roi_model_r_joint_inference.csv"), index=False)

    comps   = available_comparisons(available_models)
    comp_df = roi_model_comparisons(model_df, comps)
    comp_df.to_csv(
        os.path.join(OUTDIR, "roi_model_comparisons.csv"), index=False)

    hdf = extract_hierarchy_model_values(model_df)
    hdf.to_csv(os.path.join(OUTDIR, "hierarchy_model_values.csv"), index=False)

    gain_df = compute_hierarchy_model_gain_values(hdf, comps)
    gain_df.to_csv(
        os.path.join(OUTDIR, "hierarchy_model_gain_values.csv"), index=False)

    use_hierarchy_comps = [MAIN_GAIN_COMPARISON, M1_GAIN_COMPARISON]

    slope_df = hierarchy_gain_slopes(gain_df, use_comparisons=use_hierarchy_comps)
    slope_df.to_csv(
        os.path.join(OUTDIR, "hierarchy_model_gain_slopes.csv"), index=False)

    slope_tests = hierarchy_gain_slope_tests(slope_df, comps)
    slope_tests.to_csv(
        os.path.join(OUTDIR, "hierarchy_model_gain_slope_tests.csv"), index=False)

    pairwise_df = hierarchy_gain_pairwise_tests(
        gain_df[gain_df["comparison"].isin(use_hierarchy_comps)], comps)
    pairwise_df.to_csv(
        os.path.join(OUTDIR, "hierarchy_model_gain_pairwise.csv"), index=False)

    print("\nSaving main figure")
    plot_main_fig_core_story(
        model_df    = model_df,
        rjoint_inf  = rjoint_inf,
        comp_df     = comp_df,
        gain_df     = gain_df,
        pairwise_df = pairwise_df,
        nc_roi      = nc_roi,
    )

    if split_model_available(available_models, subjects):
        split_df = extract_split_roi_values(subjects, roi_masks, r_joint)
        split_df.to_csv(
            os.path.join(OUTDIR, "split_model_roi_subject_values.csv"), index=False)

        split_band_df = split_band_inference(split_df)
        split_band_df.to_csv(
            os.path.join(OUTDIR, "split_model_band_inference.csv"), index=False)

        split_contrast_df = split_contrast_inference(split_df)
        split_contrast_df.to_csv(
            os.path.join(OUTDIR, "split_model_contrast_inference.csv"), index=False)

        relative_df = compute_split_relative_importance(
            split_df=split_df, rois=HIERARCHY_SPLIT_ROIS)
        relative_df.to_csv(
            os.path.join(OUTDIR, "split_model_hierarchy_relative_importance.csv"),
            index=False)

        if SAVE_SUPP_FIG:
            print("\nSaving supplementary figures")
            plot_supp_fig_predclip_vs_resclip_split(
                split_df=split_df, split_band_df=split_band_df,
                split_contrast_df=split_contrast_df)
            plot_supp_fig_hierarchy_split_contributions(
                split_df=split_df, split_band_df=split_band_df,
                split_contrast_df=split_contrast_df, relative_df=relative_df)
    if RUN_CLIPBAND_DIAGNOSTIC:
        run_clipband_diagnostic(roi_masks, nc_df=nc_df)

    write_traceability_metadata(subjects=subjects, nc_df=nc_df, available_models=available_models)

    print(f"\nDone. Outputs saved in:\n{OUTDIR}")


if __name__ == "__main__":
    main()
