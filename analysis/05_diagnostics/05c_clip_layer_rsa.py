# -*- coding: utf-8 -*-
r"""
NeuroRepFood — OpenAI CLIP layer-wise RSA with processing/preparation RDMs
=========================================================================

Purpose
-------
This script characterizes whether CLIP vision-layer representations organize
the 96 NeuroRepFood food images according to processing / preparation state.

It does NOT use fMRI data. It is a stimulus/model-characterization analysis.

Workflow
--------
1. Load 96 food stimuli in ordered_stimuli.csv order.
2. Extract OpenAI CLIP ViT-B/32 visual features from:
       - patch/CLS embedding stage
       - transformer blocks 1–12
       - final projected CLIP image embedding
3. Compute CLIP layer RDMs.
4. Construct two processing/preparation model RDMs:
       - ordinal 4-level RDM: 0, 1, 2, 3
       - binary RDM: unprocessed/minimal vs processed/prepared
5. Correlate each CLIP RDM with each model RDM.
6. Use label-shuffle permutation tests.
7. Correct across layers using:
       - Benjamini-Hochberg FDR
       - max-statistic familywise correction
8. Save Kriegeskorte-style figures.

Important conceptual choice
---------------------------
The labels are NOT intended as strict NOVA nutritional categories.
They reflect the visual processing/preparation state of the depicted food image:

    Level 0 = raw / natural produce
    Level 1 = minimally processed or visually natural single-ingredient food
    Level 2 = processed or prepared simple food
    Level 3 = highly processed / composite / confectionery / fried / snack-like food

Binary split:
    Levels 0–1 = unprocessed / minimally processed
    Levels 2–3 = processed / prepared

This gives a balanced binary split:
    48 unprocessed/minimally processed
    48 processed/prepared

Dependencies
------------
pip install numpy pandas scipy scikit-learn matplotlib pillow torch torchvision
pip install git+https://github.com/openai/CLIP.git

Author
------
Adapted for NeuroRepFood.
"""

# =============================================================================
# [0] IMPORTS
# =============================================================================

import os
import json
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image

import torch

try:
    import clip as openai_clip
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "The OpenAI CLIP package is not installed.\n\n"
        "Install it with:\n"
        "    pip install git+https://github.com/openai/CLIP.git\n"
    ) from e

from scipy.stats import rankdata, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore", category=RuntimeWarning)

print("Imports OK.")


# =============================================================================
# [1] PATHS AND CONFIGURATION
# =============================================================================

# RELEASE PATH ADAPTATION ONLY.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

STIMDIR = os.path.join(REPO_ROOT, "data", "stimuli")
IMGDIR = os.path.join(STIMDIR, "images")
STIM_CSV = os.path.join(STIMDIR, "ordered_stimuli.csv")

OUTDIR = os.path.join(
    REPO_ROOT,
    "reproduced_outputs",
    "diagnostics",
    "clip_openai_layerwise_processing_rsa",
)

FEATURE_OUTDIR = os.path.join(OUTDIR, "features")
RDM_OUTDIR = os.path.join(OUTDIR, "rdms")
FIGDIR = os.path.join(OUTDIR, "figures")
PERM_OUTDIR = os.path.join(OUTDIR, "permutation_nulls")

for d in [OUTDIR, FEATURE_OUTDIR, RDM_OUTDIR, FIGDIR, PERM_OUTDIR]:
    os.makedirs(d, exist_ok=True)

# OpenAI CLIP model
MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

# Token modes:
#   cls        = global CLS token
#   mean_patch = average over image patch tokens
TOKEN_MODES = ["cls", "mean_patch"]

# For CLS, layer_00_embedding is omitted from RSA because the CLS token has not
# yet attended to image patches and is effectively image-independent.
OMIT_CLS_EMBEDDING_FROM_RSA = True

# RDM metric for CLIP features
FEATURE_RDM_METRIC = "correlation"  # "correlation" or "cosine"

# Run both models. Ordinal is the recommended main characterization.
MODEL_RDM_TYPES_TO_RUN = ["ordinal", "binary"]

# Recommended RSA method by model:
#   ordinal: Spearman captures monotonic relation to 0–3 processing distance.
#   binary : Pearson is an interpretable same-class vs different-class contrast.
RSA_METHOD_BY_MODEL = {
    "ordinal": "spearman",
    "binary": "pearson",
}

N_PERMUTATIONS = 10000
RANDOM_STATE = 42

DPI = 300
SAVE_PNG = True
SAVE_PDF = True

# Figures to generate
MAKE_FIGURES_FOR_TOKEN_MODES = ["cls", "mean_patch"]


# =============================================================================
# [2] CANONICAL STIMULUS ORDER
# =============================================================================

EXPECTED_CONDS = [
    "Almonds",
    "Apple",
    "Apricot",
    "Asparagus_green",
    "Aubergine",
    "Avocado",
    "Bacon",
    "Baguette",
    "Banana",
    "Beetroot",
    "Berliner",
    "Bitterballs",
    "Blackberries",
    "Blue_cheese",
    "Bonbon",
    "Brie",
    "Broccoli",
    "Brownie",
    "Brussels_sprout",
    "Butter_biscuit",
    "Cake",
    "Candies",
    "Carrot",
    "Cauliflower",
    "Cheesecake",
    "Cheese_souffle",
    "Cherries",
    "Cherry_tomatoes",
    "Chickpeas",
    "Chocolate_bar",
    "Chocolate_cake",
    "Choco_cone",
    "Choco_cookies",
    "Choco_lava_cake",
    "Coconut",
    "Codfish",
    "Corn",
    "Cream_cake",
    "Crisps",
    "Croissant",
    "Cucumber",
    "Currant",
    "Donut",
    "Figs",
    "Fried_egg",
    "Fries",
    "Grape_red",
    "Grape_white",
    "Ham",
    "Hamburger",
    "Herring",
    "Honey_melon",
    "Hotdog",
    "Kiwi",
    "Lasagna",
    "Lettuce",
    "Magnum",
    "Mandarin",
    "Mango",
    "Mars",
    "Melon",
    "MMs",
    "Muesli_bar",
    "Mushrooms",
    "Mussles",
    "Nutella_sandwich",
    "Olives",
    "Orange",
    "Oyster",
    "Papaya",
    "Pasta_pomodoro",
    "Pear",
    "Pees",
    "Pepper_red",
    "Pineapple",
    "Pizza",
    "Pomegranate",
    "Popcorn",
    "Pound_cake",
    "Radicchio",
    "Radish",
    "Raspberries",
    "Ribs",
    "Rice_waffle",
    "Roasted_potatoes",
    "Salami",
    "Smoked_sausage",
    "Strawberries",
    "Sushi",
    "Tiramisu",
    "Tompoes",
    "Tuna",
    "Vanilla_icecream",
    "Waffle",
    "Watermelon",
    "Zucchini",
]


def load_stimulus_order(stim_csv):
    """Load canonical stimulus order from ordered_stimuli.csv."""
    if not os.path.isfile(stim_csv):
        raise FileNotFoundError(stim_csv)

    stim_df = pd.read_csv(stim_csv)

    if "Value" not in stim_df.columns:
        raise ValueError("ordered_stimuli.csv must contain a column named 'Value'.")

    conds = stim_df["Value"].astype(str).str.strip().tolist()

    if len(conds) != 96:
        raise ValueError(f"Expected 96 stimuli, found {len(conds)}.")

    order_check = pd.DataFrame({
        "index": np.arange(96),
        "condition_from_csv": conds,
        "expected_condition": EXPECTED_CONDS,
        "match_expected": [a == b for a, b in zip(conds, EXPECTED_CONDS)],
    })

    order_check_path = os.path.join(OUTDIR, "stimulus_order_check.csv")
    order_check.to_csv(order_check_path, index=False)

    if order_check["match_expected"].all():
        print("Stimulus order matches EXPECTED_CONDS.")
    else:
        print("WARNING: ordered_stimuli.csv differs from EXPECTED_CONDS.")
        print("Using ordered_stimuli.csv as canonical order.")
        print(f"Check file: {order_check_path}")

    return conds


def get_img_path(cond):
    """Find image file for a condition name."""
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = os.path.join(IMGDIR, cond + ext)
        if os.path.isfile(p):
            return p

    # Case-insensitive fallback
    if os.path.isdir(IMGDIR):
        target = cond.lower()
        for fname in os.listdir(IMGDIR):
            stem, ext = os.path.splitext(fname)
            if stem.lower() == target and ext.lower() in [".jpg", ".jpeg", ".png"]:
                return os.path.join(IMGDIR, fname)

    return None


CONDS = load_stimulus_order(STIM_CSV)
IMG_PATHS = [get_img_path(c) for c in CONDS]

missing = [c for c, p in zip(CONDS, IMG_PATHS) if p is None]
if missing:
    raise FileNotFoundError(
        f"{len(missing)} images were not found in IMGDIR:\n"
        + "\n".join(missing[:25])
    )

print(f"Loaded {len(CONDS)} stimuli.")
print(f"IMGDIR: {IMGDIR}")


# =============================================================================
# [3] PROCESSING / PREPARATION LABELS
# =============================================================================

# -----------------------------------------------------------------------------
# Level 0: raw / natural produce
# -----------------------------------------------------------------------------
LEVEL_0_RAW_NATURAL_PRODUCE = {
    "Apple",
    "Apricot",
    "Asparagus_green",
    "Aubergine",
    "Avocado",
    "Banana",
    "Beetroot",
    "Blackberries",
    "Broccoli",
    "Brussels_sprout",
    "Carrot",
    "Cauliflower",
    "Cherries",
    "Cherry_tomatoes",
    "Coconut",
    "Corn",
    "Cucumber",
    "Currant",
    "Figs",
    "Grape_red",
    "Grape_white",
    "Honey_melon",
    "Kiwi",
    "Lettuce",
    "Mandarin",
    "Mango",
    "Melon",
    "Mushrooms",
    "Orange",
    "Papaya",
    "Pear",
    "Pees",
    "Pepper_red",
    "Pineapple",
    "Pomegranate",
    "Radicchio",
    "Radish",
    "Raspberries",
    "Strawberries",
    "Watermelon",
    "Zucchini",
}

# -----------------------------------------------------------------------------
# Level 1: minimally processed or visually natural single-ingredient foods
# -----------------------------------------------------------------------------
LEVEL_1_MINIMAL_SINGLE_INGREDIENT = {
    "Almonds",
    "Chickpeas",
    "Codfish",
    "Herring",
    "Mussles",
    "Oyster",
    "Tuna",
}

# -----------------------------------------------------------------------------
# Level 2: processed or prepared simple foods
# These are transformed/prepared, but not strongly composite/confectionery/fried.
# -----------------------------------------------------------------------------
LEVEL_2_PREPARED_SIMPLE = {
    "Baguette",
    "Blue_cheese",
    "Brie",
    "Fried_egg",
    "Olives",
    "Pasta_pomodoro",
    "Popcorn",
    "Rice_waffle",
    "Ribs",
    "Roasted_potatoes",
    "Sushi",
}

# -----------------------------------------------------------------------------
# Level 3: highly processed / composite / confectionery / fried / snack-like
# -----------------------------------------------------------------------------
LEVEL_3_HIGHLY_PROCESSED_COMPOSITE = {
    "Bacon",
    "Berliner",
    "Bitterballs",
    "Bonbon",
    "Brownie",
    "Butter_biscuit",
    "Cake",
    "Candies",
    "Cheesecake",
    "Cheese_souffle",
    "Chocolate_bar",
    "Chocolate_cake",
    "Choco_cone",
    "Choco_cookies",
    "Choco_lava_cake",
    "Cream_cake",
    "Crisps",
    "Croissant",
    "Donut",
    "Fries",
    "Ham",
    "Hamburger",
    "Hotdog",
    "Lasagna",
    "Magnum",
    "Mars",
    "MMs",
    "Muesli_bar",
    "Nutella_sandwich",
    "Pizza",
    "Pound_cake",
    "Salami",
    "Smoked_sausage",
    "Tiramisu",
    "Tompoes",
    "Vanilla_icecream",
    "Waffle",
}


def build_processing_level_map():
    """Return dict condition -> processing level 0..3."""
    level_map = {}

    for c in LEVEL_0_RAW_NATURAL_PRODUCE:
        level_map[c] = 0

    for c in LEVEL_1_MINIMAL_SINGLE_INGREDIENT:
        level_map[c] = 1

    for c in LEVEL_2_PREPARED_SIMPLE:
        level_map[c] = 2

    for c in LEVEL_3_HIGHLY_PROCESSED_COMPOSITE:
        level_map[c] = 3

    return level_map


PROCESSING_LEVEL_MAP = build_processing_level_map()

# Validate label coverage
all_labelled = set(PROCESSING_LEVEL_MAP.keys())
all_conds = set(CONDS)

missing_labels = sorted(all_conds - all_labelled)
extra_labels = sorted(all_labelled - all_conds)

if missing_labels:
    raise ValueError(
        "Some stimuli in ordered_stimuli.csv do not have processing labels:\n"
        + "\n".join(missing_labels)
    )

if extra_labels:
    raise ValueError(
        "Some processing labels are not present in ordered_stimuli.csv:\n"
        + "\n".join(extra_labels)
    )

processing_level = np.array([PROCESSING_LEVEL_MAP[c] for c in CONDS], dtype=int)

# Binary split:
#   0 = levels 0–1 = unprocessed / minimally processed
#   1 = levels 2–3 = processed / prepared
processed_binary = (processing_level >= 2).astype(int)

label_df = pd.DataFrame({
    "index": np.arange(len(CONDS)),
    "condition": CONDS,
    "processing_level_0_3": processing_level,
    "processed_binary": processed_binary,
    "binary_label": np.where(
        processed_binary == 1,
        "processed/prepared",
        "unprocessed/minimal",
    ),
})

label_csv = os.path.join(OUTDIR, "processing_preparation_labels.csv")
label_df.to_csv(label_csv, index=False)

print("\nProcessing/preparation levels:")
print(label_df["processing_level_0_3"].value_counts().sort_index())

print("\nBinary processing split:")
print(label_df["binary_label"].value_counts())

if not (np.sum(processed_binary == 0) == 48 and np.sum(processed_binary == 1) == 48):
    print("WARNING: binary split is not 48/48. Check labels.")
else:
    print("Binary split is balanced: 48 unprocessed/minimal, 48 processed/prepared.")


# =============================================================================
# [4] RDM AND RSA HELPERS
# =============================================================================

def upper_tri_vec(D):
    """Return upper-triangle vector of a square matrix."""
    iu = np.triu_indices_from(D, k=1)
    return D[iu]


def feature_rdm(X, metric="correlation"):
    """
    Compute feature RDM.

    correlation:
        1 - Pearson correlation between feature vectors.

    cosine:
        1 - cosine similarity between feature vectors.
    """
    X = np.asarray(X, dtype=np.float64)

    # Standardize columns across images.
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    if metric == "correlation":
        C = np.corrcoef(X)
        D = 1.0 - C
    elif metric == "cosine":
        S = cosine_similarity(X)
        D = 1.0 - S
    else:
        raise ValueError("metric must be 'correlation' or 'cosine'.")

    np.fill_diagonal(D, 0.0)

    # Keep NaNs if something is truly undefined, but print a warning.
    n_nan = int(np.isnan(D).sum())
    if n_nan > 0:
        print(f"  WARNING: feature RDM contains {n_nan} NaNs.")

    return D.astype(np.float32), upper_tri_vec(D).astype(np.float32)


def model_rdm_from_processing(processing_level, processed_binary, rdm_type):
    """
    Build processing/preparation model RDM.

    ordinal:
        D_ij = |level_i - level_j| / 3
        Uses four-level visual preparation scale.

    binary:
        D_ij = 0 if same binary class, 1 if different binary class.
    """
    if rdm_type == "ordinal":
        y = np.asarray(processing_level, dtype=float)
        D = np.abs(y[:, None] - y[None, :]).astype(np.float32)
        D = D / np.nanmax(D)

    elif rdm_type == "binary":
        y = np.asarray(processed_binary, dtype=int)
        D = (y[:, None] != y[None, :]).astype(np.float32)

    else:
        raise ValueError("rdm_type must be 'ordinal' or 'binary'.")

    np.fill_diagonal(D, 0.0)
    return D, upper_tri_vec(D)


def zscore_vec(v):
    """Z-score a vector safely."""
    v = np.asarray(v, dtype=np.float64)
    m = np.nanmean(v)
    s = np.nanstd(v)
    if s < 1e-12 or not np.isfinite(s):
        return np.full_like(v, np.nan, dtype=np.float64)
    return (v - m) / s


def prepare_rsa_vectors(feature_vecs, method):
    """
    Prepare layer RDM vectors for fast RSA.

    Parameters
    ----------
    feature_vecs : ndarray, shape (n_layers, n_pairs)
    method : 'pearson' or 'spearman'

    Returns
    -------
    prepared : ndarray, shape (n_layers, n_pairs)
        Row-wise z-scored vectors. For Spearman, vectors are rank-transformed first.
    valid_layer_mask : ndarray, shape (n_layers,)
        True for layers with valid finite variance.
    """
    V = np.asarray(feature_vecs, dtype=np.float64)

    if method == "spearman":
        V = np.apply_along_axis(rankdata, 1, V)
    elif method == "pearson":
        pass
    else:
        raise ValueError("method must be 'pearson' or 'spearman'.")

    Vz = np.zeros_like(V, dtype=np.float64)
    valid = np.ones(V.shape[0], dtype=bool)

    for i in range(V.shape[0]):
        ok = np.isfinite(V[i])
        if ok.sum() < 3 or np.nanstd(V[i, ok]) < 1e-12:
            valid[i] = False
            Vz[i, :] = np.nan
        else:
            Vz[i, ok] = zscore_vec(V[i, ok])
            Vz[i, ~ok] = np.nan

    return Vz, valid


def prepare_model_vec(model_vec, method):
    """Prepare one model RDM vector for fast RSA."""
    v = np.asarray(model_vec, dtype=np.float64)

    if method == "spearman":
        v = rankdata(v)
    elif method == "pearson":
        pass
    else:
        raise ValueError("method must be 'pearson' or 'spearman'.")

    return zscore_vec(v)


def fast_rsa_all_layers(prepared_feature_vecs, prepared_model_vec):
    """
    Compute RSA correlations for all layers.

    Assumes both feature and model vectors are already z-scored.
    Uses pairwise finite masks for safety.
    """
    V = np.asarray(prepared_feature_vecs, dtype=np.float64)
    m = np.asarray(prepared_model_vec, dtype=np.float64)

    corrs = np.full(V.shape[0], np.nan, dtype=np.float64)

    for i in range(V.shape[0]):
        ok = np.isfinite(V[i]) & np.isfinite(m)
        if ok.sum() < 3:
            continue
        corrs[i] = np.mean(V[i, ok] * m[ok])

    return corrs


def scipy_rsa_single(feature_vec, model_vec, method):
    """Single-layer RSA correlation with scipy, used for parametric p-values."""
    a = np.asarray(feature_vec, dtype=np.float64)
    b = np.asarray(model_vec, dtype=np.float64)

    ok = np.isfinite(a) & np.isfinite(b)

    if ok.sum() < 3:
        return np.nan, np.nan

    if method == "spearman":
        r, p = spearmanr(a[ok], b[ok])
    elif method == "pearson":
        r, p = pearsonr(a[ok], b[ok])
    else:
        raise ValueError("method must be 'pearson' or 'spearman'.")

    return float(r), float(p)


def bh_fdr(pvals):
    """
    Benjamini-Hochberg FDR correction.

    Returns q-values in the original order.
    """
    pvals = np.asarray(pvals, dtype=np.float64)
    qvals = np.full_like(pvals, np.nan, dtype=np.float64)

    finite = np.isfinite(pvals)
    if finite.sum() == 0:
        return qvals

    p = pvals[finite]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)

    q_ranked = ranked * m / (np.arange(m) + 1)

    # Enforce monotonicity from largest to smallest p.
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)

    q = np.empty_like(q_ranked)
    q[order] = q_ranked

    qvals[finite] = q
    return qvals


def permutation_test_across_layers(
    feature_vecs,
    layer_names,
    processing_level,
    processed_binary,
    rdm_type,
    method,
    n_perm=10000,
    seed=42,
):
    """
    Label-shuffle permutation test across all layers.

    Returns
    -------
    obs : ndarray, shape (n_layers,)
        Observed RSA correlations.

    null_by_layer : ndarray, shape (n_perm, n_layers)
        Layer-wise null correlations.

    max_null : ndarray, shape (n_perm,)
        Max null correlation across layers for each permutation.
        Used for familywise positive max-stat correction.
    """
    rng = np.random.default_rng(seed)

    feature_vecs = np.asarray(feature_vecs, dtype=np.float64)

    prepared_features, valid_layers = prepare_rsa_vectors(feature_vecs, method=method)

    model_D, model_vec = model_rdm_from_processing(
        processing_level=processing_level,
        processed_binary=processed_binary,
        rdm_type=rdm_type,
    )

    prepared_model = prepare_model_vec(model_vec, method=method)
    obs = fast_rsa_all_layers(prepared_features, prepared_model)

    null_by_layer = np.full((n_perm, len(layer_names)), np.nan, dtype=np.float32)
    max_null = np.full(n_perm, np.nan, dtype=np.float32)

    for pi in range(n_perm):
        perm = rng.permutation(len(processing_level))

        perm_level = processing_level[perm]
        perm_binary = processed_binary[perm]

        _, perm_model_vec = model_rdm_from_processing(
            processing_level=perm_level,
            processed_binary=perm_binary,
            rdm_type=rdm_type,
        )

        prepared_perm_model = prepare_model_vec(perm_model_vec, method=method)
        r_perm = fast_rsa_all_layers(prepared_features, prepared_perm_model)

        null_by_layer[pi, :] = r_perm.astype(np.float32)
        max_null[pi] = np.nanmax(r_perm).astype(np.float32)

        if (pi + 1) % 1000 == 0:
            print(f"    permutation {pi + 1:5d}/{n_perm}")

    p_unc = np.full(len(layer_names), np.nan, dtype=np.float64)
    p_max = np.full(len(layer_names), np.nan, dtype=np.float64)

    for li in range(len(layer_names)):
        if not np.isfinite(obs[li]):
            continue

        p_unc[li] = (np.sum(null_by_layer[:, li] >= obs[li]) + 1) / (n_perm + 1)
        p_max[li] = (np.sum(max_null >= obs[li]) + 1) / (n_perm + 1)

    q_fdr = bh_fdr(p_unc)

    return {
        "obs": obs,
        "p_unc": p_unc,
        "q_fdr": q_fdr,
        "p_max": p_max,
        "null_by_layer": null_by_layer,
        "max_null": max_null,
        "valid_layers": valid_layers,
        "model_D": model_D,
        "model_vec": model_vec,
    }


# =============================================================================
# [5] OPENAI CLIP LAYER-WISE FEATURE EXTRACTION
# =============================================================================

def load_clip_model():
    print(f"\nLoading OpenAI CLIP model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    model, preprocess = openai_clip.load(MODEL_NAME, device=DEVICE, jit=False)
    model.eval()

    return model, preprocess


def load_image_tensor(path, preprocess):
    img = Image.open(path).convert("RGB")
    return preprocess(img)


def encode_clip_vision_layers(model, image_batch):
    """
    Manual forward pass through OpenAI CLIP ViT visual encoder.

    Returns
    -------
    features : OrderedDict
        features[layer_name]["cls"]        = (B, D)
        features[layer_name]["mean_patch"] = (B, D)

    Layer definitions
    -----------------
    layer_00_embedding:
        patch embeddings + CLS token + positional embeddings + ln_pre

    layer_01_transformer ... layer_12_transformer:
        hidden states after each visual transformer block

    layer_13_projected_embedding:
        final image embedding after ln_post and projection
    """
    visual = model.visual

    x = image_batch.type(model.dtype)

    # Patch embedding: [B, 3, 224, 224] -> [B, width, grid, grid]
    x = visual.conv1(x)

    # Flatten patches: [B, width, grid, grid] -> [B, grid**2, width]
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)

    # Add learned CLS token.
    cls_token = visual.class_embedding.to(x.dtype)
    cls_token = cls_token + torch.zeros(
        x.shape[0],
        1,
        x.shape[-1],
        dtype=x.dtype,
        device=x.device,
    )
    x = torch.cat([cls_token, x], dim=1)

    # Add positional embedding and pre-transformer layer norm.
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    features = OrderedDict()

    def store_features(layer_name, xbld):
        # xbld shape: [B, tokens, width]
        cls = xbld[:, 0, :].detach().float().cpu().numpy()
        mean_patch = xbld[:, 1:, :].mean(dim=1).detach().float().cpu().numpy()

        features[layer_name] = {
            "cls": cls,
            "mean_patch": mean_patch,
        }

    # Embedding stage.
    store_features("layer_00_embedding", x)

    # Transformer expects [tokens, B, width].
    x = x.permute(1, 0, 2)

    for i, block in enumerate(visual.transformer.resblocks, start=1):
        x = block(x)
        xbld = x.permute(1, 0, 2)
        store_features(f"layer_{i:02d}_transformer", xbld)

    # Final projected CLIP image embedding.
    x_bld = x.permute(1, 0, 2)
    x_cls = visual.ln_post(x_bld[:, 0, :])

    if visual.proj is not None:
        x_proj = x_cls @ visual.proj
    else:
        x_proj = x_cls

    x_proj = x_proj.detach().float().cpu().numpy()

    features["layer_13_projected_embedding"] = {
        "cls": x_proj,
        "mean_patch": x_proj,
    }

    return features


def extract_all_clip_features(img_paths):
    model, preprocess = load_clip_model()

    all_features = {
        "cls": OrderedDict(),
        "mean_patch": OrderedDict(),
    }

    with torch.no_grad():
        for start in range(0, len(img_paths), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(img_paths))

            batch = torch.stack([
                load_image_tensor(p, preprocess) for p in img_paths[start:end]
            ]).to(DEVICE)

            batch_features = encode_clip_vision_layers(model, batch)

            for layer_name, feat_dict in batch_features.items():
                for token_mode in TOKEN_MODES:
                    if layer_name not in all_features[token_mode]:
                        all_features[token_mode][layer_name] = []
                    all_features[token_mode][layer_name].append(feat_dict[token_mode])

            print(f"  processed images {start + 1:03d}-{end:03d}/{len(img_paths)}")

    for token_mode in TOKEN_MODES:
        for layer_name in list(all_features[token_mode].keys()):
            X = np.vstack(all_features[token_mode][layer_name]).astype(np.float32)
            all_features[token_mode][layer_name] = X

            out_path = os.path.join(
                FEATURE_OUTDIR,
                f"OpenAI_CLIP_{MODEL_NAME.replace('/', '-')}_{token_mode}_{layer_name}.npy",
            )
            np.save(out_path, X)

    print("\nSaved layer-wise CLIP features.")
    return all_features


features = extract_all_clip_features(IMG_PATHS)


# =============================================================================
# [6] COMPUTE CLIP LAYER RDMs
# =============================================================================

clip_rdm_info = {}

for token_mode in TOKEN_MODES:
    print(f"\nComputing CLIP RDMs for token mode: {token_mode}")

    clip_rdm_info[token_mode] = OrderedDict()

    for layer_name, X in features[token_mode].items():

        # Omit undefined CLS embedding stage from RSA and figures.
        if (
            token_mode == "cls"
            and layer_name == "layer_00_embedding"
            and OMIT_CLS_EMBEDDING_FROM_RSA
        ):
            print(f"  Skipping {token_mode} {layer_name}: CLS token is not image-specific before attention.")
            continue

        D, v = feature_rdm(X, metric=FEATURE_RDM_METRIC)

        out_path = os.path.join(
            RDM_OUTDIR,
            f"RDM_OpenAI_CLIP_{token_mode}_{layer_name}.npy",
        )
        np.save(out_path, D.astype(np.float32))

        clip_rdm_info[token_mode][layer_name] = {
            "D": D,
            "vec": v,
            "n_features": int(X.shape[1]),
        }

        print(f"  {layer_name:30s}: RDM saved")


# =============================================================================
# [7] LAYER-WISE RSA WITH PERMUTATION CORRECTION
# =============================================================================

all_results = []

analysis_cache = {}

for model_rdm_type in MODEL_RDM_TYPES_TO_RUN:

    method = RSA_METHOD_BY_MODEL[model_rdm_type]

    print("\n" + "=" * 80)
    print(f"Model RDM: {model_rdm_type}")
    print(f"RSA method: {method}")
    print("=" * 80)

    model_D, model_vec = model_rdm_from_processing(
        processing_level=processing_level,
        processed_binary=processed_binary,
        rdm_type=model_rdm_type,
    )

    np.save(
        os.path.join(RDM_OUTDIR, f"model_RDM_processing_{model_rdm_type}.npy"),
        model_D.astype(np.float32),
    )

    for token_mode in TOKEN_MODES:
        print(f"\nToken mode: {token_mode}")

        layer_names = list(clip_rdm_info[token_mode].keys())
        feature_vecs = np.vstack([
            clip_rdm_info[token_mode][lname]["vec"]
            for lname in layer_names
        ])

        perm_res = permutation_test_across_layers(
            feature_vecs=feature_vecs,
            layer_names=layer_names,
            processing_level=processing_level,
            processed_binary=processed_binary,
            rdm_type=model_rdm_type,
            method=method,
            n_perm=N_PERMUTATIONS,
            seed=RANDOM_STATE,
        )

        analysis_cache[(model_rdm_type, token_mode)] = {
            "layer_names": layer_names,
            "feature_vecs": feature_vecs,
            **perm_res,
            "method": method,
        }

        # Save permutation arrays.
        np.save(
            os.path.join(
                PERM_OUTDIR,
                f"null_by_layer_{model_rdm_type}_{token_mode}_{method}.npy",
            ),
            perm_res["null_by_layer"].astype(np.float32),
        )
        np.save(
            os.path.join(
                PERM_OUTDIR,
                f"max_null_{model_rdm_type}_{token_mode}_{method}.npy",
            ),
            perm_res["max_null"].astype(np.float32),
        )

        for li, lname in enumerate(layer_names):
            # Parametric p-value for reference only.
            r_param, p_param = scipy_rsa_single(
                feature_vec=feature_vecs[li],
                model_vec=model_vec,
                method=method,
            )

            row = {
                "model_rdm_type": model_rdm_type,
                "rsa_method": method,
                "token_mode": token_mode,
                "layer_index_in_analysis": li,
                "layer_name": lname,
                "n_features": int(clip_rdm_info[token_mode][lname]["n_features"]),
                "feature_rdm_metric": FEATURE_RDM_METRIC,
                "rsa_r": float(perm_res["obs"][li]),
                "parametric_r_check": r_param,
                "parametric_p_reference_only": p_param,
                "perm_p_unc_one_sided_positive": float(perm_res["p_unc"][li]),
                "q_fdr_across_layers": float(perm_res["q_fdr"][li]),
                "perm_p_maxstat_fwe": float(perm_res["p_max"][li]),
                "null_mean_layer": float(np.nanmean(perm_res["null_by_layer"][:, li])),
                "null_sd_layer": float(np.nanstd(perm_res["null_by_layer"][:, li], ddof=1)),
                "valid_layer": bool(perm_res["valid_layers"][li]),
            }
            all_results.append(row)

            print(
                f"  {lname:30s} | "
                f"r={row['rsa_r']:+.3f} | "
                f"p={row['perm_p_unc_one_sided_positive']:.4f} | "
                f"q={row['q_fdr_across_layers']:.4f} | "
                f"p_max={row['perm_p_maxstat_fwe']:.4f}"
            )

results_df = pd.DataFrame(all_results)

results_csv = os.path.join(OUTDIR, "clip_openai_layerwise_processing_rsa_results.csv")
results_df.to_csv(results_csv, index=False)

print(f"\nSaved results CSV:")
print(results_csv)


# =============================================================================
# [8] KRIEGESKORTE-STYLE FIGURE
# =============================================================================

def clean_layer_label(layer_name):
    if layer_name == "layer_00_embedding":
        return "Emb."
    if layer_name == "layer_13_projected_embedding":
        return "Proj."
    if "transformer" in layer_name:
        return layer_name.replace("layer_", "L").replace("_transformer", "")
    return layer_name


def plot_rdm(ax, D, title, vmin=None, vmax=None, cmap="viridis"):
    im = ax.imshow(
        D,
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")

    return im


def add_colorbar_to_axis(ax, im, label=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.04)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, length=2)
    if label is not None:
        cb.set_label(label, fontsize=8)
    return cb


def representative_layers_for_figure(layer_names, token_mode):
    """
    Select representative layers for Panel B.
    For CLS, do not include the embedding stage.
    """
    if token_mode == "cls":
        wanted = [
            "layer_01_transformer",
            "layer_04_transformer",
            "layer_08_transformer",
            "layer_11_transformer",
            "layer_13_projected_embedding",
        ]
    else:
        wanted = [
            "layer_00_embedding",
            "layer_04_transformer",
            "layer_08_transformer",
            "layer_11_transformer",
            "layer_13_projected_embedding",
        ]

    return [x for x in wanted if x in layer_names]


def sorting_for_model_rdm(model_rdm_type):
    """
    Return stimulus order for plotting RDMs.
    Ordinal model is sorted by processing level.
    Binary model is sorted by binary label and then ordinal level.
    """
    if model_rdm_type == "ordinal":
        return np.lexsort((np.arange(len(CONDS)), processing_level))
    elif model_rdm_type == "binary":
        return np.lexsort((processing_level, processed_binary))
    else:
        return np.arange(len(CONDS))


def make_kriegeskorte_style_figure(model_rdm_type="ordinal", token_mode="cls"):

    key = (model_rdm_type, token_mode)
    cache = analysis_cache[key]

    method = cache["method"]
    layer_names = cache["layer_names"]
    obs = cache["obs"]
    q_fdr = cache["q_fdr"]
    p_max = cache["p_max"]
    max_null = cache["max_null"]
    model_D = cache["model_D"]

    sort_idx = sorting_for_model_rdm(model_rdm_type)
    model_D_sorted = model_D[np.ix_(sort_idx, sort_idx)]

    # Representative CLIP RDMs.
    reps = representative_layers_for_figure(layer_names, token_mode)

    # Peak layer by observed RSA.
    finite_obs = np.isfinite(obs)
    peak_i = np.where(finite_obs)[0][np.nanargmax(obs[finite_obs])]
    peak_layer = layer_names[peak_i]
    peak_r = float(obs[peak_i])
    peak_q = float(q_fdr[peak_i])
    peak_pmax = float(p_max[peak_i])

    fig = plt.figure(figsize=(15, 8.2), dpi=DPI)

    gs = GridSpec(
        nrows=2,
        ncols=7,
        figure=fig,
        height_ratios=[1.00, 1.20],
        width_ratios=[1.05, 1, 1, 1, 1, 1, 1.45],
        hspace=0.45,
        wspace=0.45,
    )

    # -------------------------------------------------------------------------
    # Panel A: model RDM
    # -------------------------------------------------------------------------
    ax_model = fig.add_subplot(gs[0, 0])

    if model_rdm_type == "ordinal":
        model_title = "Processing/preparation\nordinal model RDM"
        vmax = 1
    else:
        model_title = "Processed/prepared\nbinary model RDM"
        vmax = 1

    im = plot_rdm(
        ax_model,
        model_D_sorted,
        model_title,
        vmin=0,
        vmax=vmax,
        cmap="Greys",
    )
    add_colorbar_to_axis(ax_model, im, label="dissimilarity")

    ax_model.text(
        -0.25,
        1.12,
        "A",
        transform=ax_model.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # -------------------------------------------------------------------------
    # Panel B: representative CLIP RDMs
    # -------------------------------------------------------------------------
    D_rep_sorted = OrderedDict()
    rdm_min = np.inf
    rdm_max = -np.inf

    for lname in reps:
        D = clip_rdm_info[token_mode][lname]["D"]
        D = D[np.ix_(sort_idx, sort_idx)]
        D_rep_sorted[lname] = D

        vals = upper_tri_vec(D)
        rdm_min = min(rdm_min, float(np.nanmin(vals)))
        rdm_max = max(rdm_max, float(np.nanmax(vals)))

    for ri, lname in enumerate(reps):
        ax = fig.add_subplot(gs[0, ri + 1])

        li = layer_names.index(lname)
        r = obs[li]
        q = q_fdr[li]
        pm = p_max[li]

        sig_marker = ""
        if np.isfinite(pm) and pm < 0.05:
            sig_marker = " †"
        elif np.isfinite(q) and q < 0.05:
            sig_marker = " *"

        im = plot_rdm(
            ax,
            D_rep_sorted[lname],
            f"CLIP {clean_layer_label(lname)}\n{method} r = {r:+.2f}{sig_marker}",
            vmin=rdm_min,
            vmax=rdm_max,
            cmap="viridis",
        )

        if ri == len(reps) - 1:
            add_colorbar_to_axis(ax, im, label="1 − r")

        if ri == 0:
            ax.text(
                -0.25,
                1.12,
                "B",
                transform=ax.transAxes,
                fontsize=18,
                fontweight="bold",
                va="top",
                ha="left",
            )

    # -------------------------------------------------------------------------
    # Panel C: layer-wise RSA profile
    # -------------------------------------------------------------------------
    ax_line = fig.add_subplot(gs[1, 1:6])

    x = np.arange(len(layer_names))
    y = obs

    ax_line.axhline(0, color="black", lw=0.8, alpha=0.8)
    ax_line.plot(
        x,
        y,
        marker="o",
        lw=2.4,
        color="#2B5A9E",
        markersize=5,
        zorder=3,
    )

    # FDR significant layers: hollow markers.
    sig_fdr = np.isfinite(q_fdr) & (q_fdr < 0.05)
    ax_line.scatter(
        x[sig_fdr],
        y[sig_fdr],
        s=82,
        facecolor="white",
        edgecolor="#2B5A9E",
        linewidth=1.8,
        zorder=4,
        label="FDR q < .05",
    )

    # Max-stat significant layers: filled markers.
    sig_max = np.isfinite(p_max) & (p_max < 0.05)
    ax_line.scatter(
        x[sig_max],
        y[sig_max],
        s=46,
        facecolor="#2B5A9E",
        edgecolor="#2B5A9E",
        linewidth=1.0,
        zorder=5,
        label="max-stat p < .05",
    )

    ax_line.set_xticks(x)
    ax_line.set_xticklabels(
        [clean_layer_label(l) for l in layer_names],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    ax_line.set_ylabel(
        f"RSA correlation with\n{model_rdm_type} processing RDM ({method} r)",
        fontsize=10,
    )
    ax_line.set_xlabel("CLIP vision layer", fontsize=10)
    ax_line.set_title("Layer-wise processing/preparation structure", fontsize=11, pad=8)

    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)
    ax_line.tick_params(axis="both", labelsize=8)
    ax_line.legend(frameon=False, fontsize=8, loc="best")

    ax_line.text(
        -0.10,
        1.12,
        "C",
        transform=ax_line.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # -------------------------------------------------------------------------
    # Panel D: max-stat null distribution
    # -------------------------------------------------------------------------
    ax_null = fig.add_subplot(gs[1, 6])

    ax_null.hist(
        max_null,
        bins=45,
        color="0.75",
        edgecolor="white",
        linewidth=0.5,
    )
    ax_null.axvline(
        peak_r,
        color="#C94A2A",
        lw=2.7,
        label=f"observed peak r = {peak_r:+.2f}",
    )
    ax_null.axvline(
        np.nanmean(max_null),
        color="black",
        lw=1.2,
        linestyle="--",
        label="max-null mean",
    )

    ax_null.set_title(
        f"Peak layer: {clean_layer_label(peak_layer)}\n"
        f"q = {peak_q:.4f}, pmax = {peak_pmax:.4f}",
        fontsize=10,
        pad=8,
    )
    ax_null.set_xlabel("Max null RSA correlation", fontsize=9)
    ax_null.set_ylabel("Count", fontsize=9)

    ax_null.spines["top"].set_visible(False)
    ax_null.spines["right"].set_visible(False)
    ax_null.tick_params(axis="both", labelsize=8)
    ax_null.legend(frameon=False, fontsize=7, loc="best")

    ax_null.text(
        -0.23,
        1.12,
        "D",
        transform=ax_null.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # -------------------------------------------------------------------------
    # Figure title and note
    # -------------------------------------------------------------------------
    title_model = "ordinal processing/preparation" if model_rdm_type == "ordinal" else "binary processed/prepared"
    token_label = "CLS token" if token_mode == "cls" else "mean patch token"

    fig.suptitle(
        f"Food processing structure across CLIP vision layers ({token_label}, {title_model} RDM)",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    fig.text(
        0.5,
        0.012,
        (
            f"RDMs sorted by processing/preparation level. CLIP RDMs use "
            f"{FEATURE_RDM_METRIC} distance. "
            f"RSA method: {method}. "
            f"* = FDR q < .05; † = max-stat familywise p < .05. "
            f"For CLS-token analyses, the image-independent embedding-stage CLS token is omitted."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.25",
    )

    out_base = os.path.join(
        FIGDIR,
        f"openai_clip_layerwise_processing_rsa_{model_rdm_type}_{token_mode}_{method}",
    )

    if SAVE_PNG:
        fig.savefig(out_base + ".png", dpi=DPI, bbox_inches="tight", facecolor="white")

    if SAVE_PDF:
        fig.savefig(out_base + ".pdf", dpi=DPI, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print("\nSaved figure:")
    print(out_base + ".png")
    print(out_base + ".pdf")


for model_rdm_type in MODEL_RDM_TYPES_TO_RUN:
    for token_mode in MAKE_FIGURES_FOR_TOKEN_MODES:
        make_kriegeskorte_style_figure(
            model_rdm_type=model_rdm_type,
            token_mode=token_mode,
        )


# =============================================================================
# [9] SAVE CONFIG AND SUMMARY TABLES
# =============================================================================

config = {
    "MODEL_NAME": MODEL_NAME,
    "DEVICE": DEVICE,
    "BATCH_SIZE": BATCH_SIZE,
    "TOKEN_MODES": TOKEN_MODES,
    "OMIT_CLS_EMBEDDING_FROM_RSA": OMIT_CLS_EMBEDDING_FROM_RSA,
    "FEATURE_RDM_METRIC": FEATURE_RDM_METRIC,
    "MODEL_RDM_TYPES_TO_RUN": MODEL_RDM_TYPES_TO_RUN,
    "RSA_METHOD_BY_MODEL": RSA_METHOD_BY_MODEL,
    "N_PERMUTATIONS": N_PERMUTATIONS,
    "RANDOM_STATE": RANDOM_STATE,
    "IMGDIR": IMGDIR,
    "STIM_CSV": STIM_CSV,
    "OUTDIR": OUTDIR,
    "label_definition": {
        "level_0": "raw / natural produce",
        "level_1": "minimally processed or visually natural single-ingredient food",
        "level_2": "processed or prepared simple food",
        "level_3": "highly processed / composite / confectionery / fried / snack-like food",
        "binary_0": "levels 0-1: unprocessed / minimally processed",
        "binary_1": "levels 2-3: processed / prepared",
    },
}

with open(os.path.join(OUTDIR, "analysis_config.json"), "w") as f:
    json.dump(config, f, indent=2)

# Save readable label lists.
label_lists = OrderedDict()
for level in [0, 1, 2, 3]:
    label_lists[f"level_{level}"] = label_df.loc[
        label_df["processing_level_0_3"] == level,
        "condition",
    ].tolist()

label_lists["binary_unprocessed_minimal"] = label_df.loc[
    label_df["processed_binary"] == 0,
    "condition",
].tolist()

label_lists["binary_processed_prepared"] = label_df.loc[
    label_df["processed_binary"] == 1,
    "condition",
].tolist()

with open(os.path.join(OUTDIR, "processing_label_lists.json"), "w") as f:
    json.dump(label_lists, f, indent=2)

print("\nDone.")
print(f"Outputs saved to: {OUTDIR}")
print(f"Labels: {label_csv}")
print(f"Results: {results_csv}")
print(f"Figures: {FIGDIR}")