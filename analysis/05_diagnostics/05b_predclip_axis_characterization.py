# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:05:07 2026

@author: G.Marrazzo
"""

r"""
PredCLIP axis characterization v2
================================================

Purpose
-------
Characterize what the CLIP-predicted calorie axis actually captures.

This script tests whether the PredCLIP component of perceived calorie reflects:

    1. perceived calorie content
    2. food processedness / naturalness
    3. raw produce vs baked / confectionery / processed foods
    4. broader CLIP semantic category structure

Main additions over the earlier version
---------------------------------------
1. Validates stimulus order against the expected 96-image list.
2. Builds an explicit annotation table for:
       processedness
       naturalness
       preparation level
       fruit/vegetable
       bakery/confectionery
       composite/prepared food
       fast-food/snack structure
       animal product
       sweetness/savouriness
3. Computes contrastive CLIP text axes, e.g.:
       processed food - natural raw produce
       baked confectionery - fresh fruit/vegetables
       prepared/composite food - single-ingredient food
4. Tests how well these annotation axes explain:
       RawCalorie
       PredCLIP
       ResCLIP
5. Saves diagnostic residuals:
       PredCLIP residual after removing annotation/category structure

Important
---------
The residualized vectors saved here are diagnostic. If you use them in neural
encoding models, the cleanest implementation is still fold-wise residualisation
inside the neural outer CV fold.

Requires
--------
    FEAT_DIR/CLIP.npy
    FEAT_DIR/CLIP_full512.npy
        If missing, the script tries to extract it from images using OpenAI CLIP.

    DIAG_DIR/calorie_pred_cv_CLIP.npy
    DIAG_DIR/calorie_res_cv_CLIP.npy
    DIAG_DIR/calorie_group_raw_z.npy

    STIM_CSV = ordered_stimuli.csv
    IMGDIR   = image directory used during feature extraction
"""

# =============================================================================
# [0] IMPORTS
# =============================================================================

import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity

import torch
import clip as openai_clip

warnings.filterwarnings("ignore", category=RuntimeWarning)

print("Imports OK.")


# =============================================================================
# [1] PATHS
# =============================================================================

# RELEASE PATH ADAPTATION ONLY.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

STIMDIR = os.path.join(REPO_ROOT, "data", "stimuli")
IMGDIR = os.path.join(STIMDIR, "images")
STIM_CSV = os.path.join(STIMDIR, "ordered_stimuli.csv")

FEAT_DIR = os.path.join(REPO_ROOT, "reproduced_outputs", "feature_extraction")
DIAG_DIR = os.path.join(
    REPO_ROOT,
    "reproduced_outputs",
    "diagnostics",
    "perceived_calorie_prediction_diagnostics",
)
OUTDIR = os.path.join(
    REPO_ROOT,
    "reproduced_outputs",
    "diagnostics",
    "predCLIP_axis_characterization_v2",
)

os.makedirs(OUTDIR, exist_ok=True)

N_STIM = 96
N_GRID = 12
RANDOM_STATE = 42
ALPHAS = np.logspace(-4, 4, 25)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# [2] EXPECTED STIMULUS ORDER
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


# =============================================================================
# [3] SMALL HELPERS
# =============================================================================

def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return ((x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)).astype(np.float32)


def safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)

    if ok.sum() < 3:
        return np.nan, np.nan

    if np.nanstd(x[ok]) < 1e-12 or np.nanstd(y[ok]) < 1e-12:
        return np.nan, np.nan

    r, p = pearsonr(x[ok], y[ok])
    return float(r), float(p)


def safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)

    if ok.sum() < 3:
        return np.nan, np.nan

    if np.nanstd(x[ok]) < 1e-12 or np.nanstd(y[ok]) < 1e-12:
        return np.nan, np.nan

    r, p = spearmanr(x[ok], y[ok])
    return float(r), float(p)


def cv_predict_and_score(X, y, label, n_splits=5):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).squeeze()

    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        RidgeCV(alphas=ALPHAS),
    )

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pred = cross_val_predict(pipe, X, y, cv=cv)

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2) + 1e-8)

    r, p = safe_pearson(y, pred)
    rho, p_rho = safe_spearman(y, pred)

    return {
        "model": label,
        "n_features": int(X.shape[1]),
        "cv_r2": float(1.0 - ss_res / ss_tot),
        "pearson_r": r,
        "pearson_p": p,
        "spearman_rho": rho,
        "spearman_p": p_rho,
    }, pred.astype(np.float32)


def residualize_target(y, X, label):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).squeeze()

    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LinearRegression(),
    )

    pipe.fit(X, y)
    y_hat = pipe.predict(X).astype(np.float32)
    y_res = (y - y_hat).astype(np.float32)

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2) + 1e-8)

    return {
        "target": label,
        "r2_full_sample": float(1.0 - ss_res / ss_tot),
        "residual_variance_fraction": float(np.var(y_res) / (np.var(y) + 1e-8)),
    }, zscore(y_hat), zscore(y_res)


def cosine_rdm_vec(X):
    X = np.asarray(X, dtype=np.float32)
    S = cosine_similarity(X)
    D = 1.0 - S
    iu = np.triu_indices_from(D, k=1)
    return D[iu].astype(np.float32)


def nearest_neighbor_overlap(Xa, Xb, k=5):
    Sa = cosine_similarity(Xa)
    Sb = cosine_similarity(Xb)

    np.fill_diagonal(Sa, -np.inf)
    np.fill_diagonal(Sb, -np.inf)

    nna = np.argsort(Sa, axis=1)[:, -k:]
    nnb = np.argsort(Sb, axis=1)[:, -k:]

    overlaps = []

    for i in range(Xa.shape[0]):
        overlaps.append(len(set(nna[i]).intersection(set(nnb[i]))) / float(k))

    return float(np.mean(overlaps))


def get_img_path(cond):
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = os.path.join(IMGDIR, cond + ext)
        if os.path.isfile(p):
            return p

    if os.path.isdir(IMGDIR):
        low = cond.lower()
        for fname in os.listdir(IMGDIR):
            stem, ext = os.path.splitext(fname)
            if stem.lower() == low and ext.lower() in [".jpg", ".jpeg", ".png"]:
                return os.path.join(IMGDIR, fname)

    return None


def require_npy(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required file not found:\n  {path}")
    return np.load(path).astype(np.float32).squeeze()


# =============================================================================
# [4] LOAD STIMULUS ORDER
# =============================================================================

stim_df = pd.read_csv(STIM_CSV)
CONDS = stim_df["Value"].astype(str).tolist()

assert len(CONDS) == N_STIM, f"Expected {N_STIM} conditions, got {len(CONDS)}"

print(f"\nStimulus order loaded: {N_STIM} conditions")
print(f"  First 5: {CONDS[:5]}")
print(f"  Last 5 : {CONDS[-5:]}")

order_check = pd.DataFrame({
    "index": np.arange(N_STIM),
    "stim_csv_condition": CONDS,
    "expected_condition_from_user": EXPECTED_CONDS,
    "match": [a == b for a, b in zip(CONDS, EXPECTED_CONDS)],
})

order_check.to_csv(os.path.join(OUTDIR, "stimulus_order_check.csv"), index=False)

if not order_check["match"].all():
    print("\nWARNING: STIM_CSV order differs from EXPECTED_CONDS.")
    print("The script will use STIM_CSV as the canonical order.")
    print("Check saved file: stimulus_order_check.csv")
else:
    print("  Stimulus order matches EXPECTED_CONDS.")

stim_labels = [c.replace("_", " ")[:22] for c in CONDS]

img_paths = [get_img_path(c) for c in CONDS]
n_missing = sum(p is None for p in img_paths)

if n_missing:
    print(f"  WARNING: {n_missing} images not found in {IMGDIR}")
else:
    print(f"  All {N_STIM} images found.")


# =============================================================================
# [5] LOAD SCORES
# =============================================================================

pred_clip_z = require_npy(os.path.join(DIAG_DIR, "calorie_pred_cv_CLIP.npy"))
res_clip_z = require_npy(os.path.join(DIAG_DIR, "calorie_res_cv_CLIP.npy"))
calorie_raw_z = require_npy(os.path.join(DIAG_DIR, "calorie_group_raw_z.npy"))

assert pred_clip_z.shape == (N_STIM,)
assert res_clip_z.shape == (N_STIM,)
assert calorie_raw_z.shape == (N_STIM,)

pred_clip_z = zscore(pred_clip_z)
res_clip_z = zscore(res_clip_z)
calorie_raw_z = zscore(calorie_raw_z)

print("\nLoaded calorie decomposition scores:")
print(f"  PredCLIP range : {pred_clip_z.min():+.3f} -> {pred_clip_z.max():+.3f}")
print(f"  ResCLIP  range : {res_clip_z.min():+.3f} -> {res_clip_z.max():+.3f}")
print(f"  RawCal   range : {calorie_raw_z.min():+.3f} -> {calorie_raw_z.max():+.3f}")

score_df = pd.DataFrame({
    "index": np.arange(N_STIM),
    "condition": CONDS,
    "RawCalorie_z": calorie_raw_z,
    "PredCLIP_z": pred_clip_z,
    "ResCLIP_z": res_clip_z,
})


# =============================================================================
# [6] MANUAL / SEMI-MANUAL ANNOTATION TABLE
# =============================================================================

FRUIT = {
    "Apple", "Apricot", "Avocado", "Banana", "Blackberries", "Cherries",
    "Coconut", "Currant", "Figs", "Grape_red", "Grape_white",
    "Honey_melon", "Kiwi", "Mandarin", "Mango", "Melon", "Orange",
    "Papaya", "Pear", "Pineapple", "Pomegranate", "Raspberries",
    "Strawberries", "Watermelon",
}

VEGETABLE = {
    "Asparagus_green", "Aubergine", "Beetroot", "Broccoli",
    "Brussels_sprout", "Carrot", "Cauliflower", "Cherry_tomatoes",
    "Chickpeas", "Corn", "Cucumber", "Lettuce", "Mushrooms", "Olives",
    "Pees", "Pepper_red", "Radicchio", "Radish", "Zucchini",
}

NUTS_SEEDS = {
    "Almonds",
}

MEAT_FISH_EGG = {
    "Bacon", "Codfish", "Fried_egg", "Ham", "Herring", "Mussles",
    "Oyster", "Ribs", "Salami", "Smoked_sausage", "Tuna",
}

DAIRY_CHEESE = {
    "Blue_cheese", "Brie",
}

BAKERY_BREAD = {
    "Baguette", "Croissant", "Rice_waffle",
}

SWEET_BAKERY_CONFECTIONERY = {
    "Berliner", "Bonbon", "Brownie", "Butter_biscuit", "Cake", "Candies",
    "Cheesecake", "Choco_cone", "Choco_cookies", "Choco_lava_cake",
    "Chocolate_bar", "Chocolate_cake", "Cream_cake", "Donut", "Magnum",
    "Mars", "MMs", "Nutella_sandwich", "Pound_cake", "Tiramisu",
    "Tompoes", "Vanilla_icecream", "Waffle",
}

SAVORY_PREPARED = {
    "Bitterballs", "Cheese_souffle", "Crisps", "Fries", "Hamburger",
    "Hotdog", "Lasagna", "Pasta_pomodoro", "Pizza", "Popcorn",
    "Roasted_potatoes", "Sushi", "Muesli_bar",
}

FAST_FOOD_OR_FRIED = {
    "Bitterballs", "Cheese_souffle", "Fries", "Hamburger", "Hotdog",
    "Pizza", "Crisps",
}

COMPOSITE_FOOD = {
    "Bitterballs", "Cheesecake", "Cheese_souffle", "Chocolate_cake",
    "Choco_lava_cake", "Cream_cake", "Hamburger", "Hotdog", "Lasagna",
    "Nutella_sandwich", "Pasta_pomodoro", "Pizza", "Pound_cake",
    "Roasted_potatoes", "Sushi", "Tiramisu", "Tompoes", "Waffle",
}

ULTRA_PROCESSED = (
    SWEET_BAKERY_CONFECTIONERY
    | FAST_FOOD_OR_FRIED
    | {"Crisps", "Muesli_bar", "Popcorn", "Smoked_sausage", "Salami", "Bacon", "Ham"}
)

MINIMALLY_PROCESSED = {
    "Almonds", "Baguette", "Blue_cheese", "Brie", "Codfish", "Fried_egg",
    "Herring", "Mussles", "Oyster", "Rice_waffle", "Tuna",
}

PREPARED_LEVEL_2 = {
    "Baguette", "Blue_cheese", "Brie", "Codfish", "Fried_egg", "Herring",
    "Mussles", "Oyster", "Rice_waffle", "Roasted_potatoes", "Sushi",
    "Pasta_pomodoro",
}

PREPARED_LEVEL_3 = (
    SWEET_BAKERY_CONFECTIONERY
    | SAVORY_PREPARED
    | {"Bacon", "Ham", "Salami", "Smoked_sausage"}
)


def category_for_condition(cond):
    if cond in FRUIT:
        return "fruit"
    if cond in VEGETABLE:
        return "vegetable_legume"
    if cond in NUTS_SEEDS:
        return "nuts_seeds"
    if cond in MEAT_FISH_EGG:
        return "meat_fish_egg"
    if cond in DAIRY_CHEESE:
        return "dairy_cheese"
    if cond in SWEET_BAKERY_CONFECTIONERY:
        return "sweet_bakery_confectionery"
    if cond in SAVORY_PREPARED:
        return "savory_prepared"
    if cond in BAKERY_BREAD:
        return "bread_bakery"
    return "other"


def annotation_for_condition(cond):
    category = category_for_condition(cond)

    fruit_veg = int(cond in FRUIT or cond in VEGETABLE)
    animal_product = int(cond in MEAT_FISH_EGG or cond in DAIRY_CHEESE)
    bakery_confectionery = int(
        cond in SWEET_BAKERY_CONFECTIONERY
        or cond in BAKERY_BREAD
    )
    sweet_food = int(cond in FRUIT or cond in SWEET_BAKERY_CONFECTIONERY)
    savory_food = int(
        cond in VEGETABLE
        or cond in MEAT_FISH_EGG
        or cond in DAIRY_CHEESE
        or cond in SAVORY_PREPARED
        or cond in BAKERY_BREAD
        or cond in NUTS_SEEDS
    )
    dessert_snack = int(
        cond in SWEET_BAKERY_CONFECTIONERY
        or cond in {"Crisps", "Popcorn", "Muesli_bar", "Almonds"}
    )
    fast_food = int(cond in FAST_FOOD_OR_FRIED)
    composite_food = int(cond in COMPOSITE_FOOD)

    if cond in FRUIT or cond in VEGETABLE:
        processedness = 0
    elif cond in NUTS_SEEDS or cond in MINIMALLY_PROCESSED:
        processedness = 1
    elif cond in PREPARED_LEVEL_2:
        processedness = 2
    elif cond in ULTRA_PROCESSED or cond in PREPARED_LEVEL_3:
        processedness = 3
    else:
        processedness = 1

    if cond in FRUIT or cond in VEGETABLE:
        naturalness = 3
    elif cond in NUTS_SEEDS or cond in {"Codfish", "Herring", "Mussles", "Oyster", "Tuna"}:
        naturalness = 2
    elif cond in MINIMALLY_PROCESSED or cond in PREPARED_LEVEL_2:
        naturalness = 1
    else:
        naturalness = 0

    if cond in FRUIT or cond in VEGETABLE or cond in NUTS_SEEDS:
        preparation = 0
    elif cond in MINIMALLY_PROCESSED:
        preparation = 1
    elif cond in PREPARED_LEVEL_2:
        preparation = 2
    elif cond in PREPARED_LEVEL_3:
        preparation = 3
    else:
        preparation = 1

    raw_produce = int(cond in FRUIT or cond in VEGETABLE)
    single_ingredient = int(
        cond in FRUIT
        or cond in VEGETABLE
        or cond in NUTS_SEEDS
        or cond in MEAT_FISH_EGG
        or cond in DAIRY_CHEESE
    )

    return {
        "condition": cond,
        "category_manual": category,
        "processedness_0_3": processedness,
        "naturalness_0_3": naturalness,
        "preparation_0_3": preparation,
        "fruit_veg": fruit_veg,
        "raw_produce": raw_produce,
        "animal_product": animal_product,
        "sweet_food": sweet_food,
        "savory_food": savory_food,
        "dessert_snack": dessert_snack,
        "bakery_confectionery": bakery_confectionery,
        "fast_food_or_fried": fast_food,
        "composite_food": composite_food,
        "single_ingredient": single_ingredient,
    }


ann_df = pd.DataFrame([annotation_for_condition(c) for c in CONDS])
ann_df.insert(0, "index", np.arange(N_STIM))

full_df = score_df.merge(ann_df, on=["index", "condition"], how="left")

full_df.to_csv(os.path.join(OUTDIR, "stimulus_scores_and_manual_annotations.csv"), index=False)


print("\nSaved annotation table:")
print("  stimulus_scores_and_manual_annotations.csv")


# =============================================================================
# [7] SAVE BASIC ANNOTATION FEATURE VECTORS
# =============================================================================

annotation_feature_dir = os.path.join(OUTDIR, "annotation_feature_vectors")
os.makedirs(annotation_feature_dir, exist_ok=True)

ANNOTATION_NUMERIC_COLS = [
    "processedness_0_3",
    "naturalness_0_3",
    "preparation_0_3",
    "fruit_veg",
    "raw_produce",
    "animal_product",
    "sweet_food",
    "savory_food",
    "dessert_snack",
    "bakery_confectionery",
    "fast_food_or_fried",
    "composite_food",
    "single_ingredient",
]

for col in ANNOTATION_NUMERIC_COLS:
    np.save(
        os.path.join(annotation_feature_dir, f"{col}_z.npy"),
        zscore(full_df[col].to_numpy(dtype=np.float32)),
    )

print(f"\nSaved z-scored annotation feature vectors to:\n  {annotation_feature_dir}")


# =============================================================================
# [8] IMAGE GRIDS
# =============================================================================

def plot_image_grid(indices, scores, title, fname, score_label="score"):
    n = len(indices)
    ncols = min(6, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.25, nrows * 2.75))
    axes = np.asarray(axes).reshape(-1)

    for i, idx in enumerate(indices):
        ax = axes[i]
        path = img_paths[idx]

        if path and os.path.isfile(path):
            img = mpimg.imread(path)
            ax.imshow(img)
        else:
            ax.set_facecolor("#dddddd")
            ax.text(
                0.5,
                0.5,
                "no image",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=7,
            )

        ax.set_title(
            f"{stim_labels[idx]}\n"
            f"{score_label}={scores[idx]:+.2f}\n"
            f"proc={full_df.loc[idx, 'processedness_0_3']} "
            f"nat={full_df.loc[idx, 'naturalness_0_3']}",
            fontsize=7,
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTDIR, fname)
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {fname}")


print("\nPlotting ranked image grids...")

rank_pred = np.argsort(pred_clip_z)
top_pred = rank_pred[-N_GRID:][::-1]
bottom_pred = rank_pred[:N_GRID]

plot_image_grid(
    top_pred,
    pred_clip_z,
    f"Highest PredCLIP images (n={N_GRID})",
    "grid_top_predCLIP.png",
    "PredCLIP",
)

plot_image_grid(
    bottom_pred,
    pred_clip_z,
    f"Lowest PredCLIP images (n={N_GRID})",
    "grid_bottom_predCLIP.png",
    "PredCLIP",
)

rank_res = np.argsort(res_clip_z)
top_res = rank_res[-N_GRID:][::-1]
bottom_res = rank_res[:N_GRID]

plot_image_grid(
    top_res,
    res_clip_z,
    f"Highest ResCLIP images (n={N_GRID})",
    "grid_top_resCLIP.png",
    "ResCLIP",
)

plot_image_grid(
    bottom_res,
    res_clip_z,
    f"Lowest ResCLIP images (n={N_GRID})",
    "grid_bottom_resCLIP.png",
    "ResCLIP",
)


# =============================================================================
# [9] LOAD / EXTRACT FULL 512-DIM CLIP FEATURES
# =============================================================================

clip512_path = os.path.join(FEAT_DIR, "CLIP_full512.npy")

if os.path.isfile(clip512_path):
    clip_full_n = np.load(clip512_path).astype(np.float32)
    print(f"\nLoaded CLIP_full512.npy: {clip_full_n.shape}")
else:
    print(f"\nCLIP_full512.npy not found at:\n  {clip512_path}")
    print("Re-extracting CLIP_full512 from images using canonical STIM_CSV order.")

    clip_model, clip_preprocess = openai_clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    from PIL import Image

    clip_full = np.zeros((N_STIM, 512), dtype=np.float32)

    with torch.no_grad():
        for i, cond in enumerate(CONDS):
            path = img_paths[i]
            if path is None or not os.path.isfile(path):
                print(f"  WARNING: missing image for {cond}; leaving zeros.")
                continue

            img = clip_preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
            emb = clip_model.encode_image(img).cpu().numpy().astype(np.float32)
            clip_full[i] = emb.squeeze()

    clip_full_n = clip_full / (np.linalg.norm(clip_full, axis=1, keepdims=True) + 1e-8)
    np.save(clip512_path, clip_full_n)
    print(f"  Saved CLIP_full512.npy: {clip_full_n.shape}")

assert clip_full_n.shape == (N_STIM, 512)


# =============================================================================
# [10] CLIP ORDER / RDM SANITY CHECKS
# =============================================================================

clip_50d_path = os.path.join(FEAT_DIR, "CLIP.npy")
if not os.path.isfile(clip_50d_path):
    raise FileNotFoundError(f"Missing required file:\n  {clip_50d_path}")

clip_50d = np.load(clip_50d_path).astype(np.float32)
assert clip_50d.shape[0] == N_STIM

print("\nCLIP order / geometry sanity checks:")

rdm_50 = cosine_rdm_vec(StandardScaler().fit_transform(clip_50d))
rdm_512 = cosine_rdm_vec(clip_full_n)

rdm_r, rdm_p = safe_pearson(rdm_50, rdm_512)
nn_overlap_5 = nearest_neighbor_overlap(
    StandardScaler().fit_transform(clip_50d),
    clip_full_n,
    k=5,
)

summary_rows = []

summary_rows.append({
    "check": "RDM correlation: CLIP_50D vs CLIP_512D",
    "value": rdm_r,
    "p": rdm_p,
})

summary_rows.append({
    "check": "mean 5-nearest-neighbor overlap: CLIP_50D vs CLIP_512D",
    "value": nn_overlap_5,
    "p": np.nan,
})

print(f"  RDM corr 50D vs 512D       : r={rdm_r:+.3f}, p={rdm_p:.4g}")
print(f"  Mean 5-NN overlap 50D/512D : {nn_overlap_5:.3f}")

r2_50_info, pred_cal_50 = cv_predict_and_score(clip_50d, calorie_raw_z, "CLIP_50D_to_RawCalorie")
r2_512_info, pred_cal_512 = cv_predict_and_score(clip_full_n, calorie_raw_z, "CLIP_512D_to_RawCalorie")

pred_50_512_r, pred_50_512_p = safe_pearson(pred_cal_50, pred_cal_512)

summary_rows.append({
    "check": "CV R2: CLIP_50D -> RawCalorie",
    "value": r2_50_info["cv_r2"],
    "p": np.nan,
})

summary_rows.append({
    "check": "CV R2: CLIP_512D -> RawCalorie",
    "value": r2_512_info["cv_r2"],
    "p": np.nan,
})

summary_rows.append({
    "check": "Correlation: 50D-predicted calorie vs 512D-predicted calorie",
    "value": pred_50_512_r,
    "p": pred_50_512_p,
})

print(f"  CV R² 50D  -> raw calorie  : {r2_50_info['cv_r2']:+.3f}")
print(f"  CV R² 512D -> raw calorie  : {r2_512_info['cv_r2']:+.3f}")
print(f"  r(pred50, pred512)         : {pred_50_512_r:+.3f}")

pd.DataFrame(summary_rows).to_csv(
    os.path.join(OUTDIR, "clip_order_geometry_sanity_checks.csv"),
    index=False,
)

np.save(os.path.join(OUTDIR, "rawCalorie_pred_from_CLIP50_cv.npy"), zscore(pred_cal_50))
np.save(os.path.join(OUTDIR, "rawCalorie_pred_from_CLIP512_cv.npy"), zscore(pred_cal_512))


# =============================================================================
# [11] LOAD CLIP MODEL FOR TEXT PROBES
# =============================================================================

print("\nLoading CLIP model for text probes...")
clip_model, _ = openai_clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()


def encode_text_prompts(prompts):
    with torch.no_grad():
        tokens = openai_clip.tokenize(prompts).to(DEVICE)
        emb = clip_model.encode_text(tokens).cpu().numpy().astype(np.float32)

    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return emb


def prompt_axis_score(pos_prompts, neg_prompts):
    pos_emb = encode_text_prompts(pos_prompts)
    neg_emb = encode_text_prompts(neg_prompts)

    pos_mean = pos_emb.mean(axis=0)
    neg_mean = neg_emb.mean(axis=0)

    axis = pos_mean - neg_mean
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    scores = clip_full_n @ axis
    return zscore(scores)


# =============================================================================
# [12] SINGLE TEXT PROBE CORRELATIONS
# =============================================================================

TEXT_PROBES = [
    "a high calorie food",
    "a low calorie food",
    "a healthy food",
    "an unhealthy food",
    "a processed food",
    "an unprocessed natural food",
    "a raw vegetable or fruit",
    "fresh produce",
    "a baked dessert",
    "a confectionery food",
    "a packaged snack",
    "fast food",
    "a creamy food",
    "a fatty oily food",
    "a dense rich food",
    "a light food",
    "a fried food",
    "a sweet food",
    "a savory food",
    "a dessert",
    "a vegetable dish",
    "a fruit",
    "a snack",
    "a full meal",
    "bread or pastry",
    "meat or fish",
    "a salad",
    "a drink or beverage",
]

print("\nRunning single text probe correlations...")

probe_rows = []

for probe in TEXT_PROBES:
    emb = encode_text_prompts([probe])
    cosine = (clip_full_n @ emb.T).squeeze()
    cosine_z = zscore(cosine)

    r_pred, p_pred = safe_pearson(cosine_z, pred_clip_z)
    r_raw, p_raw = safe_pearson(cosine_z, calorie_raw_z)
    r_res, p_res = safe_pearson(cosine_z, res_clip_z)

    probe_rows.append({
        "probe": probe,
        "r_predCLIP": r_pred,
        "p_predCLIP": p_pred,
        "r_rawCalorie": r_raw,
        "p_rawCalorie": p_raw,
        "r_resCLIP": r_res,
        "p_resCLIP": p_res,
    })

probe_df = pd.DataFrame(probe_rows).sort_values("r_predCLIP", ascending=False)
probe_df.to_csv(os.path.join(OUTDIR, "single_text_probe_correlations.csv"), index=False)

print("\nTop text probes by correlation with PredCLIP:")
print(
    probe_df[["probe", "r_predCLIP", "r_rawCalorie", "r_resCLIP"]]
    .head(12)
    .round(3)
    .to_string(index=False)
)

fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#c0392b" if r > 0 else "#2980b9" for r in probe_df["r_predCLIP"]]
ax.barh(probe_df["probe"], probe_df["r_predCLIP"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson r with PredCLIP score")
ax.set_title("Single CLIP text-probe correlations with PredCLIP")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "single_text_probe_correlations.png"), dpi=220, bbox_inches="tight")
plt.close()


# =============================================================================
# [13] CONTRASTIVE CLIP SEMANTIC AXES
# =============================================================================

PROMPT_AXES = OrderedDict([
    (
        "processed_minus_natural",
        {
            "positive": [
                "a processed food",
                "a packaged food",
                "an ultra processed food",
                "a manufactured snack",
                "fast food",
                "a prepared processed food",
            ],
            "negative": [
                "an unprocessed natural food",
                "fresh produce",
                "a raw fruit",
                "a raw vegetable",
                "a single ingredient whole food",
                "a natural plant food",
            ],
        },
    ),
    (
        "baked_confectionery_minus_fresh_produce",
        {
            "positive": [
                "a baked dessert",
                "a pastry",
                "a cake",
                "a cookie",
                "a chocolate dessert",
                "a sweet confectionery food",
            ],
            "negative": [
                "fresh fruit",
                "fresh vegetables",
                "raw produce",
                "a salad vegetable",
                "a natural fruit",
                "a light fresh food",
            ],
        },
    ),
    (
        "prepared_composite_minus_single_ingredient",
        {
            "positive": [
                "a prepared meal",
                "a composite food",
                "a cooked dish",
                "a mixed ingredient food",
                "a restaurant dish",
                "a plated meal",
            ],
            "negative": [
                "a single ingredient food",
                "a whole food ingredient",
                "a raw fruit",
                "a raw vegetable",
                "plain fish",
                "plain nuts",
            ],
        },
    ),
    (
        "energy_dense_treat_minus_light_food",
        {
            "positive": [
                "an energy dense food",
                "a rich high calorie food",
                "a fatty sweet food",
                "a dense dessert",
                "a calorie dense snack",
                "a heavy rich food",
            ],
            "negative": [
                "a low calorie food",
                "a light food",
                "a watery fruit",
                "a fresh vegetable",
                "a salad",
                "a lean light food",
            ],
        },
    ),
    (
        "animal_savory_processed_minus_plant_fresh",
        {
            "positive": [
                "processed meat",
                "a savory meat food",
                "a sausage",
                "bacon",
                "a fried savory snack",
                "a fast food meat item",
            ],
            "negative": [
                "fresh fruit",
                "fresh vegetables",
                "plant produce",
                "a raw vegetable",
                "a natural plant food",
                "a fresh salad ingredient",
            ],
        },
    ),
])

print("\nComputing contrastive CLIP semantic axes...")

axis_scores = {}
axis_rows = []

for axis_name, prompts in PROMPT_AXES.items():
    score = prompt_axis_score(prompts["positive"], prompts["negative"])
    axis_scores[axis_name] = score

    r_pred, p_pred = safe_pearson(score, pred_clip_z)
    r_raw, p_raw = safe_pearson(score, calorie_raw_z)
    r_res, p_res = safe_pearson(score, res_clip_z)

    axis_rows.append({
        "axis": axis_name,
        "r_predCLIP": r_pred,
        "p_predCLIP": p_pred,
        "r_rawCalorie": r_raw,
        "p_rawCalorie": p_raw,
        "r_resCLIP": r_res,
        "p_resCLIP": p_res,
    })

    full_df[f"clip_axis_{axis_name}_z"] = score
    np.save(os.path.join(OUTDIR, f"clip_axis_{axis_name}_z.npy"), score)

axis_df = pd.DataFrame(axis_rows).sort_values("r_predCLIP", ascending=False)
axis_df.to_csv(os.path.join(OUTDIR, "contrastive_clip_axis_correlations.csv"), index=False)

print("\nContrastive CLIP axes:")
print(axis_df.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(axis_df))
ax.bar(x - 0.25, axis_df["r_predCLIP"], width=0.25, label="PredCLIP")
ax.bar(x, axis_df["r_rawCalorie"], width=0.25, label="RawCalorie")
ax.bar(x + 0.25, axis_df["r_resCLIP"], width=0.25, label="ResCLIP")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(axis_df["axis"], rotation=30, ha="right")
ax.set_ylabel("Pearson r")
ax.set_title("Contrastive CLIP semantic axes")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "contrastive_clip_axis_correlations.png"), dpi=220, bbox_inches="tight")
plt.close()


# =============================================================================
# [14] ANNOTATION CORRELATIONS WITH SCORES
# =============================================================================

print("\nAnnotation correlations with RawCalorie / PredCLIP / ResCLIP...")

ann_corr_rows = []

for col in ANNOTATION_NUMERIC_COLS:
    x = full_df[col].to_numpy(dtype=np.float32)

    r_raw, p_raw = safe_pearson(x, calorie_raw_z)
    r_pred, p_pred = safe_pearson(x, pred_clip_z)
    r_res, p_res = safe_pearson(x, res_clip_z)

    rho_raw, _ = safe_spearman(x, calorie_raw_z)
    rho_pred, _ = safe_spearman(x, pred_clip_z)
    rho_res, _ = safe_spearman(x, res_clip_z)

    ann_corr_rows.append({
        "annotation": col,
        "pearson_r_rawCalorie": r_raw,
        "pearson_p_rawCalorie": p_raw,
        "pearson_r_predCLIP": r_pred,
        "pearson_p_predCLIP": p_pred,
        "pearson_r_resCLIP": r_res,
        "pearson_p_resCLIP": p_res,
        "spearman_rho_rawCalorie": rho_raw,
        "spearman_rho_predCLIP": rho_pred,
        "spearman_rho_resCLIP": rho_res,
    })

ann_corr_df = pd.DataFrame(ann_corr_rows).sort_values("pearson_r_predCLIP", ascending=False)
ann_corr_df.to_csv(os.path.join(OUTDIR, "manual_annotation_correlations.csv"), index=False)

print(
    ann_corr_df[[
        "annotation",
        "pearson_r_predCLIP",
        "pearson_r_rawCalorie",
        "pearson_r_resCLIP",
    ]]
    .round(3)
    .to_string(index=False)
)

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(ann_corr_df))
ax.bar(x - 0.25, ann_corr_df["pearson_r_predCLIP"], width=0.25, label="PredCLIP")
ax.bar(x, ann_corr_df["pearson_r_rawCalorie"], width=0.25, label="RawCalorie")
ax.bar(x + 0.25, ann_corr_df["pearson_r_resCLIP"], width=0.25, label="ResCLIP")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(ann_corr_df["annotation"], rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Pearson r")
ax.set_title("Manual annotation correlations")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "manual_annotation_correlations.png"), dpi=220, bbox_inches="tight")
plt.close()


# =============================================================================
# [15] CATEGORY MEANS
# =============================================================================

cat_summary = (
    full_df
    .groupby("category_manual", sort=False)
    .agg(
        n=("condition", "count"),
        mean_rawCalorie_z=("RawCalorie_z", "mean"),
        mean_predCLIP_z=("PredCLIP_z", "mean"),
        mean_resCLIP_z=("ResCLIP_z", "mean"),
        mean_processedness=("processedness_0_3", "mean"),
        mean_naturalness=("naturalness_0_3", "mean"),
    )
    .reset_index()
    .sort_values("mean_predCLIP_z", ascending=False)
)

cat_summary.to_csv(os.path.join(OUTDIR, "category_score_summary.csv"), index=False)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(cat_summary))
ax.bar(x - 0.25, cat_summary["mean_predCLIP_z"], width=0.25, label="PredCLIP")
ax.bar(x, cat_summary["mean_rawCalorie_z"], width=0.25, label="RawCalorie")
ax.bar(x + 0.25, cat_summary["mean_resCLIP_z"], width=0.25, label="ResCLIP")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cat_summary["category_manual"], rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Mean z-score")
ax.set_title("Mean scores by manual food category")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "category_score_summary.png"), dpi=220, bbox_inches="tight")
plt.close()


# =============================================================================
# [16] PREDICTION OF RAW / PRED / RES FROM ANNOTATIONS
# =============================================================================

print("\nCross-validated prediction from manual annotations...")

X_num = full_df[ANNOTATION_NUMERIC_COLS].to_numpy(dtype=np.float32)

cat_dummies = pd.get_dummies(full_df["category_manual"], prefix="cat").astype(np.float32)
X_cat = cat_dummies.to_numpy(dtype=np.float32)

X_num_cat = np.hstack([X_num, X_cat]).astype(np.float32)

X_clip_axes = np.column_stack([axis_scores[k] for k in axis_scores.keys()]).astype(np.float32)
X_num_clipaxes = np.hstack([X_num, X_clip_axes]).astype(np.float32)

target_dict = OrderedDict([
    ("RawCalorie", calorie_raw_z),
    ("PredCLIP", pred_clip_z),
    ("ResCLIP", res_clip_z),
])

predictor_sets = OrderedDict([
    ("manual_numeric", X_num),
    ("manual_category_dummies", X_cat),
    ("manual_numeric_plus_category", X_num_cat),
    ("contrastive_clip_axes", X_clip_axes),
    ("manual_numeric_plus_clip_axes", X_num_clipaxes),
])

pred_rows = []
cv_pred_store = {}

for target_name, y in target_dict.items():
    for pred_name, X in predictor_sets.items():
        info, pred = cv_predict_and_score(X, y, f"{pred_name}_to_{target_name}")
        info["target"] = target_name
        info["predictor_set"] = pred_name
        pred_rows.append(info)
        cv_pred_store[(target_name, pred_name)] = pred

pred_summary = pd.DataFrame(pred_rows)
pred_summary.to_csv(os.path.join(OUTDIR, "annotation_prediction_cv_summary.csv"), index=False)

print(
    pred_summary[[
        "target",
        "predictor_set",
        "n_features",
        "cv_r2",
        "pearson_r",
        "spearman_rho",
    ]]
    .sort_values(["target", "cv_r2"], ascending=[True, False])
    .round(3)
    .to_string(index=False)
)

for (target_name, pred_name), pred in cv_pred_store.items():
    np.save(
        os.path.join(OUTDIR, f"cv_pred_{target_name}_from_{pred_name}.npy"),
        zscore(pred),
    )


# =============================================================================
# [17] FULL-SAMPLE RESIDUALIZATION DIAGNOSTICS
# =============================================================================

print("\nFull-sample residualization diagnostics.")

# This is the control set I would first use to test whether PredCLIP is mainly
# processedness/naturalness/category structure.
CONTROL_COLS_CORE = [
    "processedness_0_3",
    "naturalness_0_3",
    "preparation_0_3",
    "fruit_veg",
    "raw_produce",
    "bakery_confectionery",
    "fast_food_or_fried",
    "composite_food",
    "single_ingredient",
]

X_control_core = full_df[CONTROL_COLS_CORE].to_numpy(dtype=np.float32)

# Include category dummies too for a stronger category-control diagnostic.
X_control_core_cat = np.hstack([X_control_core, X_cat]).astype(np.float32)

resid_rows = []

info, pred_hat_core, pred_res_core = residualize_target(
    pred_clip_z,
    X_control_core,
    "PredCLIP_residualized_from_core_annotations",
)
info["control_set"] = "core_annotations"
resid_rows.append(info)

info, pred_hat_cat, pred_res_cat = residualize_target(
    pred_clip_z,
    X_control_core_cat,
    "PredCLIP_residualized_from_core_annotations_plus_category",
)
info["control_set"] = "core_annotations_plus_category"
resid_rows.append(info)

resid_summary = pd.DataFrame(resid_rows)
resid_summary.to_csv(os.path.join(OUTDIR, "predCLIP_residualization_summary.csv"), index=False)

np.save(os.path.join(OUTDIR, "PredCLIP_hat_from_core_annotations_z.npy"), pred_hat_core)
np.save(os.path.join(OUTDIR, "PredCLIP_resid_from_core_annotations_z.npy"), pred_res_core)

np.save(os.path.join(OUTDIR, "PredCLIP_hat_from_core_annotations_plus_category_z.npy"), pred_hat_cat)
np.save(os.path.join(OUTDIR, "PredCLIP_resid_from_core_annotations_plus_category_z.npy"), pred_res_cat)

full_df["PredCLIP_hat_core_annotations_z"] = pred_hat_core
full_df["PredCLIP_resid_core_annotations_z"] = pred_res_core
full_df["PredCLIP_hat_core_annotations_plus_category_z"] = pred_hat_cat
full_df["PredCLIP_resid_core_annotations_plus_category_z"] = pred_res_cat

print(resid_summary.round(3).to_string(index=False))


# =============================================================================
# [18] SCATTER DIAGNOSTICS
# =============================================================================

scatter_specs = [
    (calorie_raw_z, pred_clip_z, "RawCalorie z", "PredCLIP z", "scatter_raw_vs_predCLIP.png"),
    (calorie_raw_z, res_clip_z, "RawCalorie z", "ResCLIP z", "scatter_raw_vs_resCLIP.png"),
    (pred_clip_z, res_clip_z, "PredCLIP z", "ResCLIP z", "scatter_predCLIP_vs_resCLIP.png"),
    (
        full_df["processedness_0_3"].to_numpy(dtype=float),
        pred_clip_z,
        "Processedness 0-3",
        "PredCLIP z",
        "scatter_processedness_vs_predCLIP.png",
    ),
    (
        full_df["naturalness_0_3"].to_numpy(dtype=float),
        pred_clip_z,
        "Naturalness 0-3",
        "PredCLIP z",
        "scatter_naturalness_vs_predCLIP.png",
    ),
    (
        full_df["clip_axis_processed_minus_natural_z"].to_numpy(dtype=float),
        pred_clip_z,
        "CLIP processed-minus-natural axis z",
        "PredCLIP z",
        "scatter_clip_processed_axis_vs_predCLIP.png",
    ),
]

for x, y, xl, yl, fname in scatter_specs:
    r, p = safe_pearson(x, y)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.scatter(x, y, s=35, alpha=0.82)
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.axvline(0, color="0.6", linewidth=0.8)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(f"{xl} vs {yl}\nr={r:+.3f}, p={p:.4g}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=220, bbox_inches="tight")
    plt.close()


# =============================================================================
# [19] RANKED CONSOLE LISTS + CSV
# =============================================================================

rank_table = full_df.copy()
rank_table["rank_predCLIP_low_to_high"] = rank_table["PredCLIP_z"].rank(method="first")
rank_table["rank_rawCalorie_low_to_high"] = rank_table["RawCalorie_z"].rank(method="first")
rank_table["rank_resCLIP_low_to_high"] = rank_table["ResCLIP_z"].rank(method="first")

rank_table.to_csv(os.path.join(OUTDIR, "stimulus_rank_table_full.csv"), index=False)

print("\nTop 12 images by PredCLIP score:")
for idx in top_pred:
    print(
        f"  [{idx:3d}] {CONDS[idx]:30s}  "
        f"PredCLIP={pred_clip_z[idx]:+.3f}  "
        f"RawCal={calorie_raw_z[idx]:+.3f}  "
        f"ResCLIP={res_clip_z[idx]:+.3f}  "
        f"cat={full_df.loc[idx, 'category_manual']:28s}  "
        f"proc={full_df.loc[idx, 'processedness_0_3']}  "
        f"nat={full_df.loc[idx, 'naturalness_0_3']}"
    )

print("\nBottom 12 images by PredCLIP score:")
for idx in bottom_pred:
    print(
        f"  [{idx:3d}] {CONDS[idx]:30s}  "
        f"PredCLIP={pred_clip_z[idx]:+.3f}  "
        f"RawCal={calorie_raw_z[idx]:+.3f}  "
        f"ResCLIP={res_clip_z[idx]:+.3f}  "
        f"cat={full_df.loc[idx, 'category_manual']:28s}  "
        f"proc={full_df.loc[idx, 'processedness_0_3']}  "
        f"nat={full_df.loc[idx, 'naturalness_0_3']}"
    )


# =============================================================================
# [20] FINAL SAVE
# =============================================================================

full_df.to_csv(os.path.join(OUTDIR, "stimulus_scores_annotations_axes_residuals_FULL.csv"), index=False)

print("\nKey outputs:")
print("  stimulus_scores_annotations_axes_residuals_FULL.csv")
print("  contrastive_clip_axis_correlations.csv")
print("  manual_annotation_correlations.csv")
print("  annotation_prediction_cv_summary.csv")
print("  predCLIP_residualization_summary.csv")
print("  PredCLIP_resid_from_core_annotations_z.npy")
print("  PredCLIP_resid_from_core_annotations_plus_category_z.npy")

print(f"\nAll outputs saved to:\n{OUTDIR}")
print("Done.")