# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:10:14 2026

@author: G.Marrazzo
"""

# -*- coding: utf-8 -*-
"""
Feature extraction — NeuroRepFood encoding model
=================================================
Extracts and saves all feature bands needed for banded ridge encoding.

Bands:
  1. Gabor          — precomputed V1-style Gabor filter bank
  2. Color          — Lab histogram (8^3 bins, correlation distance)
  3. AlexNetMid     — conv3 + conv5 PCA 100D  (mid-level visual)
  4. AlexNetHigh    — fc6 PCA 50D             (high-level visual)
  5. CORnetV4       — precomputed CORnet V4 PCA
  6. CORnetIT       — precomputed CORnet IT PCA
  7. CLIP           — ViT-B/32 PCA 100D       (semantic visual)
  8. Palatability   — group-mean z-scored ratings (1D)
  9. Calorie        — group-mean z-scored ratings (1D)
  10. Health        — group-mean z-scored ratings (1D)
  11. Familiarity   — group-mean z-scored ratings (1D)
  12. SavorySweet   — binary categorical (1D)
  13. CalorieObj    — binary categorical high/low (1D)

All outputs are (96, D) float32, aligned to ordered_stimuli.csv ordering.
Behavioural ratings also saved per-subject as (96, N_subjects) arrays.

Outputs saved to: OUTDIR/feature_models/
"""

# =============================================================================
# [0] IMPORTS
# =============================================================================

import os
import glob
import numpy as np
import pandas as pd
import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.color import rgb2lab
from scipy.spatial.distance import pdist

print("Imports OK")


# =============================================================================
# [1] PATHS + CONFIG
# =============================================================================

# RELEASE PATH ADAPTATION ONLY.
# Everything below this path block retains the executed historical numerical code.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

MAINDIR   = os.path.join(REPO_ROOT, 'data')
STIMDIR   = os.path.join(MAINDIR, 'stimuli')
IMGDIR    = os.path.join(STIMDIR, 'images')
FEAT_DIR  = os.path.join(STIMDIR, 'cornet_features')
OUTDIR    = os.path.join(REPO_ROOT, 'reproduced_outputs', 'feature_extraction')
os.makedirs(OUTDIR, exist_ok=True)

STIM_CSV  = os.path.join(STIMDIR, 'ordered_stimuli.csv')
GABOR_NPZ = os.path.join(STIMDIR, 'gabor_v1_rsa_model.npz')

PILOT_SUBJECTS = [
    104, 105, 109, 112, 115, 117, 118, 122, 124, 125, 126,
    129, 132, 134, 137, 139, 140, 141, 143, 144, 145, 146, 149,
    150, 151
]

ALEXNET_MID_PCA  = 50   # was 100 — capped at n_samples=96
ALEXNET_HIGH_PCA = 50   # unchanged but now consistent
CLIP_PCA         = 50   # was 100 — same reason
RANDOM_STATE     = 42

# =============================================================================
# [2] CANONICAL STIMULUS ORDER
# =============================================================================

print('\n[2] Loading stimulus order')

stim_df = pd.read_csv(STIM_CSV, delimiter=',')
CONDS   = stim_df['Value'].tolist()   # list of 96 food name strings
N       = len(CONDS)

print(f'  N conditions : {N}')
print(f'  First 5      : {CONDS[:5]}')
print(f'  Last 5       : {CONDS[-5:]}')

# Verify all images exist
missing_imgs = [c for c in CONDS
                if not os.path.isfile(os.path.join(IMGDIR, f'{c}.jpg'))]
if missing_imgs:
    print(f'  WARNING: {len(missing_imgs)} images missing: {missing_imgs}')
else:
    print(f'  All {N} images found in {IMGDIR}')

# Save canonical ordering reference
np.save(os.path.join(OUTDIR, 'stimulus_names.npy'),
        np.array(CONDS, dtype=object))
print(f'  stimulus_names.npy saved')


# =============================================================================
# [3] CATEGORICAL LABELS
# =============================================================================

print('\n[3] Categorical features')

# ── definitions from RSA script ──────────────────────────────────────────────
high_cal_savory = [
    'Almonds','Avocado','Bacon','Baguette','Bitterballs','Blue_cheese','Brie',
    'Cheese_souffle','Crisps','Fried_egg','Fries','Ham','Hamburger','Herring',
    'Hotdog','Lasagna','Olives','Pasta_pomodoro','Pizza','Ribs','Roasted_potatoes',
    'Salami','Smoked_sausage','Sushi'
]
high_cal_sweet = [
    'Berliner','Bonbon','Brownie','Butter_biscuit','Cake','Candies','Pound_cake',
    'Cheesecake','Choco_cookies','Choco_lava_cake','Choco_cone','Chocolate_bar',
    'Chocolate_cake','Cream_cake','Croissant','Donut','Magnum','Mars','MMs',
    'Nutella_sandwich','Tiramisu','Tompoes','Vanilla_icecream','Waffle'
]
low_cal_savory = [
    'Asparagus_green','Aubergine','Beetroot','Broccoli','Brussels_sprout','Carrot',
    'Cauliflower','Cherry_tomatoes','Chickpeas','Codfish','Corn','Cucumber',
    'Lettuce','Mushrooms','Mussles','Oyster','Pees','Pepper_red','Popcorn',
    'Radish','Radicchio','Rice_waffle','Tuna','Zucchini'
]
low_cal_sweet = [
    'Apple','Apricot','Banana','Blackberries','Cherries','Coconut','Currant',
    'Figs','Grape_red','Grape_white','Honey_melon','Kiwi','Mandarin','Mango',
    'Melon','Muesli_bar','Orange','Papaya','Pear','Pineapple','Pomegranate',
    'Raspberries','Strawberries','Watermelon'
]

high_caloric = set(high_cal_savory + high_cal_sweet)
low_caloric  = set(low_cal_savory  + low_cal_sweet)
savory_set   = set(high_cal_savory + low_cal_savory)
sweet_set    = set(high_cal_sweet  + low_cal_sweet)

calorie_obj = np.array([1 if c in high_caloric else 0 for c in CONDS],
                       dtype=np.float32)
savory_obj  = np.array([1 if c in savory_set  else 0 for c in CONDS],
                       dtype=np.float32)

# check all stimuli are covered
n_missing_cal = np.sum(np.isnan(
    np.array([np.nan if c not in (high_caloric | low_caloric) else 0
              for c in CONDS])
))
n_missing_sav = np.sum(np.isnan(
    np.array([np.nan if c not in (savory_set | sweet_set) else 0
              for c in CONDS])
))
print(f'  CalorieObj : coverage {N - n_missing_cal}/{N}')
print(f'  SavorySweet: coverage {N - n_missing_sav}/{N}')

# scale to z-score equivalent (mean 0, std 1) for encoding model
feat_calorie_obj = StandardScaler().fit_transform(calorie_obj.reshape(-1, 1))
feat_savory      = StandardScaler().fit_transform(savory_obj.reshape(-1, 1))

np.save(os.path.join(OUTDIR, 'CalorieObjective.npy'), feat_calorie_obj.astype(np.float32))
np.save(os.path.join(OUTDIR, 'SavorySweet.npy'),      feat_savory.astype(np.float32))
print(f'  CalorieObjective.npy : {feat_calorie_obj.shape}  saved')
print(f'  SavorySweet.npy      : {feat_savory.shape}  saved')


# =============================================================================
# [4] BEHAVIORAL RATINGS — PER SUBJECT + GROUP MEAN
# =============================================================================

print('\n[4] Behavioral ratings')

QUESTIONS = {
    'Palatability': 'Palatability',
    'Calorie':      'Calorie content',
    'Health':       'Health value',
    'Familiarity':  'Familiarity',
}

rating_store = {q: [] for q in QUESTIONS}   # q -> list of (96,) arrays

for subj in PILOT_SUBJECTS:
    rating_path = os.path.join(MAINDIR, 'ratings', f'{subj}_rating.txt')
    if not os.path.isfile(rating_path):
        print(f'  WARNING: ratings missing for sub-{subj}: {rating_path}')
        continue

    df = pd.read_csv(rating_path, delimiter='\t')
    df.columns = df.columns.str.strip()

    for key, q_label in QUESTIONS.items():
        q_df = df[df['question'] == q_label].copy()
        q_df['picture'] = pd.Categorical(q_df['picture'],
                                         categories=CONDS, ordered=True)
        q_df = q_df.sort_values('picture')

        if len(q_df) != N:
            print(f'  WARNING: sub-{subj} {key}: {len(q_df)} rows, expected {N}')
            rating_store[key].append(np.full(N, np.nan, dtype=np.float32))
        else:
            rating_store[key].append(
                np.asarray(q_df['rating'], dtype=np.float32)
            )

# Save per-subject matrices and group-mean z-scored feature vectors
for key in QUESTIONS:
    mat = np.column_stack(rating_store[key])   # (96, n_subjects)
    print(f'  {key}: matrix shape {mat.shape}  '
          f'NaN subjects: {np.isnan(mat).any(axis=0).sum()}')

    # save per-subject (96, n_subj) — useful for subject-specific RSA later
    np.save(os.path.join(OUTDIR, f'{key}_persubject.npy'), mat.astype(np.float32))

    # group mean: z-score each subject then average across subjects
    mat_z = ((mat - np.nanmean(mat, axis=0, keepdims=True))
             / np.nanstd(mat,  axis=0, keepdims=True, ddof=1))
    mean_z = np.nanmean(mat_z, axis=1)   # (96,)

    # final z-score across conditions for encoding model
    feat = StandardScaler().fit_transform(mean_z.reshape(-1, 1))
    np.save(os.path.join(OUTDIR, f'{key}.npy'), feat.astype(np.float32))
    print(f'  {key}.npy (group mean z) : {feat.shape}  saved')


# =============================================================================
# [5] GABOR — precomputed
# =============================================================================

print('\n[5] Gabor features')

gabordata = np.load(GABOR_NPZ, allow_pickle=True)
gabor_raw = gabordata['X'].astype(np.float32)   # (96, D)
print(f'  Raw Gabor shape: {gabor_raw.shape}')

assert gabor_raw.shape[0] == N, \
    f'Gabor has {gabor_raw.shape[0]} rows, expected {N}'

feat_gabor = StandardScaler().fit_transform(gabor_raw)
np.save(os.path.join(OUTDIR, 'Gabor.npy'), feat_gabor.astype(np.float32))
print(f'  Gabor.npy : {feat_gabor.shape}  saved')


# =============================================================================
# [6] CORnet V4 and IT — precomputed PCA matrices
# =============================================================================

print('\n[6] CORnet features')

V4 = np.load(os.path.join(FEAT_DIR, 'V4_PCA_matrix.npy')).astype(np.float32)
IT = np.load(os.path.join(FEAT_DIR, 'IT_PCA_matrix.npy')).astype(np.float32)

print(f'  V4 raw: {V4.shape}   IT raw: {IT.shape}')
assert V4.shape[0] == N and IT.shape[0] == N, \
    f'CORnet matrices must have {N} rows'

feat_v4 = StandardScaler().fit_transform(V4)
feat_it = StandardScaler().fit_transform(IT)

np.save(os.path.join(OUTDIR, 'CORnetV4.npy'), feat_v4.astype(np.float32))
np.save(os.path.join(OUTDIR, 'CORnetIT.npy'), feat_it.astype(np.float32))
print(f'  CORnetV4.npy : {feat_v4.shape}  saved')
print(f'  CORnetIT.npy : {feat_it.shape}  saved')


# =============================================================================
# [7] COLOR — Lab histogram features
# =============================================================================

print('\n[7] Color features (Lab histogram)')

def load_rgb(path, size=(224, 224)):
    im = Image.open(path).convert('RGB').resize(size, Image.BILINEAR)
    return np.asarray(im).astype(np.float32) / 255.0

def lab_hist_features(rgb, bins=8):
    lab = rgb2lab(rgb)
    L_edges = np.linspace(0,   100, bins + 1)
    a_edges = np.linspace(-128, 127, bins + 1)
    b_edges = np.linspace(-128, 127, bins + 1)
    hist, _ = np.histogramdd(
        np.stack([lab[..., 0].ravel(),
                  lab[..., 1].ravel(),
                  lab[..., 2].ravel()], axis=1),
        bins=(L_edges, a_edges, b_edges)
    )
    feat = hist.ravel().astype(np.float32)
    feat /= feat.sum() + 1e-12
    return feat

color_feats = []
for cond in CONDS:
    img_path = os.path.join(IMGDIR, f'{cond}.jpg')
    rgb = load_rgb(img_path)
    color_feats.append(lab_hist_features(rgb))

feat_color = np.stack(color_feats).astype(np.float32)   # (96, 512)
feat_color = StandardScaler().fit_transform(feat_color)
np.save(os.path.join(OUTDIR, 'Color.npy'), feat_color.astype(np.float32))
print(f'  Color.npy : {feat_color.shape}  saved')


# =============================================================================
# [8] ALEXNET — conv3, conv5, fc6
# =============================================================================

print('\n[8] AlexNet features')

alexnet = tv_models.alexnet(pretrained=True).eval()

_hooks = {}
def _make_hook(name):
    def fn(module, inp, out):
        _hooks[name] = out.detach().cpu().numpy()
    return fn

alexnet.features[6].register_forward_hook(_make_hook('conv3'))
alexnet.features[10].register_forward_hook(_make_hook('conv5'))
alexnet.classifier[4].register_forward_hook(_make_hook('fc6'))

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

conv3_list, conv5_list, fc6_list = [], [], []

for cond in CONDS:
    img_path = os.path.join(IMGDIR, f'{cond}.jpg')
    img = Image.open(img_path).convert('RGB')
    x   = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        alexnet(x)
    conv3_list.append(_hooks['conv3'].flatten())
    conv5_list.append(_hooks['conv5'].flatten())
    fc6_list.append(  _hooks['fc6'].flatten())

conv3_raw = np.stack(conv3_list).astype(np.float32)
conv5_raw = np.stack(conv5_list).astype(np.float32)
fc6_raw   = np.stack(fc6_list).astype(np.float32)
print(f'  conv3: {conv3_raw.shape}  conv5: {conv5_raw.shape}  fc6: {fc6_raw.shape}')

# Mid-level: conv3 + conv5 concatenated → PCA
mid_raw    = np.hstack([conv3_raw, conv5_raw])
pca_mid    = PCA(n_components=ALEXNET_MID_PCA, random_state=RANDOM_STATE)
feat_mid   = pca_mid.fit_transform(mid_raw).astype(np.float32)
var_mid    = pca_mid.explained_variance_ratio_.sum()
print(f'  AlexNetMid PCA {ALEXNET_MID_PCA}D var explained: {var_mid:.3f}')

# High-level: fc6 → PCA
pca_high   = PCA(n_components=ALEXNET_HIGH_PCA, random_state=RANDOM_STATE)
feat_high  = pca_high.fit_transform(fc6_raw).astype(np.float32)
var_high   = pca_high.explained_variance_ratio_.sum()
print(f'  AlexNetHigh PCA {ALEXNET_HIGH_PCA}D var explained: {var_high:.3f}')

feat_mid  = StandardScaler().fit_transform(feat_mid)
feat_high = StandardScaler().fit_transform(feat_high)

np.save(os.path.join(OUTDIR, 'AlexNetMid.npy'),  feat_mid.astype(np.float32))
np.save(os.path.join(OUTDIR, 'AlexNetHigh.npy'), feat_high.astype(np.float32))
print(f'  AlexNetMid.npy  : {feat_mid.shape}  saved')
print(f'  AlexNetHigh.npy : {feat_high.shape}  saved')


# =============================================================================
# [9] CLIP — ViT-B/32
# =============================================================================

print('\n[9] CLIP features')

try:
    import clip
    clip_available = True
except ImportError:
    clip_available = False
    print('  WARNING: clip not installed. Run: pip install git+https://github.com/openai/CLIP.git')

if clip_available:
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, clip_preprocess = clip.load('ViT-B/32', device=device)
    clip_model.eval()

    clip_feats = []
    for cond in CONDS:
        img_path = os.path.join(IMGDIR, f'{cond}.jpg')
        img  = clip_preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(img).cpu().numpy().flatten()
        clip_feats.append(feat.astype(np.float32))

    clip_raw  = np.stack(clip_feats).astype(np.float32)   # (96, 512)
    print(f'  CLIP raw: {clip_raw.shape}')

    pca_clip   = PCA(n_components=CLIP_PCA, random_state=RANDOM_STATE)
    feat_clip  = pca_clip.fit_transform(clip_raw).astype(np.float32)
    var_clip   = pca_clip.explained_variance_ratio_.sum()
    print(f'  CLIP PCA {CLIP_PCA}D var explained: {var_clip:.3f}')

    feat_clip  = StandardScaler().fit_transform(feat_clip)
    np.save(os.path.join(OUTDIR, 'CLIP.npy'), feat_clip.astype(np.float32))
    print(f'  CLIP.npy : {feat_clip.shape}  saved')


# =============================================================================
# [10] SUMMARY + INTER-MODEL CORRELATION CHECK
# =============================================================================

print(f'\n{"="*60}')
print(f'Feature models saved to: {OUTDIR}')
print(f'{"="*60}')

saved = {}
for fname in sorted(os.listdir(OUTDIR)):
    if (fname.endswith('.npy')
            and not fname.startswith('stimulus')
            and not fname.endswith('_persubject.npy')):
        arr = np.load(os.path.join(OUTDIR, fname))
        saved[fname.replace('.npy', '')] = arr
        print(f'  {fname.replace(".npy",""):25s}: {arr.shape}')

# Inter-model correlation via first PC of each
print(f'\nInter-model correlations (first PC of each):')
first_pcs = {}
for name, arr in saved.items():
    if arr.ndim == 1 or arr.shape[1] == 1:
        first_pcs[name] = arr.flatten()
    else:
        first_pcs[name] = PCA(n_components=1,
                              random_state=RANDOM_STATE).fit_transform(arr).flatten()

model_names = list(first_pcs.keys())
header = f'{"":25s}' + ''.join(f'{n[:9]:>10s}' for n in model_names)
print(header)
for n1 in model_names:
    row = f'{n1[:25]:25s}'
    for n2 in model_names:
        r = np.corrcoef(first_pcs[n1], first_pcs[n2])[0, 1]
        row += f'{r:>10.2f}'
    print(row)

print('\nAll done.')

import numpy as np
import os
from sklearn.cross_decomposition import CCA
from itertools import combinations

OUTDIR = os.path.join(REPO_ROOT, 'reproduced_outputs', 'feature_extraction')

# load all feature matrices
feature_files = {
    'Gabor':           'Gabor.npy',
    'Color':           'Color.npy',
    'AlexNetMid':      'AlexNetMid.npy',
    'AlexNetHigh':     'AlexNetHigh.npy',
    'CORnetV4':        'CORnetV4.npy',
    'CORnetIT':        'CORnetIT.npy',
    'CLIP':            'CLIP.npy',
    'Palatability':    'Palatability.npy',
    'Calorie':         'Calorie.npy',
    'Health':          'Health.npy',
    'Familiarity':     'Familiarity.npy',
    'CalorieObj':      'CalorieObjective.npy',
    'SavorySweet':     'SavorySweet.npy',
}

features = {}
for name, fname in feature_files.items():
    arr = np.load(os.path.join(OUTDIR, fname))
    features[name] = arr
    print(f'{name:20s}: {arr.shape}')

# ── 1. Pearson r between all pairs (full matrix, not just first PC) ──────────
# For multi-dim features: use RV coefficient (generalisation of r² to matrices)
def rv_coefficient(X, Y):
    """
    RV coefficient between two (n, p) and (n, q) matrices.
    Measures similarity of covariance structures.
    Range [0, 1]. 1 = identical structure.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XXT  = X @ X.T
    YYT  = Y @ Y.T
    num  = np.trace(XXT @ YYT)
    den  = np.sqrt(np.trace(XXT @ XXT) * np.trace(YYT @ YYT))
    return num / (den + 1e-12)

names = list(features.keys())
n     = len(names)
RV    = np.zeros((n, n))

for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        RV[i, j] = rv_coefficient(features[n1], features[n2])

import pandas as pd
rv_df = pd.DataFrame(RV, index=names, columns=names)
print('\nRV coefficient matrix (0=orthogonal, 1=identical structure):')
print(rv_df.round(3).to_string())

# ── 2. Flag pairs above threshold ────────────────────────────────────────────
threshold = 0.5
print(f'\nPairs with RV > {threshold}:')
for i in range(n):
    for j in range(i+1, n):
        if RV[i, j] > threshold:
            print(f'  {names[i]:20s} ↔ {names[j]:20s}  RV={RV[i,j]:.3f}')

# ── 3. Variance inflation factor for the 1D behavioral features ───────────────
# For the 1D features specifically, check VIF
from sklearn.linear_model import LinearRegression

behavioral_1d = ['Palatability', 'Calorie', 'Health', 'Familiarity',
                 'CalorieObj', 'SavorySweet']

print('\nVIF for 1D behavioral/categorical features:')
B = np.hstack([features[k] for k in behavioral_1d])   # (96, 6)
for i, name in enumerate(behavioral_1d):
    y    = B[:, i]
    Xoth = np.delete(B, i, axis=1)
    r2   = LinearRegression().fit(Xoth, y).score(Xoth, y)
    vif  = 1 / (1 - r2 + 1e-12)
    flag = '  <-- HIGH' if vif > 5 else ''
    print(f'  {name:20s}: VIF = {vif:.2f}{flag}')

# =============================================================================
# [11] CLIP_full512 — HISTORICAL LATER EXTRACTION STEP
# =============================================================================
#
# This block reproduces the operation executed in
# predclip_axis_characterization_v2.py when CLIP_full512.npy was absent.
# It is appended here only so this single release script creates the complete
# feature-matrix set used by the project. The numerical operations are unchanged.

print("\n[11] CLIP_full512 historical extraction")

import clip as openai_clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_STIM = N
img_paths = [os.path.join(IMGDIR, f"{cond}.jpg") for cond in CONDS]
clip512_path = os.path.join(OUTDIR, "CLIP_full512.npy")

print(f"\nCLIP_full512.npy not found at:\n  {clip512_path}")
print("Re-extracting CLIP_full512 from images using canonical STIM_CSV order.")

clip_model, clip_preprocess = openai_clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

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

print("\nComplete historical feature-matrix extraction finished.")
print(f"Outputs: {OUTDIR}")
