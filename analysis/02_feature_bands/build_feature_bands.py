# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:26:34 2026

@author: G.Marrazzo
"""

"""
Band construction — NeuroRepFood encoding model
================================================
Builds the primary banded-ridge feature matrices and additional diagnostic
feature bands for baseline-robustness analyses.

Primary band structure:
  Band 1: LowVis        — Gabor + Color
                          kernel-normalised, concatenated, PCA 80D

  Band 2: HighVis       — AlexNetMid + AlexNetHigh + CORnetIT + CLIP
                          kernel-normalised, concatenated, PCA 80D

  Band 3: Calorie       — subjective perceived calorie rating (1D)

  Band 4: Palatability  — Palatability residualised against Familiarity
                          and Calorie (1D)

  Band 5: Nuisance      — Familiarity (1D)

Additional diagnostic bands:
  HighVisNoCLIP         — AlexNetMid + AlexNetHigh + CORnetIT
                          kernel-normalised, concatenated, PCA 80D

  CLIPown               — standalone CLIP.npy representation used to derive
                          CaloriePredCLIP, centred and scaled to unit kernel
                          trace, with no additional PCA

Purpose of diagnostic bands:
  These bands allow a baseline-robustness diagnostic in which CLIP is not
  jointly compressed inside the HighVis PCA pool, but instead receives its
  own feature band and its own ridge regularisation. This tests whether the
  original HighVis baseline underestimates HighLevelVTC prediction because
  CLIP-relevant semantic directions are diluted during joint PCA compression.

Outputs saved to FEAT_DIR/bands/:
  band_LowVis.npy           (96, 80)
  band_HighVis.npy          (96, 80)   primary HighVis, unchanged
  band_HighVisNoCLIP.npy    (96, 80)   diagnostic only
  band_CLIPown.npy          (96, 50)   diagnostic only
  band_Calorie.npy          (96, 1)
  band_Palatability.npy     (96, 1)
  band_Nuisance.npy         (96, 1)
  band_meta.npy             dict with diagnostics and provenance
"""

# =============================================================================
# [0] IMPORTS
# =============================================================================

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from itertools import combinations

print('Imports OK')


# =============================================================================
# [1] PATHS
# =============================================================================

# RELEASE PATH ADAPTATION ONLY.
# Numerical band-construction code below is unchanged from the executed
# historical script archived in Python(2).zip (12 June 2026 version).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

FEAT_DIR   = os.path.join(REPO_ROOT, 'reproduced_outputs', 'feature_extraction')
BAND_DIR   = os.path.join(REPO_ROOT, 'reproduced_outputs', 'feature_bands')
os.makedirs(BAND_DIR, exist_ok=True)

PCA_DIMS     = 80    # applied to both LowVis and HighVis
RANDOM_STATE = 42
N_STIM       = 96

# =============================================================================
# [2] HELPERS
# =============================================================================

def load(fname):
    arr = np.load(os.path.join(FEAT_DIR, fname)).astype(np.float32)
    assert arr.shape[0] == N_STIM, \
        f'{fname}: expected {N_STIM} rows, got {arr.shape[0]}'
    assert np.all(np.isfinite(arr)), \
        f'{fname}: contains non-finite values'
    return arr


def scale_to_unit_kernel(X):
    """
    Scale X so its linear kernel K = X X^T has unit trace.
    Ensures no single block dominates the concatenated PCA by amplitude.
    Works in float64 internally for numerical stability.
    """
    X  = X.astype(np.float64)
    X  = X - X.mean(axis=0)
    tr = np.trace(X @ X.T)
    if tr < 1e-12:
        raise ValueError('Kernel has near-zero trace — check feature matrix')
    return (X / np.sqrt(tr)).astype(np.float32)


def build_band(blocks, name, n_components=PCA_DIMS):
    """
    Kernel-normalise each block, concatenate, PCA to n_components.
    Returns (96, n_components) float32.
    """
    print(f'\n  Building {name}:')
    scaled = []
    for block_name, X in blocks:
        Xs = scale_to_unit_kernel(X)
        print(f'    {block_name:15s}: {X.shape} → kernel-normalised')
        scaled.append(Xs)

    concat = np.hstack(scaled).astype(np.float32)
    print(f'    Concatenated    : {concat.shape}')

    n_comp = min(n_components, concat.shape[0] - 1, concat.shape[1])
    pca    = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    result = pca.fit_transform(concat).astype(np.float32)
    var    = pca.explained_variance_ratio_.sum()

    print(f'    After PCA {n_comp}D   : {result.shape}  '
          f'var explained = {var:.3f}')
    return result, pca


def rv_coefficient(X, Y):
    """RV coefficient. Range [0,1]. 1 = identical covariance structure."""
    X = X.astype(np.float64) - X.mean(axis=0)
    Y = Y.astype(np.float64) - Y.mean(axis=0)
    XXT = X @ X.T
    YYT = Y @ Y.T
    num = np.trace(XXT @ YYT)
    den = np.sqrt(np.trace(XXT @ XXT) * np.trace(YYT @ YYT))
    return float(num / (den + 1e-12))


# =============================================================================
# [3] BAND 1 — LowVis
# =============================================================================

print('\n[3] Band 1 — LowVis (Gabor + Color → PCA 80D)')

gabor = load('Gabor.npy')   # (96, 384)
color = load('Color.npy')   # (96, 512)

band_lowvis, pca_lowvis = build_band(
    [('Gabor', gabor), ('Color', color)],
    name='LowVis'
)

np.save(os.path.join(BAND_DIR, 'band_LowVis.npy'), band_lowvis)
print(f'  band_LowVis.npy saved: {band_lowvis.shape}')


# =============================================================================
# [4] BAND 2 — HighVis
# =============================================================================

print('\n[4] Band 2 — HighVis (AlexNetMid + AlexNetHigh + CORnetIT + CLIP → PCA 80D)')

alexmid  = load('AlexNetMid.npy')    # (96, 50)
alexhigh = load('AlexNetHigh.npy')   # (96, 50)
cornet   = load('CORnetIT.npy')      # (96, 93)
clip_    = load('CLIP.npy')          # (96, 50)

band_highvis, pca_highvis = build_band(
    [('AlexNetMid',  alexmid),
     ('AlexNetHigh', alexhigh),
     ('CORnetIT',    cornet),
     ('CLIP',        clip_)],
    name='HighVis'
)

np.save(os.path.join(BAND_DIR, 'band_HighVis.npy'), band_highvis)
print(f'  band_HighVis.npy saved: {band_highvis.shape}')

# =============================================================================
# [4b] DIAGNOSTIC BANDS — HighVis without CLIP + CLIP as own band
# =============================================================================

print('\n[4b] Diagnostic bands — HighVisNoCLIP and CLIPown')

# Diagnostic baseline:
#   D1_visual_clipband = LowVis + HighVisNoCLIP + CLIPown
#
# This keeps the original HighVis band untouched, but creates a stricter
# diagnostic baseline in which CLIP is not jointly compressed with AlexNet/CORnet.
# CLIP will therefore receive its own band and its own ridge regularization
# in the model-fitting script.

band_highvis_noclip, pca_highvis_noclip = build_band(
    [('AlexNetMid',  alexmid),
     ('AlexNetHigh', alexhigh),
     ('CORnetIT',    cornet)],
    name='HighVisNoCLIP'
)

np.save(os.path.join(BAND_DIR, 'band_HighVisNoCLIP.npy'), band_highvis_noclip)
print(f'  band_HighVisNoCLIP.npy saved: {band_highvis_noclip.shape}')


# CLIPown:
# Use the same standalone CLIP.npy representation that was used to derive
# CaloriePredCLIP. Do not re-PCA it. The diagnostic asks whether giving CLIP
# its own band/regularization rescues the visual baseline, not whether another
# transformed CLIP space helps.

band_clipown = scale_to_unit_kernel(clip_)

np.save(os.path.join(BAND_DIR, 'band_CLIPown.npy'), band_clipown)
print(f'  band_CLIPown.npy saved: {band_clipown.shape}')
# =============================================================================
# [5] BAND 3 — Calorie
# =============================================================================

print('\n[5] Band 3 — Calorie')

calorie_raw = load('Calorie.npy')   # (96, 1)

# z-score for numerical consistency with other bands
band_calorie = ((calorie_raw - calorie_raw.mean())
                / calorie_raw.std()).astype(np.float32)

np.save(os.path.join(BAND_DIR, 'band_Calorie.npy'), band_calorie)
print(f'  band_Calorie.npy saved: {band_calorie.shape}  '
      f'mean={band_calorie.mean():.4f}  std={band_calorie.std():.4f}')


# =============================================================================
# [6] BAND 4 — Palatability (residualised against Familiarity + Calorie)
# =============================================================================

print('\n[6] Band 4 — Palatability | Familiarity, Calorie')

pal_raw = load('Palatability.npy').flatten()    # (96,)
fam_raw = load('Familiarity.npy').flatten()     # (96,)
cal_raw = load('Calorie.npy').flatten()         # (96,)

# partial out Familiarity and Calorie from Palatability
X_ctrl    = np.column_stack([fam_raw, cal_raw])
pal_resid = pal_raw - LinearRegression().fit(X_ctrl, pal_raw).predict(X_ctrl)
pal_resid = ((pal_resid - pal_resid.mean()) / pal_resid.std()).astype(np.float32)

band_palatability = pal_resid.reshape(-1, 1)
np.save(os.path.join(BAND_DIR, 'band_Palatability.npy'), band_palatability)

# residualisation diagnostics
print(f'  Correlations before / after residualisation:')
for name, arr in [('Familiarity', fam_raw), ('Calorie', cal_raw)]:
    r_before = np.corrcoef(pal_raw, arr)[0, 1]
    r_after  = np.corrcoef(pal_resid, arr)[0, 1]
    print(f'    Palatability ↔ {name:12s}: '
          f'r={r_before:+.3f} → {r_after:+.3f}')

r_orig = np.corrcoef(pal_resid, pal_raw)[0, 1]
print(f'    Palatability_resid ↔ Palatability_orig: r={r_orig:+.3f}  '
      f'(shared variance retained)')
print(f'  band_Palatability.npy saved: {band_palatability.shape}')


# =============================================================================
# [7] BAND 5 — Nuisance (Familiarity)
# =============================================================================

print('\n[7] Band 5 — Nuisance (Familiarity)')

fam_arr = load('Familiarity.npy')   # (96, 1)

band_nuisance = ((fam_arr - fam_arr.mean())
                 / fam_arr.std()).astype(np.float32)

np.save(os.path.join(BAND_DIR, 'band_Nuisance.npy'), band_nuisance)
print(f'  band_Nuisance.npy saved: {band_nuisance.shape}')


# =============================================================================
# [8] FINAL DIAGNOSTICS
# =============================================================================

print(f'\n{"="*60}')
print('Final band diagnostics')
print(f'{"="*60}')

bands = {
    'LowVis':          band_lowvis,
    'HighVis':         band_highvis,          # original primary HighVis
    'HighVisNoCLIP':   band_highvis_noclip,   # diagnostic only
    'CLIPown':         band_clipown,          # diagnostic only
    'Calorie':         band_calorie,
    'Palatability':    band_palatability,
    'Nuisance':        band_nuisance,
}

# shapes and basic stats
print('\nBand summary:')
total_dims = 0
for name, arr in bands.items():
    d = arr.shape[1] if arr.ndim > 1 else 1
    total_dims += d
    print(f'  {name:15s}: shape={arr.shape}  '
          f'mean={arr.mean():+.4f}  std={arr.std():.4f}')
print(f'\n  Total dims   : {total_dims}')
print(f'  n samples    : {N_STIM}')
print(f'  Note: banded ridge operates on {N_STIM}×{N_STIM} kernels — '
      f'feature dimensionality does not affect feasibility')

# RV matrix
print('\nRV coefficient matrix (target: all off-diagonal < 0.70):')
names = list(bands.keys())
print(f'  {"":15s}' + ''.join(f'{n:>15s}' for n in names))
for n1 in names:
    row = f'  {n1:15s}'
    for n2 in names:
        rv  = rv_coefficient(bands[n1], bands[n2])
        flag = ' !' if n1 != n2 and rv > 0.70 else '  '
        row += f'{rv:>13.3f}{flag}'
    print(row)

# flag any remaining high-RV pairs
print('\nPairs with RV > 0.50:')
any_high = False
for n1, n2 in combinations(names, 2):
    rv = rv_coefficient(bands[n1], bands[n2])
    if rv > 0.50:
        print(f'  {n1:15s} ↔ {n2:15s}  RV={rv:.3f}')
        any_high = True
if not any_high:
    print('  None — band structure is clean.')

# PCA variance explained
print('\nPCA diagnostics:')
print(f'  LowVis  PCA {pca_lowvis.n_components_}D: '
      f'var explained = {pca_lowvis.explained_variance_ratio_.sum():.3f}')
print(f'  HighVis PCA {pca_highvis.n_components_}D: '
      f'var explained = {pca_highvis.explained_variance_ratio_.sum():.3f}')
print(f'  HighVisNoCLIP PCA {pca_highvis_noclip.n_components_}D: '
      f'var explained = {pca_highvis_noclip.explained_variance_ratio_.sum():.3f}')


# save metadata
meta = {
    'band_names':   list(bands.keys()),
    'band_shapes':  {k: v.shape for k, v in bands.items()},
    'lowvis_var_explained':  float(pca_lowvis.explained_variance_ratio_.sum()),
    'highvis_var_explained': float(pca_highvis.explained_variance_ratio_.sum()),
    'highvis_noclip_var_explained': float(pca_highvis_noclip.explained_variance_ratio_.sum()),
    'pca_dims':     PCA_DIMS,
    'n_stim':       N_STIM,
    'palatability_residualised_against': ['Familiarity', 'Calorie'],

    # Important provenance note
    'diagnostic_bands_added': {
        'HighVisNoCLIP': 'AlexNetMid + AlexNetHigh + CORnetIT, kernel-normalised, concatenated, PCA 80D',
        'CLIPown': 'CLIP as its own diagnostic band, kernel-normalised, PCA 50D',
        'purpose': 'Baseline-robustness diagnostic testing whether original HighVis joint PCA compressed CLIP-relevant semantic directions.'
    }
}
np.save(os.path.join(BAND_DIR, 'band_meta.npy'), meta, allow_pickle=True)

print(f'\nAll bands saved to: {BAND_DIR}')
print('Done.')