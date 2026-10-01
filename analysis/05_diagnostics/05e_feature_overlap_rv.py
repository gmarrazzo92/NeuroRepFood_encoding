#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature-space RV overlap diagnostic.

The RV coefficient implementation and feature ordering are copied from the
executed historical feature-extraction script. The historical script printed
the matrix but did not save it. This helper persists the same calculation.
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

FEAT_DIR = os.path.join(REPO_ROOT, "reproduced_outputs", "feature_extraction")
OUTDIR = os.path.join(
    REPO_ROOT, "reproduced_outputs", "diagnostics", "feature_overlap_rv"
)
os.makedirs(OUTDIR, exist_ok=True)

feature_files = {
    "Gabor":        "Gabor.npy",
    "Color":        "Color.npy",
    "AlexNetMid":   "AlexNetMid.npy",
    "AlexNetHigh":  "AlexNetHigh.npy",
    "CORnetV4":     "CORnetV4.npy",
    "CORnetIT":     "CORnetIT.npy",
    "CLIP":         "CLIP.npy",
    "Palatability": "Palatability.npy",
    "Calorie":      "Calorie.npy",
    "Health":       "Health.npy",
    "Familiarity":  "Familiarity.npy",
    "CalorieObj":   "CalorieObjective.npy",
    "SavorySweet":  "SavorySweet.npy",
}

features = {}
for name, fname in feature_files.items():
    path = os.path.join(FEAT_DIR, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    arr = np.load(path)
    features[name] = arr
    print(f"{name:20s}: {arr.shape}")


def rv_coefficient(X, Y):
    """
    RV coefficient between two (n, p) and (n, q) matrices.
    Measures similarity of covariance structures.
    Range [0, 1]. 1 = identical structure.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XXT = X @ X.T
    YYT = Y @ Y.T
    num = np.trace(XXT @ YYT)
    den = np.sqrt(np.trace(XXT @ XXT) * np.trace(YYT @ YYT))
    return num / (den + 1e-12)


names = list(features.keys())
n = len(names)
RV = np.zeros((n, n))

for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        RV[i, j] = rv_coefficient(features[n1], features[n2])

rv_df = pd.DataFrame(RV, index=names, columns=names)
rv_df.index.name = "Feature space"

full_path = os.path.join(OUTDIR, "rv_coefficient_matrix_full.csv")
rv_df.to_csv(full_path)

print("\nRV coefficient matrix (0=orthogonal, 1=identical structure):")
print(rv_df.round(3).to_string())

threshold = 0.5
print(f"\nPairs with RV > {threshold}:")
for i in range(n):
    for j in range(i + 1, n):
        if RV[i, j] > threshold:
            print(f"  {names[i]:20s} ↔ {names[j]:20s}  RV={RV[i,j]:.3f}")

manuscript_names = [
    "Gabor", "Color", "AlexNetMid", "AlexNetHigh", "CORnetIT",
    "CLIP", "Palatability", "Calorie", "Health", "Familiarity",
]
s9 = rv_df.loc[manuscript_names, manuscript_names].copy()
s9_path = os.path.join(OUTDIR, "rv_coefficient_matrix_manuscript_S9.csv")
s9.to_csv(s9_path)

print(f"\nSaved full matrix : {full_path}")
print(f"Saved S9 subset   : {s9_path}")
