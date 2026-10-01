#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Manuscript figure/table reproduction from canonical regenerated outputs
=======================================================================

Recreate the manuscript-facing quantitative figures and supplementary-table
CSVs DIRECTLY from the regenerated outputs produced by analysis steps 01-07.

This script deliberately does NOT read ``historical_outputs``. The regenerated
analysis outputs are the source of truth for the revised manuscript/SI.

Default inputs
--------------
    <repo>/reproduced_outputs/
        feature_extraction/
        diagnostics/
        roi_inference/
        robustness/permutation_null/
        surface_maps/

Figure 3 static Workbench screenshot inputs
-------------------------------------------
    <repo>/figure_reproduction/static_inputs/figure3/
        ROI.png
        M0_r_joint_uncorr.png
        M2_r_joint_uncorr.png
        M2-M0_r_joint_uncorr.png

``ROI.png`` is the unchanged ROI panel. Panels B-D should be screenshots made
from the regenerated Step-07 Workbench maps using the same wb_view display
settings as the manuscript figure.

Default outputs
---------------
    <repo>/reproduced_outputs/manuscript/
        figures/
        tables/

Recreated figures
-----------------
- Figure 1D
- Figure 2
- Figure 3 (assembled from the four Workbench screenshots above)
- Supplementary Figures S1, S2, S3, S4, S6

Not recreated here
------------------
- Figure 1A-B: explanatory/design artwork
- Figure 1C: stimulus-image panel
- Supplementary Figure S5: stimulus montage

Tables exported as CSV
----------------------
- Supplementary Tables S1, S2, S3, S4, S4b, S5, S5b, S5c, S5d,
  S6, S7, S7b, S8, S8b, S9, S10, S11, S12, S13, S14.

Important source choices
------------------------
- S1-S3 and Supplementary Figure S1 use the regenerated perceived-calorie
  diagnostic.
- Figure 1D uses the regenerated PredCLIP characterization.
- Supplementary Figure S2 uses the regenerated CLIP-layer RSA.
- Figure 2, S3, S4, S4b-S7b and S10 use the regenerated ROI-inference output.
- S8/S8b use the regenerated permutation-control output.
- S9 uses the regenerated feature-overlap RV matrix from Step 05e.
- S11-S14 use the regenerated ResCLIP diagnostics.
- Figure 3 is assembled from the regenerated Step-07 map screenshots plus the
  unchanged ROI panel.

Typical use from repository root
--------------------------------
    python figure_reproduction/recreate_manuscript_outputs.py

By default the requested manuscript output subdirectories are cleared before
regeneration so stale historical products cannot remain. Use ``--no-clean`` to
disable this behavior.
"""

from __future__ import annotations

import argparse
import math
import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats
from PIL import Image



ROIS = ["EarlyVisual", "IntermediateVisual", "HighLevelVTC"]
ROI_LABELS = {
    "EarlyVisual": "Early visual",
    "IntermediateVisual": "Intermediate visual",
    "HighLevelVTC": "High-level VTC",
}
PREDICTORS = ["LowVis", "HighVis", "LowHigh", "CLIP", "LowHighCLIP"]
MODEL_SHORT = {
    "M0_visual": "M0",
    "M1_visual_rawcal": "M1",
    "M2_visual_predclip": "M2",
    "M3_visual_resclip": "M3",
    "M4_visual_pred_resclip": "M4",
}
DIAG_SHORT = {
    "D0_visual_noclip": "D0",
    "D1_visual_clipband": "D1",
    "D2_visual_clipband_predclip": "D2",
}
COMP_SHORT = {
    "M1_rawcal_gt_M0_visual": "M1>M0",
    "M2_predclip_gt_M0_visual": "M2>M0",
    "M3_resclip_gt_M0_visual": "M3>M0",
    "M4_pred_res_gt_M0_visual": "M4>M0",
    "M2_predclip_gt_M1_rawcal": "M2>M1",
    "M2_predclip_gt_M3_resclip": "M2>M3",
    "M4_pred_res_gt_M2_predclip": "M4>M2",
    "M4_pred_res_gt_M1_rawcal": "M4>M1",
}
DIAG_COMP_SHORT = {
    "D1_clipband_gt_D0_noclip": "D1>D0",
    "D2_predclip_gt_D1_clipband": "D2>D1",
    "D2_predclip_gt_D0_noclip": "D2>D0",
}

MODEL_COLORS = {
    "M0": "#6BA043",
    "M1": "#4C78A8",
    "M2": "#D65F5F",
    "D0": "#8f8f8f",
    "D1": "#59A14F",
    "D2": "#E15759",
}
BAND_COLORS = {
    "LowVis": "#4878CF",
    "HighVis": "#6ACC65",
    "CaloriePredCLIP": "#D65F5F",
    "CalorieResCLIP": "#8C564B",
}
RANDOM_STATE = 42
POINT_SIZE = 28
POINT_ALPHA = 0.70

REPO_ROOT = None
REPRO = None
OUT_ROOT = None
FIG_OUT = None
TAB_OUT = None
FIG3_INPUT = None


def fmt_decimal(x, digits=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        x = float(x)
    except Exception:
        return str(x)
    s = f"{x:.{digits}f}"
    if abs(x) < 1:
        if s.startswith("0"):
            s = s[1:]
        elif s.startswith("-0"):
            s = "-" + s[2:]
    return s


def fmt_p(x):
    try:
        x = float(x)
    except Exception:
        return ""
    if not np.isfinite(x):
        return ""
    if x < 0.001:
        return "< .001"
    return fmt_decimal(x, 3)


def fmt_ci(mean, lo, hi, digits=3):
    return f"{fmt_decimal(mean,digits)} [{fmt_decimal(lo,digits)}, {fmt_decimal(hi,digits)}]"


def fmt_mean_sd(mean, sd, digits=3):
    return f"{fmt_decimal(mean,digits)} ± {fmt_decimal(sd,digits)}"


def ci95_t(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    tcrit = float(stats.t.ppf(0.975, len(x)-1))
    return m - tcrit * se, m + tcrit * se


def save_fig(fig, name):
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    png = FIG_OUT / f"{name}.png"
    pdf = FIG_OUT / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved figure: {png}")
    return png


def stars(q):
    try:
        q = float(q)
    except Exception:
        return ""
    if not np.isfinite(q) or q >= .05:
        return ""
    if q < .001:
        return "***"
    if q < .01:
        return "**"
    return "*"


def deterministic_jitter(n, scale=.07):
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-scale, scale, n)


def jitter_positions(n, width=.08, seed=RANDOM_STATE):
    if n <= 1:
        return np.zeros(n)
    return np.random.default_rng(seed).uniform(-width, width, size=n)


def add_bracket(ax, x1, x2, y, text, h=.0015, fontsize=8):
    if not text:
        return
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=.8, color="black", clip_on=False)
    ax.text((x1+x2)/2, y+h, text, ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


def set_plot_defaults():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, label):
    ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=20,
            fontweight="bold", ha="left", va="top")


def noise_summary():
    p = REPRO / "roi_inference" / "noise_ceiling_roi_values.csv"
    d = pd.read_csv(p)
    rows = []
    for roi in ROIS:
        vals = d.loc[d.roi == roi, "nc_r"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        lo, hi = ci95_t(vals)
        rows.append({"roi": roi, "mean": float(np.mean(vals)), "ci_low": lo, "ci_high": hi, "n": len(vals)})
    return pd.DataFrame(rows)


def draw_noise(ax, xcenter, width, roi, nc, add_label=False):
    r = nc[nc.roi == roi]
    if len(r) != 1:
        return
    row = r.iloc[0]
    x1, x2 = xcenter-width/2, xcenter+width/2
    ax.fill_between([x1,x2], [row.ci_low]*2, [row.ci_high]*2, color="0.86",
                    edgecolor="none", zorder=0, label="Noise ceiling" if add_label else None)
    ax.hlines(row["mean"], x1, x2, color="0.58", lw=1.0, linestyles="--", zorder=1)


# =============================================================================
# Figure recreation — historical plotting logic
# =============================================================================
# These functions deliberately reproduce the plotting logic of the executed
# historical scripts rather than redesigning the figures. Numerical inputs come
# from the canonical regenerated analysis outputs.

from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

ALPHA = 0.05
MAINFIG_USE_FDR_ONLY = True
SUPPFIG_USE_FDR_ONLY = True
DPI = 300
HIERARCHY_SPLIT_ROIS = ["EarlyVisual", "IntermediateVisual", "HighLevelVTC"]
NC_COLOR = "#333333"
NC_ALPHA = 0.13

# Exact labels/colors used by the final historical ROI plotting script.
ROI_LABELS_EXACT = {
    "EarlyVisual": "Early visual",
    "IntermediateVisual": "Intermediate visual",
    "HighLevelVTC": "HighLevelVTC",
}
MODELS_EXACT = {
    "M0_visual": {"label":"M0 Visual", "short_label":"Visual", "color":"#2CA02C"},
    "M1_visual_rawcal": {"label":"M1 +RawCal", "short_label":"+RawCal", "color":"#4C78A8"},
    "M2_visual_predclip": {"label":"M2 +PredCLIP", "short_label":"+PredCLIP", "color":"#D65F5F"},
    "M3_visual_resclip": {"label":"M3 +ResCLIP", "short_label":"+ResCLIP", "color":"#8C564B"},
    "M4_visual_pred_resclip": {"label":"M4 +Pred+Res", "short_label":"+Pred+Res", "color":"#E15759"},
}
DIAGNOSTIC_MODELS_EXACT = {
    "D0_visual_noclip": {"label":"D0 no CLIP", "short_label":"NoCLIP", "color":"#7F7F7F"},
    "D1_visual_clipband": {"label":"D1 CLIP band", "short_label":"+CLIPband", "color":"#9467BD"},
    "D2_visual_clipband_predclip": {"label":"D2 CLIP band + PredCLIP", "short_label":"+CLIPband+PredCLIP", "color":"#D65F5F"},
}
SPLIT_FEATURE_SPACES_EXACT = ["LowVis", "HighVis", "CaloriePredCLIP", "CalorieResCLIP"]
SPLIT_BAND_LABELS_EXACT = {
    "LowVis":"LowVis", "HighVis":"HighVis",
    "CaloriePredCLIP":"PredCLIP", "CalorieResCLIP":"ResCLIP",
}
SPLIT_BAND_COLORS_EXACT = {
    "LowVis":"#4878CF", "HighVis":"#6ACC65",
    "CaloriePredCLIP":"#D65F5F", "CalorieResCLIP":"#8C564B",
}
MAIN_GAIN_COMPARISON = "M2_predclip_gt_M0_visual"
M1_GAIN_COMPARISON = "M1_rawcal_gt_M0_visual"
MAIN_DIAGNOSTIC_COMPARISON = "D2_predclip_gt_D1_clipband"


def _save_historical_style(fig, name, dpi=300, facecolor=None):
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    png = FIG_OUT / f"{name}.png"
    pdf = FIG_OUT / f"{name}.pdf"
    kw = {"dpi": dpi, "bbox_inches": "tight"}
    if facecolor is not None:
        kw["facecolor"] = facecolor
    fig.savefig(png, **kw)
    # Save both raster and vector versions of each figure.
    kw_pdf = {"bbox_inches": "tight"}
    if facecolor is not None:
        kw_pdf["facecolor"] = facecolor
    fig.savefig(pdf, **kw_pdf)
    plt.close(fig)
    print(f"saved figure: {png}")
    return png


def _reset_mpl_defaults():
    plt.rcdefaults()


def set_plot_defaults():
    # Exact rcParams from the final historical ROI inference/plotting script.
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


def safe_float(x):
    try:
        x = float(x)
    except Exception:
        return np.nan
    return x if np.isfinite(x) else np.nan


def p_to_stars(p=None, q=None):
    val = q if q is not None and np.isfinite(q) else p
    if val is None or not np.isfinite(val): return ""
    if val < 0.001: return "***"
    if val < 0.01: return "**"
    if val < 0.05: return "*"
    return ""


def should_mark(row, use_fdr_only=True):
    if row is None: return False
    if use_fdr_only:
        q = safe_float(row.get("q_fdr_bh", np.nan))
        return np.isfinite(q) and q < ALPHA
    p = safe_float(row.get("p", np.nan))
    return np.isfinite(p) and p < ALPHA


def mark_text(row, use_fdr_only=True):
    if row is None: return ""
    if use_fdr_only:
        return p_to_stars(q=safe_float(row.get("q_fdr_bh", np.nan)))
    return p_to_stars(p=safe_float(row.get("p", np.nan)))


def roi_label(roi):
    return ROI_LABELS_EXACT.get(roi, roi)


def hierarchy_rois():
    return ["EarlyVisual", "IntermediateVisual", "HighLevelVTC"]


def panel_label(ax, label):
    ax.text(-0.18, 1.08, label, transform=ax.transAxes,
            fontsize=20, fontweight="bold", ha="left", va="top")


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0: return np.nan, np.nan, np.nan
    m = float(np.mean(values))
    lo, hi = ci95_t(values)
    return m, lo, hi


def nice_ylim(values, include_zero=True, pad_fraction=0.18, min_span=0.04):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        lo, hi = -min_span, min_span
    else:
        lo, hi = float(np.min(values)), float(np.max(values))
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = max(hi - lo, min_span)
    pad = pad_fraction * span
    return lo - pad, hi + pad


def draw_bracket(ax, x1, x2, y, h, txt):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y],
            color="black", linewidth=1.1, clip_on=False)
    ax.text((x1+x2)/2, y+h, txt, ha="center", va="bottom", fontsize=12)


def get_rjoint_row(df, roi, model):
    row = df[(df["roi"] == roi) & (df["model"] == model)]
    return None if len(row) != 1 else row.iloc[0].to_dict()


def get_comp_row(df, roi, comparison):
    row = df[(df["roi"] == roi) & (df["comparison"] == comparison)]
    return None if len(row) != 1 else row.iloc[0].to_dict()


def get_pair_row(df, comparison, pair):
    row = df[(df["comparison"] == comparison) & (df["pair"] == pair)]
    return None if len(row) != 1 else row.iloc[0].to_dict()


def get_model_values(df, roi, model):
    vals = df.loc[(df["roi"] == roi) & (df["model"] == model), "r_joint"].to_numpy(float)
    return vals[np.isfinite(vals)]


def get_gain_values(df, roi, comparison):
    vals = df.loc[(df["roi"] == roi) & (df["comparison"] == comparison), "delta_r"].to_numpy(float)
    return vals[np.isfinite(vals)]


def _nc_roi_from_csv(path):
    d = pd.read_csv(path)
    out = {}
    for roi in hierarchy_rois():
        vals = d.loc[d["roi"] == roi, "nc_r"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[roi] = None
            continue
        m = float(np.mean(vals)); s = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        lo, hi = ci95_t(vals)
        out[roi] = {"mean":m, "sem":s, "lo":float(lo), "hi":float(hi),
                    "ci95_lo":float(lo), "ci95_hi":float(hi), "n":int(vals.size)}
    return out


def draw_nc_band(ax, i, nc_entry, x_half_width=0.45, color=NC_COLOR, alpha=NC_ALPHA):
    if nc_entry is None: return
    lo, hi, mn = nc_entry["lo"], nc_entry["hi"], nc_entry["mean"]
    ax.fill_between([i-x_half_width, i+x_half_width], [lo,lo], [hi,hi],
                    color=color, alpha=alpha, linewidth=0, zorder=0)
    ax.hlines(mn, i-x_half_width, i+x_half_width,
              colors=color, linewidths=1.2, linestyles="--", zorder=1, alpha=0.6)


# -----------------------------------------------------------------------------
# Figure 1D — exact historical plotting code from predclip_axis_characterization
# -----------------------------------------------------------------------------
def figure_1d():
    _reset_mpl_defaults()
    p = REPRO / "diagnostics" / "predCLIP_axis_characterization_v2" / "single_text_probe_correlations.csv"
    probe_df = pd.read_csv(p).sort_values("r_predCLIP", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#c0392b" if r > 0 else "#2980b9" for r in probe_df["r_predCLIP"]]
    ax.barh(probe_df["probe"], probe_df["r_predCLIP"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r with PredCLIP score")
    ax.set_title("Single CLIP text-probe correlations with PredCLIP")
    ax.invert_yaxis()
    plt.tight_layout()
    return _save_historical_style(fig, "Figure_1D", dpi=220)


# -----------------------------------------------------------------------------
# Figure 2 — exact final historical ROI plotting logic
# -----------------------------------------------------------------------------
def figure_2():
    _reset_mpl_defaults(); set_plot_defaults()
    b = REPRO / "roi_inference"
    model_df = pd.read_csv(b / "roi_model_r_joint_subject_values.csv")
    rjoint_inf = pd.read_csv(b / "roi_model_r_joint_inference.csv")
    comp_df = pd.read_csv(b / "roi_model_comparisons.csv")
    gain_df = pd.read_csv(b / "hierarchy_model_gain_values.csv")
    pairwise_df = pd.read_csv(b / "hierarchy_model_gain_pairwise.csv")
    nc_roi = _nc_roi_from_csv(b / "noise_ceiling_roi_values.csv")

    rois = hierarchy_rois(); x = np.arange(len(rois))
    m0="M0_visual"; m1="M1_visual_rawcal"; m2="M2_visual_predclip"
    gain_comp=MAIN_GAIN_COMPARISON; gain_comp_m1=M1_GAIN_COMPARISON
    fig=plt.figure(figsize=(17.5,5.8),constrained_layout=True)
    gs=fig.add_gridspec(1,3,width_ratios=[1.0,1.55,1.15])
    axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[0,2])
    nc_handle=Patch(facecolor=NC_COLOR,alpha=NC_ALPHA*3,edgecolor=NC_COLOR,linewidth=0.5,label="Noise ceiling")

    panel_label(axA,"A")
    values_A=[]
    for roi in rois:
        values_A.extend(get_model_values(model_df,roi,m0)); nc=nc_roi.get(roi)
        if nc is not None: values_A.append(nc["hi"])
    yloA,yhiA=nice_ylim(values_A,include_zero=True,pad_fraction=0.22,min_span=0.05)
    for i,roi in enumerate(rois):
        draw_nc_band(axA,i,nc_roi.get(roi),x_half_width=0.40)
        vals=get_model_values(model_df,roi,m0); m,lo,hi=mean_ci(vals)
        axA.bar(i,m,width=0.60,color=MODELS_EXACT[m0]["color"],alpha=0.90,edgecolor="white",linewidth=0.8)
        axA.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt="none",color="black",capsize=3,linewidth=1.1)
        jit=jitter_positions(len(vals),width=0.09,seed=RANDOM_STATE+i)
        axA.scatter(np.full(len(vals),i)+jit,vals,s=POINT_SIZE,facecolors="white",edgecolors="0.25",linewidth=0.9,alpha=POINT_ALPHA,zorder=3)
        row=get_rjoint_row(rjoint_inf,roi,m0)
        if should_mark(row,MAINFIG_USE_FDR_ONLY):
            axA.text(i,hi+0.04*(yhiA-yloA),mark_text(row,MAINFIG_USE_FDR_ONLY),ha="center",va="bottom",fontsize=13)
    axA.axhline(0,color="black",linewidth=0.9); axA.set_xlim(-0.6,len(rois)-0.4); axA.set_ylim(yloA,yhiA)
    axA.set_xticks(x); axA.set_xticklabels([roi_label(r) for r in rois],rotation=20,ha="right")
    axA.set_ylabel(r"Mean held-out $r_{joint}$"); axA.set_title("Visual baseline encoding"); axA.grid(axis="y",color="0.90",linewidth=0.8)
    if any(nc_roi.get(r) is not None for r in rois): axA.legend(handles=[nc_handle],frameon=False,loc="upper left",fontsize=9)

    panel_label(axB,"B")
    width=0.22; off0=-0.25; off1=0.00; off2=+0.25
    values_B=[]
    for roi in rois:
        for mk in [m0,m1,m2]: values_B.extend(get_model_values(model_df,roi,mk))
        nc=nc_roi.get(roi)
        if nc is not None: values_B.append(nc["hi"])
    yloB,yhiB=nice_ylim(values_B,include_zero=True,pad_fraction=0.30,min_span=0.06)
    for i,roi in enumerate(rois):
        draw_nc_band(axB,i,nc_roi.get(roi),x_half_width=0.45); max_hi=-np.inf
        for mk,off in [(m0,off0),(m1,off1),(m2,off2)]:
            vals=get_model_values(model_df,roi,mk); m,lo,hi=mean_ci(vals); xpos=i+off
            axB.bar(xpos,m,width=width,color=MODELS_EXACT[mk]["color"],alpha=0.90,edgecolor="white",linewidth=0.8,label=MODELS_EXACT[mk]["label"] if i==0 else None)
            axB.errorbar(xpos,m,yerr=[[m-lo],[hi-m]],fmt="none",color="black",capsize=3,linewidth=1.1)
            jit=jitter_positions(len(vals),width=0.038,seed=RANDOM_STATE+10*i+[m0,m1,m2].index(mk))
            axB.scatter(np.full(len(vals),xpos)+jit,vals,s=POINT_SIZE-4,facecolors="white",edgecolors="0.25",linewidth=0.7,alpha=POINT_ALPHA,zorder=3)
            row=get_rjoint_row(rjoint_inf,roi,mk)
            if should_mark(row,MAINFIG_USE_FDR_ONLY):
                axB.text(xpos,hi+0.035*(yhiB-yloB),mark_text(row,MAINFIG_USE_FDR_ONLY),ha="center",va="bottom",fontsize=11)
            max_hi=max(max_hi,hi)
        crow_m1=get_comp_row(comp_df,roi,gain_comp_m1)
        if should_mark(crow_m1,MAINFIG_USE_FDR_ONLY):
            y=max_hi+0.08*(yhiB-yloB); h=0.020*(yhiB-yloB); draw_bracket(axB,i+off0,i+off1,y,h,mark_text(crow_m1,MAINFIG_USE_FDR_ONLY))
        crow_m2=get_comp_row(comp_df,roi,gain_comp)
        if should_mark(crow_m2,MAINFIG_USE_FDR_ONLY):
            y=max_hi+0.16*(yhiB-yloB); h=0.020*(yhiB-yloB); draw_bracket(axB,i+off0,i+off2,y,h,mark_text(crow_m2,MAINFIG_USE_FDR_ONLY))
    axB.axhline(0,color="black",linewidth=0.9); axB.set_xlim(-0.6,len(rois)-0.4); axB.set_ylim(yloB,yhiB)
    axB.set_xticks(x); axB.set_xticklabels([roi_label(r) for r in rois],rotation=20,ha="right")
    axB.set_ylabel(r"Mean held-out $r_{joint}$"); axB.set_title("Calorie-related models improve visual encoding"); axB.grid(axis="y",color="0.90",linewidth=0.8)
    handles,labels=axB.get_legend_handles_labels()
    if any(nc_roi.get(r) is not None for r in rois): handles.append(nc_handle); labels.append("Noise ceiling")
    axB.legend(handles,labels,frameon=False,loc="upper left",fontsize=9)

    panel_label(axC,"C")
    values_C=[]
    for roi in rois:
        values_C.extend(get_gain_values(gain_df,roi,gain_comp)); values_C.extend(get_gain_values(gain_df,roi,gain_comp_m1))
    yloC,yhiC=nice_ylim(values_C,include_zero=True,pad_fraction=0.28,min_span=0.025)
    meansC_m2=[]; meansC_m1=[]; hisC=[]
    for i,roi in enumerate(rois):
        vals_m2=get_gain_values(gain_df,roi,gain_comp); m2_,lo,hi=mean_ci(vals_m2); meansC_m2.append(m2_); hisC.append(hi)
        axC.bar(i,m2_,width=0.58,color=MODELS_EXACT[m2]["color"],alpha=0.85,edgecolor="white",linewidth=0.8,label="M2−M0" if i==0 else None)
        axC.errorbar(i,m2_,yerr=[[m2_-lo],[hi-m2_]],fmt="none",color="black",capsize=3,linewidth=1.1)
        jit=jitter_positions(len(vals_m2),width=0.09,seed=RANDOM_STATE+100+i)
        axC.scatter(np.full(len(vals_m2),i)+jit,vals_m2,s=POINT_SIZE,facecolors="white",edgecolors="0.25",linewidth=0.8,alpha=POINT_ALPHA,zorder=3)
        crow=get_comp_row(comp_df,roi,gain_comp)
        if should_mark(crow,MAINFIG_USE_FDR_ONLY): axC.text(i,hi+0.04*(yhiC-yloC),mark_text(crow,MAINFIG_USE_FDR_ONLY),ha="center",va="bottom",fontsize=13)
        vals_m1=get_gain_values(gain_df,roi,gain_comp_m1); meansC_m1.append(float(np.nanmean(vals_m1)) if vals_m1.size else np.nan)
    axC.plot(x,meansC_m2,color="#9D3F3F",linewidth=2.2,alpha=0.95,zorder=4)
    valid_m1=[(i,v) for i,v in enumerate(meansC_m1) if np.isfinite(v)]
    if valid_m1:
        axC.plot([v[0] for v in valid_m1],[v[1] for v in valid_m1],color=MODELS_EXACT[m1]["color"],linewidth=2.2,linestyle="--",alpha=0.85,zorder=5,label="M1−M0")
    pair_row=get_pair_row(pairwise_df,gain_comp,"HighLevelVTC_gt_Early")
    if should_mark(pair_row,MAINFIG_USE_FDR_ONLY):
        y=max(hisC[0],hisC[2])+0.10*(yhiC-yloC); h=0.025*(yhiC-yloC); draw_bracket(axC,0,2,y,h,mark_text(pair_row,MAINFIG_USE_FDR_ONLY))
    axC.axhline(0,color="black",linewidth=0.9); axC.set_xlim(-0.6,len(rois)-0.4); axC.set_ylim(yloC,yhiC)
    axC.set_xticks(x); axC.set_xticklabels([roi_label(r) for r in rois],rotation=20,ha="right")
    axC.set_ylabel(r"Model gain, $\Delta r_{joint}$"); axC.set_title("Calorie gain increases along the visual hierarchy"); axC.grid(axis="y",color="0.90",linewidth=0.8); axC.legend(frameon=False,loc="upper left",fontsize=9)
    return _save_historical_style(fig,"Figure_2",dpi=300)


# -----------------------------------------------------------------------------
# Supplementary Figure S1 — exact historical diagnostic plotting code
# -----------------------------------------------------------------------------
def supplementary_s1():
    _reset_mpl_defaults()
    b=REPRO/"diagnostics"/"perceived_calorie_prediction_diagnostics"
    summary_df=pd.read_csv(b/"calorie_prediction_cv_summary.csv")
    sub_df=pd.read_csv(b/"calorie_prediction_subject_specific_cv_summary.csv")
    rdm_df=pd.read_csv(b/"calorie_rdm_diagnostics.csv")
    calorie_group_z=np.load(b/"calorie_group_raw_z.npy")
    best_name=summary_df.sort_values("cv_r2",ascending=False).iloc[0]["predictor"]
    best_pred=np.load(b/f"calorie_pred_cv_{best_name}.npy")
    best_res=np.load(b/f"calorie_res_cv_{best_name}.npy")
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    ax=axes[0,0]; x=np.arange(len(summary_df)); ax.bar(x,summary_df["cv_r2"].values); ax.axhline(0,color="black",linewidth=0.8); ax.set_xticks(x); ax.set_xticklabels(summary_df["predictor"],rotation=30,ha="right"); ax.set_ylabel("CV R²"); ax.set_title("Perceived calorie prediction\nfrom feature spaces")
    ax=axes[0,1]; ax.scatter(calorie_group_z,best_pred,s=35,alpha=0.8); r=stats.pearsonr(calorie_group_z,best_pred)[0]; rho=stats.spearmanr(calorie_group_z,best_pred).correlation; ax.set_xlabel("Raw perceived calorie, z"); ax.set_ylabel(f"Predicted calorie from {best_name}"); ax.set_title(f"Best predictor: {best_name}\nr={r:.3f}, rho={rho:.3f}")
    ax=axes[0,2]; ax.scatter(calorie_group_z,best_res,s=35,alpha=0.8); r=stats.pearsonr(calorie_group_z,best_res)[0]; ax.set_xlabel("Raw perceived calorie, z"); ax.set_ylabel("Residual calorie"); ax.set_title(f"Residual from {best_name}\nr(raw,res)={r:.3f}")
    ax=axes[1,0]; ax.hist(best_res,bins=20,alpha=0.85); ax.axvline(0,color="black",linestyle="--",linewidth=1); ax.set_xlabel("Residual calorie"); ax.set_ylabel("Images"); ax.set_title("Residual calorie distribution")
    ax=axes[1,1]; pred_order=summary_df["predictor"].tolist(); data=[sub_df.loc[sub_df["predictor"]==p,"cv_r2"].values for p in pred_order]
    try:
        ax.boxplot(data,tick_labels=pred_order,showfliers=False)
    except TypeError:  # Matplotlib < 3.9
        ax.boxplot(data,labels=pred_order,showfliers=False)
    ax.axhline(0,color="black",linewidth=0.8); ax.set_xticklabels(pred_order,rotation=30,ha="right"); ax.set_ylabel("Subject-specific CV R²"); ax.set_title("Prediction of individual calorie ratings")
    ax=axes[1,2]; plot_cols=["corr_RDM_rawCalorie_predictorFeatures","corr_RDM_resCalorie_predictorFeatures"]; M=rdm_df.set_index("predictor")[plot_cols]; im=ax.imshow(M.values,vmin=-1,vmax=1,aspect="auto"); ax.set_xticks(np.arange(len(plot_cols))); ax.set_xticklabels(["Raw vs features","Residual vs features"],rotation=30,ha="right"); ax.set_yticks(np.arange(len(M.index))); ax.set_yticklabels(M.index); ax.set_title("RDM correlations"); plt.colorbar(im,ax=ax,fraction=0.046)
    plt.tight_layout()
    return _save_historical_style(fig,"Supplementary_Figure_S1",dpi=250)


# -----------------------------------------------------------------------------
# Supplementary Figure S2 — exact historical CLIP-layer figure reconstruction
# from the stored results, RDMs and max-stat null distribution.
# -----------------------------------------------------------------------------
def clean_layer_label(name):
    if name=="layer_00_embedding": return "Emb."
    if name=="layer_13_projected_embedding": return "Proj."
    if "transformer" in name: return name.replace("layer_","L").replace("_transformer","")
    return name


def _plot_rdm(ax,D,title,vmin=None,vmax=None,cmap="viridis"):
    im=ax.imshow(D,interpolation="nearest",cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_title(title,fontsize=10,pad=6); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for spine in ax.spines.values(): spine.set_linewidth(0.8); spine.set_color("black")
    return im


def _add_colorbar_to_axis(ax,im,label=None):
    divider=make_axes_locatable(ax); cax=divider.append_axes("right",size="4%",pad=0.04); cb=plt.colorbar(im,cax=cax); cb.ax.tick_params(labelsize=7,length=2)
    if label is not None: cb.set_label(label,fontsize=8)
    return cb


def _upper_tri_vec(D):
    iu=np.triu_indices_from(D,k=1); return np.asarray(D)[iu]


def supplementary_s2():
    _reset_mpl_defaults()
    b=REPRO/"diagnostics"/"clip_openai_layerwise_processing_rsa"
    results=pd.read_csv(b/"clip_openai_layerwise_processing_rsa_results.csv")
    d=results[(results["model_rdm_type"]=="ordinal")&(results["token_mode"]=="cls")].sort_values("layer_index_in_analysis").reset_index(drop=True)
    if d.empty: raise RuntimeError("Ordinal CLS CLIP-layer RSA results not found.")
    layer_names=d["layer_name"].tolist(); method=str(d["rsa_method"].iloc[0]); obs=d["rsa_r"].to_numpy(float); q_fdr=d["q_fdr_across_layers"].to_numpy(float); p_max=d["perm_p_maxstat_fwe"].to_numpy(float)
    max_null=np.load(b/"permutation_nulls"/"max_null_ordinal_cls_spearman.npy")
    model_D=np.load(b/"rdms"/"model_RDM_processing_ordinal.npy")
    labels=pd.read_csv(b/"processing_preparation_labels.csv")
    processing_level=labels["processing_level_0_3"].to_numpy(); sort_idx=np.lexsort((np.arange(len(labels)),processing_level)); model_D_sorted=model_D[np.ix_(sort_idx,sort_idx)]
    reps=[x for x in ["layer_01_transformer","layer_04_transformer","layer_08_transformer","layer_11_transformer","layer_13_projected_embedding"] if x in layer_names]
    finite_obs=np.isfinite(obs); peak_i=np.where(finite_obs)[0][np.nanargmax(obs[finite_obs])]; peak_layer=layer_names[peak_i]; peak_r=float(obs[peak_i]); peak_q=float(q_fdr[peak_i]); peak_pmax=float(p_max[peak_i])
    fig=plt.figure(figsize=(15,8.2),dpi=300)
    gs=GridSpec(nrows=2,ncols=7,figure=fig,height_ratios=[1.00,1.20],width_ratios=[1.05,1,1,1,1,1,1.45],hspace=0.45,wspace=0.45)
    ax_model=fig.add_subplot(gs[0,0]); im=_plot_rdm(ax_model,model_D_sorted,"Processing/preparation\nordinal model RDM",vmin=0,vmax=1,cmap="Greys"); _add_colorbar_to_axis(ax_model,im,label="dissimilarity"); ax_model.text(-0.25,1.12,"A",transform=ax_model.transAxes,fontsize=18,fontweight="bold",va="top",ha="left")
    D_rep_sorted={}; rdm_min=np.inf; rdm_max=-np.inf
    for lname in reps:
        D=np.load(b/"rdms"/f"RDM_OpenAI_CLIP_cls_{lname}.npy"); D=D[np.ix_(sort_idx,sort_idx)]; D_rep_sorted[lname]=D; vals=_upper_tri_vec(D); rdm_min=min(rdm_min,float(np.nanmin(vals))); rdm_max=max(rdm_max,float(np.nanmax(vals)))
    for ri,lname in enumerate(reps):
        ax=fig.add_subplot(gs[0,ri+1]); li=layer_names.index(lname); r=obs[li]; q=q_fdr[li]; pm=p_max[li]; sig_marker=" †" if np.isfinite(pm) and pm<0.05 else (" *" if np.isfinite(q) and q<0.05 else "")
        im=_plot_rdm(ax,D_rep_sorted[lname],f"CLIP {clean_layer_label(lname)}\n{method} r = {r:+.2f}{sig_marker}",vmin=rdm_min,vmax=rdm_max,cmap="viridis")
        if ri==len(reps)-1: _add_colorbar_to_axis(ax,im,label="1 − r")
        if ri==0: ax.text(-0.25,1.12,"B",transform=ax.transAxes,fontsize=18,fontweight="bold",va="top",ha="left")
    ax_line=fig.add_subplot(gs[1,1:6]); x=np.arange(len(layer_names)); y=obs; ax_line.axhline(0,color="black",lw=0.8,alpha=0.8); ax_line.plot(x,y,marker="o",lw=2.4,color="#2B5A9E",markersize=5,zorder=3)
    sig_fdr=np.isfinite(q_fdr)&(q_fdr<0.05); ax_line.scatter(x[sig_fdr],y[sig_fdr],s=82,facecolor="white",edgecolor="#2B5A9E",linewidth=1.8,zorder=4,label="FDR q < .05")
    sig_max=np.isfinite(p_max)&(p_max<0.05); ax_line.scatter(x[sig_max],y[sig_max],s=46,facecolor="#2B5A9E",edgecolor="#2B5A9E",linewidth=1.0,zorder=5,label="max-stat p < .05")
    ax_line.set_xticks(x); ax_line.set_xticklabels([clean_layer_label(l) for l in layer_names],rotation=45,ha="right",fontsize=8); ax_line.set_ylabel(f"RSA correlation with\nordinal processing RDM ({method} r)",fontsize=10); ax_line.set_xlabel("CLIP vision layer",fontsize=10); ax_line.set_title("Layer-wise processing/preparation structure",fontsize=11,pad=8); ax_line.spines["top"].set_visible(False); ax_line.spines["right"].set_visible(False); ax_line.tick_params(axis="both",labelsize=8); ax_line.legend(frameon=False,fontsize=8,loc="best"); ax_line.text(-0.10,1.12,"C",transform=ax_line.transAxes,fontsize=18,fontweight="bold",va="top",ha="left")
    ax_null=fig.add_subplot(gs[1,6]); ax_null.hist(max_null,bins=45,color="0.75",edgecolor="white",linewidth=0.5); ax_null.axvline(peak_r,color="#C94A2A",lw=2.7,label=f"observed peak r = {peak_r:+.2f}"); ax_null.axvline(np.nanmean(max_null),color="black",lw=1.2,linestyle="--",label="max-null mean"); ax_null.set_title(f"Peak layer: {clean_layer_label(peak_layer)}\nq = {peak_q:.4f}, pmax = {peak_pmax:.4f}",fontsize=10,pad=8); ax_null.set_xlabel("Max null RSA correlation",fontsize=9); ax_null.set_ylabel("Count",fontsize=9); ax_null.spines["top"].set_visible(False); ax_null.spines["right"].set_visible(False); ax_null.tick_params(axis="both",labelsize=8); ax_null.legend(frameon=False,fontsize=7,loc="best"); ax_null.text(-0.23,1.12,"D",transform=ax_null.transAxes,fontsize=18,fontweight="bold",va="top",ha="left")
    fig.suptitle("Food processing structure across CLIP vision layers (CLS token, ordinal processing/preparation RDM)",fontsize=15,fontweight="bold",y=0.985)
    fig.text(0.5,0.012,(f"RDMs sorted by processing/preparation level. CLIP RDMs use correlation distance. RSA method: {method}. * = FDR q < .05; † = max-stat familywise p < .05. For CLS-token analyses, the image-independent embedding-stage CLS token is omitted."),ha="center",va="bottom",fontsize=8,color="0.25")
    return _save_historical_style(fig,"Supplementary_Figure_S2",dpi=300,facecolor="white")


# -----------------------------------------------------------------------------
# Supplementary Figures S3 and S4 — exact final historical plotting logic
# -----------------------------------------------------------------------------
def supplementary_s3_s4():
    _reset_mpl_defaults(); set_plot_defaults()
    b=REPRO/"roi_inference"
    split_df=pd.read_csv(b/"split_model_roi_subject_values.csv")
    split_band_df=pd.read_csv(b/"split_model_band_inference.csv")
    split_contrast_df=pd.read_csv(b/"split_model_contrast_inference.csv")
    relative_df=pd.read_csv(b/"split_model_hierarchy_relative_importance.csv")
    rois=hierarchy_rois(); x=np.arange(len(rois)); pred_metric="r_split_CaloriePredCLIP"; res_metric="r_split_CalorieResCLIP"

    fig,ax=plt.subplots(figsize=(11.5,5.4),constrained_layout=True); width=0.28; off_pred=-0.17; off_res=+0.17
    allvals=[]; allvals.extend(split_df[pred_metric].to_numpy(float)); allvals.extend(split_df[res_metric].to_numpy(float)); ylo,yhi=nice_ylim(allvals,include_zero=True,pad_fraction=0.30,min_span=0.03)
    for i,roi in enumerate(rois):
        max_hi=-np.inf
        for metric,off,key in [(pred_metric,off_pred,"CaloriePredCLIP"),(res_metric,off_res,"CalorieResCLIP")]:
            vals=split_df.loc[split_df["roi"]==roi,metric].to_numpy(float); vals=vals[np.isfinite(vals)]; m,lo,hi=mean_ci(vals); xpos=i+off
            ax.bar(xpos,m,width=width,color=SPLIT_BAND_COLORS_EXACT[key],alpha=0.88,edgecolor="white",linewidth=0.8,label=SPLIT_BAND_LABELS_EXACT[key] if i==0 else None)
            ax.errorbar(xpos,m,yerr=[[m-lo],[hi-m]],fmt="none",color="black",capsize=3,linewidth=1.1)
            jit=jitter_positions(len(vals),width=0.045,seed=RANDOM_STATE+200+10*i+(0 if key=="CaloriePredCLIP" else 1)); ax.scatter(np.full(len(vals),xpos)+jit,vals,s=POINT_SIZE-3,facecolors="white",edgecolors="0.25",linewidth=0.8,alpha=POINT_ALPHA,zorder=3)
            brow=split_band_df[(split_band_df["roi"]==roi)&(split_band_df["band"]==key)]; row=brow.iloc[0].to_dict() if len(brow)==1 else None
            if should_mark(row,SUPPFIG_USE_FDR_ONLY): ax.text(xpos,hi+0.04*(yhi-ylo),mark_text(row,SUPPFIG_USE_FDR_ONLY),ha="center",va="bottom",fontsize=12)
            max_hi=max(max_hi,hi)
        crow=split_contrast_df[(split_contrast_df["roi"]==roi)&(split_contrast_df["contrast"]=="PredCLIP_gt_ResCLIP")]; row=crow.iloc[0].to_dict() if len(crow)==1 else None
        if should_mark(row,SUPPFIG_USE_FDR_ONLY):
            y=max_hi+0.10*(yhi-ylo); h=0.025*(yhi-ylo); draw_bracket(ax,i+off_pred,i+off_res,y,h,mark_text(row,SUPPFIG_USE_FDR_ONLY))
    ax.axhline(0,color="black",linewidth=0.9); ax.set_xlim(-0.6,len(rois)-0.4); ax.set_ylim(ylo,yhi); ax.set_xticks(x); ax.set_xticklabels([roi_label(r) for r in rois],rotation=20,ha="right"); ax.set_ylabel(r"M4 split contribution, $r_{split}$"); ax.set_title("Supplementary: PredCLIP versus ResCLIP split contributions"); ax.grid(axis="y",color="0.90",linewidth=0.8); ax.legend(frameon=False,loc="upper left")
    p3=_save_historical_style(fig,"Supplementary_Figure_S3",dpi=300)

    def nice_ylim_from_values(values,pad=0.12,min_span=0.001,include_zero=True):
        values=np.asarray(values,float); values=values[np.isfinite(values)]
        if values.size==0: lo,hi=-min_span/2,min_span/2
        else: lo,hi=float(np.min(values)),float(np.max(values))
        if include_zero: lo,hi=min(lo,0.0),max(hi,0.0)
        span=max(hi-lo,min_span); return lo-pad*span,hi+pad*span
    def _panel_label(ax,label): ax.text(-0.14,1.04,label,transform=ax.transAxes,fontsize=17,fontweight="bold",ha="left",va="top",clip_on=False)
    def _fdr_stars(row):
        if row is None: return ""
        q=safe_float(row.get("q_fdr_bh",np.nan)); return "" if not np.isfinite(q) or q>=ALPHA else p_to_stars(q=q)
    def _draw_bracket(ax,x1,x2,y,h,text):
        ax.plot([x1,x1,x2,x2],[y,y+h,y+h,y],color="black",linewidth=1.0,clip_on=False); ax.text((x1+x2)/2,y+h,text,ha="center",va="bottom",fontsize=11,clip_on=False)
    rois=[r for r in HIERARCHY_SPLIT_ROIS if r in set(split_df["roi"])]; bands=list(SPLIT_FEATURE_SPACES_EXACT); x=np.arange(len(rois),dtype=float)
    fig=plt.figure(figsize=(15.8,6.6),constrained_layout=False); gs=fig.add_gridspec(1,2,width_ratios=[1.55,1.0],left=0.07,right=0.985,bottom=0.21,top=0.78,wspace=0.32); axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); fig.suptitle("M4 split contributions across the visual hierarchy",y=0.965,fontsize=14); _panel_label(axA,"A")
    offsets=np.linspace(-0.31,0.31,len(bands)); width=0.15; allvals=[]
    for roi in rois:
        for fs in bands: allvals.extend(split_df.loc[split_df["roi"]==roi,f"r_split_{fs}"].to_numpy(float))
    allvals=np.asarray(allvals,float); allvals=allvals[np.isfinite(allvals)]
    if allvals.size==0: yloA,yhiA=-0.01,0.01
    else: yloA,yhiA=nice_ylim_from_values(allvals,pad=0.28,min_span=0.035); yloA=min(yloA,-0.002); yhiA=max(yhiA,0.012)
    yrangeA=yhiA-yloA; annotation_ymax=yhiA; legend_handles=[]; legend_labels=[]
    for i,roi in enumerate(rois):
        roi_bar_hi=-np.inf
        for j,fs in enumerate(bands):
            vals=split_df.loc[split_df["roi"]==roi,f"r_split_{fs}"].to_numpy(float); vals=vals[np.isfinite(vals)]; m,lo,hi=mean_ci(vals); xpos=x[i]+offsets[j]
            bar=axA.bar(xpos,m,width=width,color=SPLIT_BAND_COLORS_EXACT[fs],alpha=0.90,edgecolor="white",linewidth=0.8,label=SPLIT_BAND_LABELS_EXACT.get(fs,fs),zorder=2)
            if i==0: legend_handles.append(bar[0]); legend_labels.append(SPLIT_BAND_LABELS_EXACT.get(fs,fs))
            if np.isfinite(m): axA.errorbar(xpos,m,yerr=[[m-lo],[hi-m]],fmt="none",color="black",capsize=3,linewidth=1.0,zorder=4)
            if vals.size>0:
                jit=jitter_positions(vals.size,width=0.035,seed=RANDOM_STATE+500+i*20+j); axA.scatter(np.full(vals.size,xpos)+jit,vals,s=14,facecolors="white",edgecolors="0.28",linewidth=0.7,alpha=0.72,zorder=5)
            brow=split_band_df[(split_band_df["roi"]==roi)&(split_band_df["band"]==fs)]; brow_dict=brow.iloc[0].to_dict() if len(brow)==1 else None; star=_fdr_stars(brow_dict)
            if star and np.isfinite(hi):
                star_y=hi+0.045*yrangeA; axA.text(xpos,star_y,star,ha="center",va="bottom",fontsize=10.5,clip_on=False); annotation_ymax=max(annotation_ymax,star_y+0.02*yrangeA)
            if np.isfinite(hi): roi_bar_hi=max(roi_bar_hi,hi)
        crow=split_contrast_df[(split_contrast_df["roi"]==roi)&(split_contrast_df["contrast"]=="PredCLIP_gt_ResCLIP")]; crow_dict=crow.iloc[0].to_dict() if len(crow)==1 else None; star=_fdr_stars(crow_dict)
        if star and np.isfinite(roi_bar_hi):
            pred_x=x[i]+offsets[bands.index("CaloriePredCLIP")]; res_x=x[i]+offsets[bands.index("CalorieResCLIP")]; bracket_y=roi_bar_hi+0.13*yrangeA; bracket_h=0.035*yrangeA; _draw_bracket(axA,pred_x,res_x,bracket_y,bracket_h,star); annotation_ymax=max(annotation_ymax,bracket_y+bracket_h+0.04*yrangeA)
    yhiA_final=max(yhiA,annotation_ymax+0.02*yrangeA); axA.axhline(0,color="black",linewidth=0.9,zorder=1); axA.set_xlim(-0.60,len(rois)-0.40); axA.set_ylim(yloA,yhiA_final); axA.set_xticks(x); axA.set_xticklabels([ROI_LABELS_EXACT.get(r,r) for r in rois],rotation=18,ha="right"); axA.set_ylabel(r"M4 split contribution, $r_{split}$"); axA.set_title("Absolute split contributions",pad=8); axA.grid(axis="y",color="0.90",linewidth=0.7,zorder=0)
    _panel_label(axB,"B"); bottoms=np.zeros(len(rois),float)
    for fs in bands:
        vals=[]
        for roi in rois:
            row=relative_df[(relative_df["roi"]==roi)&(relative_df["band"]==fs)]; vals.append(float(row["mean_share"].iloc[0]) if len(row)==1 else np.nan)
        vals=np.asarray(vals,float); axB.bar(x,vals,bottom=bottoms,width=0.62,color=SPLIT_BAND_COLORS_EXACT[fs],edgecolor="white",linewidth=0.8,alpha=0.92,zorder=2)
        for i,v in enumerate(vals):
            if np.isfinite(v) and v>=0.12:
                txt_color="white" if v>=0.18 else "black"; axB.text(x[i],bottoms[i]+v/2,f"{100*v:.0f}%",ha="center",va="center",fontsize=9,color=txt_color,clip_on=True,zorder=4)
        bottoms=bottoms+np.nan_to_num(vals,nan=0.0)
    axB.set_xlim(-0.60,len(rois)-0.40); axB.set_ylim(0,1.0); axB.set_xticks(x); axB.set_xticklabels([ROI_LABELS_EXACT.get(r,r) for r in rois],rotation=18,ha="right"); axB.set_yticks(np.linspace(0,1,6)); axB.set_yticklabels([f"{int(100*t)}%" for t in np.linspace(0,1,6)]); axB.set_ylabel("Share of positive split contribution"); axB.set_title("Relative importance",pad=8); axB.grid(axis="y",color="0.90",linewidth=0.7,zorder=0)
    fig.legend(legend_handles,legend_labels,loc="upper center",bbox_to_anchor=(0.50,0.895),ncol=len(bands),frameon=False,columnspacing=1.8,handlelength=1.6)
    fig.text(0.07,0.075,"Panel A: bars show group mean ± 95% CI; dots are subjects; stars/brackets use BH-FDR q<.05. Panel B is descriptive: negative split contributions are clipped to 0 before subject-wise normalization.",ha="left",va="center",fontsize=8.5,color="0.35")
    p4=_save_historical_style(fig,"Supplementary_Figure_S4",dpi=300)
    return p3,p4


# -----------------------------------------------------------------------------
# Supplementary Figure S6 — exact final historical diagnostic plotting logic
# -----------------------------------------------------------------------------
def supplementary_s6():
    _reset_mpl_defaults(); set_plot_defaults()
    b=REPRO/"roi_inference"
    diag_df=pd.read_csv(b/"clipband_diagnostic_roi_model_r_joint_subject_values.csv")
    diag_rjoint_inf=pd.read_csv(b/"clipband_diagnostic_roi_model_r_joint_inference.csv")
    diag_comp_df=pd.read_csv(b/"clipband_diagnostic_roi_model_comparisons.csv")
    diag_gain_df=pd.read_csv(b/"clipband_diagnostic_hierarchy_model_gain_values.csv")
    nc_path=b/"clipband_diagnostic_noise_ceiling_roi_values.csv"
    diag_nc_roi=_nc_roi_from_csv(nc_path if nc_path.is_file() else b/"noise_ceiling_roi_values.csv")
    rois=hierarchy_rois(); x=np.arange(len(rois)); d0="D0_visual_noclip"; d1="D1_visual_clipband"; d2="D2_visual_clipband_predclip"; gain_comp=MAIN_DIAGNOSTIC_COMPARISON
    fig=plt.figure(figsize=(13.5,5.4),constrained_layout=True); gs=fig.add_gridspec(1,2,width_ratios=[1.45,1.0]); axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); nc_handle=Patch(facecolor=NC_COLOR,alpha=NC_ALPHA*3,edgecolor=NC_COLOR,linewidth=0.5,label="Noise ceiling")
    panel_label(axA,"A"); models=[d0,d1,d2]; offsets=[-0.25,0.0,0.25]; width=0.22; values_A=[]
    for roi in rois:
        for mk in models: values_A.extend(get_model_values(diag_df,roi,mk))
        entry=diag_nc_roi.get(roi)
        if entry is not None: values_A.append(entry["hi"])
    yloA,yhiA=nice_ylim(values_A,include_zero=True,pad_fraction=0.26,min_span=0.06)
    for i,roi in enumerate(rois):
        entry=diag_nc_roi.get(roi); draw_nc_band(axA,i,entry,x_half_width=0.45); max_hi=-np.inf
        for mk,off in zip(models,offsets):
            vals=get_model_values(diag_df,roi,mk); m,lo,hi=mean_ci(vals); xpos=i+off
            axA.bar(xpos,m,width=width,color=DIAGNOSTIC_MODELS_EXACT[mk]["color"],alpha=0.90,edgecolor="white",linewidth=0.8,label=DIAGNOSTIC_MODELS_EXACT[mk]["label"] if i==0 else None); axA.errorbar(xpos,m,yerr=[[m-lo],[hi-m]],fmt="none",color="black",capsize=3,linewidth=1.1)
            jit=jitter_positions(len(vals),width=0.038,seed=RANDOM_STATE+700+10*i+models.index(mk)); axA.scatter(np.full(len(vals),xpos)+jit,vals,s=POINT_SIZE-4,facecolors="white",edgecolors="0.25",linewidth=0.7,alpha=POINT_ALPHA,zorder=3)
            row=get_rjoint_row(diag_rjoint_inf,roi,mk)
            if should_mark(row,SUPPFIG_USE_FDR_ONLY): axA.text(xpos,hi+0.035*(yhiA-yloA),mark_text(row,SUPPFIG_USE_FDR_ONLY),ha="center",va="bottom",fontsize=11)
            max_hi=max(max_hi,hi)
        row_d1=get_comp_row(diag_comp_df,roi,"D1_clipband_gt_D0_noclip")
        if should_mark(row_d1,SUPPFIG_USE_FDR_ONLY): draw_bracket(axA,i+offsets[0],i+offsets[1],max_hi+0.08*(yhiA-yloA),0.020*(yhiA-yloA),mark_text(row_d1,SUPPFIG_USE_FDR_ONLY))
        row_d2=get_comp_row(diag_comp_df,roi,"D2_predclip_gt_D1_clipband")
        if should_mark(row_d2,SUPPFIG_USE_FDR_ONLY): draw_bracket(axA,i+offsets[1],i+offsets[2],max_hi+0.16*(yhiA-yloA),0.020*(yhiA-yloA),mark_text(row_d2,SUPPFIG_USE_FDR_ONLY))
    axA.axhline(0,color="black",linewidth=0.9); axA.set_xlim(-0.6,len(rois)-0.4); axA.set_ylim(yloA,yhiA); axA.set_xticks(x); axA.set_xticklabels([roi_label(r) for r in rois],rotation=20,ha="right"); axA.set_ylabel(r"Mean held-out $r_{joint}$"); axA.set_title("CLIP-separated baseline diagnostic"); axA.grid(axis="y",color="0.90",linewidth=0.8); handles,labels=axA.get_legend_handles_labels(); handles.append(nc_handle); labels.append("Noise ceiling"); axA.legend(handles,labels,frameon=False,loc="upper left",fontsize=9)
    panel_label(axB,"B"); values_B=[]
    for roi in rois: values_B.extend(get_gain_values(diag_gain_df,roi,gain_comp))
    yloB,yhiB=nice_ylim(values_B,include_zero=True,pad_fraction=0.30,min_span=0.025)
    for i,roi in enumerate(rois):
        vals=get_gain_values(diag_gain_df,roi,gain_comp); m,lo,hi=mean_ci(vals); axB.bar(i,m,width=0.55,color=DIAGNOSTIC_MODELS_EXACT[d2]["color"],alpha=0.85,edgecolor="white",linewidth=0.8); axB.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt="none",color="black",capsize=3,linewidth=1.1); jit=jitter_positions(len(vals),width=0.09,seed=RANDOM_STATE+900+i); axB.scatter(np.full(len(vals),i)+jit,vals,s=POINT_SIZE,facecolors="white",edgecolors="0.25",linewidth=0.8,alpha=POINT_ALPHA,zorder=3); row=get_comp_row(diag_comp_df,roi,gain_comp)
        if should_mark(row,SUPPFIG_USE_FDR_ONLY): axB.text(i,hi+0.04*(yhiB-yloB),mark_text(row,SUPPFIG_USE_FDR_ONLY),ha="center",va="bottom",fontsize=13)
    axB.axhline(0,color="black",linewidth=0.9); axB.set_xlim(-0.6,len(rois)-0.4); axB.set_ylim(yloB,yhiB); axB.set_xticks(x); axB.set_xticklabels([roi_label(r) for r in rois],rotation=20,ha="right"); axB.set_ylabel(r"Diagnostic gain, $\Delta r_{joint}$"); axB.set_title("PredCLIP beyond CLIP-separated baseline"); axB.grid(axis="y",color="0.90",linewidth=0.8)
    return _save_historical_style(fig,"Supplementary_Figure_S6",dpi=300)


# -----------------------------------------------------------------------------
# Figure 3 — historical Workbench screenshot assembly
# -----------------------------------------------------------------------------
FIG3_FILES = {
    "A": {
        "filename": "ROI.png",
        "title": "Visual hierarchy ROI groups",
        "description": (
            "Grouped HCP-MMP visual hierarchy ROIs: EarlyVisual, "
            "IntermediateVisual, HighLevelVTC."
        ),
    },
    "B": {
        "filename": "M0_r_joint_uncorr.png",
        "title": "M0 visual baseline",
        "description": (
            "Descriptive group-mean held-out r_joint map for the visual "
            "baseline model M0."
        ),
    },
    "C": {
        "filename": "M2_r_joint_uncorr.png",
        "title": "M2 visual + PredCLIP",
        "description": (
            "Descriptive group-mean held-out r_joint map for M2: visual "
            "baseline plus PredCLIP."
        ),
    },
    "D": {
        "filename": "M2-M0_r_joint_uncorr.png",
        "title": r"$\Delta r$: M2 - M0",
        "description": "Descriptive group-mean model-gain map for M2 minus M0.",
    },
}

FIG3_DPI = 300
FIG3_FIGSIZE = (15.5, 13.5)
FIG3_PANEL_LABEL_SIZE = 22
FIG3_TITLE_SIZE = 13
FIG3_DO_CROP = False
FIG3_CROP_PIXELS = (0, 0, 0, 0)


def _sha256_file(path, chunk_size=1024 * 1024):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _metadata_path(path):
    """Return a release-safe repository-relative path for generated metadata."""
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except Exception:
        return path.name


def _figure3_file_info(path):
    path = Path(path)
    d = {"path": _metadata_path(path), "exists": path.is_file()}
    if path.is_file():
        st = path.stat()
        d.update({
            "size_bytes": int(st.st_size),
            "modified_time": datetime.fromtimestamp(st.st_mtime).isoformat(
                timespec="seconds"
            ),
            "sha256": _sha256_file(path),
        })
    return d


def _figure3_load_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    img = Image.open(path).convert("RGB")
    original_size = img.size
    if FIG3_DO_CROP:
        left, top, right, bottom = FIG3_CROP_PIXELS
        w, h = img.size
        if left + right >= w or top + bottom >= h:
            raise ValueError(
                f"Invalid Figure 3 crop {FIG3_CROP_PIXELS} for {img.size}: {path}"
            )
        img = img.crop((left, top, w-right, h-bottom))
    return img, original_size, img.size


def _figure3_add_panel(ax, img, letter, title):
    ax.imshow(img)
    ax.set_axis_off()
    ax.text(
        0.005, 0.985, letter,
        transform=ax.transAxes,
        fontsize=FIG3_PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
            boxstyle="square,pad=0.15",
        ),
    )
    ax.set_title(title, fontsize=FIG3_TITLE_SIZE, fontweight="bold", pad=6)


def figure_3():
    """
    Assemble the four Workbench screenshot panels using the exact historical
    layout. No numerical/statistical operation is performed here.
    """
    _reset_mpl_defaults()
    images = {}
    input_records = {}

    for panel, spec in FIG3_FILES.items():
        path = FIG3_INPUT / spec["filename"]
        img, original_size, final_size = _figure3_load_image(path)
        images[panel] = img
        rec = _figure3_file_info(path)
        rec.update({
            "panel": panel,
            "title": spec["title"],
            "description": spec["description"],
            "original_image_size_pixels": original_size,
            "final_image_size_pixels_after_crop": final_size,
        })
        input_records[panel] = rec
        print(
            f"Figure 3 {panel}: loaded {path} | "
            f"original={original_size}, final={final_size}"
        )

    fig, axes = plt.subplots(
        4, 1,
        figsize=FIG3_FIGSIZE,
        dpi=FIG3_DPI,
        constrained_layout=False,
    )
    for ax, panel in zip(axes, ["A", "B", "C", "D"]):
        _figure3_add_panel(
            ax, images[panel], panel, FIG3_FILES[panel]["title"]
        )

    plt.subplots_adjust(
        left=0.015, right=0.99, top=0.985, bottom=0.02, hspace=0.10
    )

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    png = FIG_OUT / "Figure_3.png"
    pdf = FIG_OUT / "Figure_3.pdf"
    meta = FIG_OUT / "Figure_3_metadata.json"

    fig.savefig(png, dpi=FIG3_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, dpi=FIG3_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "purpose": (
            "Assemble the unchanged ROI panel and regenerated Workbench "
            "screenshots into manuscript Figure 3."
        ),
        "created": datetime.now().isoformat(timespec="seconds"),
        "layout": {
            "rows": 4,
            "columns": 1,
            "figsize_inches": FIG3_FIGSIZE,
            "dpi": FIG3_DPI,
            "crop": FIG3_DO_CROP,
            "crop_pixels_left_top_right_bottom": FIG3_CROP_PIXELS,
        },
        "panel_mapping": {
            panel: {
                "title": spec["title"],
                "description": spec["description"],
                "filename": spec["filename"],
            }
            for panel, spec in FIG3_FILES.items()
        },
        "input_files": input_records,
        "step07_map_provenance": {
            "B": (
                "reproduced_outputs/surface_maps/workbench_dscalars/"
                "wb_group_mean_r_joint_by_model__HighLevelVTCmasked.dscalar.nii "
                ":: mean_r_joint__M0_visual"
            ),
            "C": (
                "reproduced_outputs/surface_maps/workbench_dscalars/"
                "wb_group_mean_r_joint_by_model__HighLevelVTCmasked.dscalar.nii "
                ":: mean_r_joint__M2_visual_predclip"
            ),
            "D": (
                "reproduced_outputs/surface_maps/workbench_dscalars/"
                "wb_group_mean_model_gains_delta_r__HighLevelVTCmasked.dscalar.nii "
                ":: mean_delta_r__M2_predclip_gt_M0_visual"
            ),
            "display_mask_note": (
                "The legacy __HighLevelVTCmasked filenames use the historical "
                "broad visual-cortex display mask; inferential HighLevelVTC "
                "remains FFC+VVC+TE1p+TE2p."
            ),
        },
        "outputs": {
            "png": _figure3_file_info(png),
            "pdf": _figure3_file_info(pdf),
        },
        "note": (
            "Workbench screenshots are descriptive renderings. Formal "
            "statistical inference is ROI-level."
        ),
    }
    meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved figure: {png}")
    print(f"saved figure: {pdf}")
    print(f"saved metadata: {meta}")
    return png


# =============================================================================
# Supplementary-table CSV recreation
# =============================================================================

def write_table(table_id, df):
    TAB_OUT.mkdir(parents=True, exist_ok=True)
    p=TAB_OUT/f"Supplementary_Table_{table_id}.csv"
    df.to_csv(p,index=False)
    print(f"saved table: {p}")
    return p


def table_s1():
    d=pd.read_csv(REPRO/"diagnostics"/"perceived_calorie_prediction_diagnostics"/"calorie_prediction_cv_summary.csv").set_index("predictor").loc[PREDICTORS].reset_index()
    return write_table("S1",pd.DataFrame({
        "Predictor":d.predictor,
        "Group CV R2":[fmt_decimal(x) for x in d.cv_r2],
        "Pearson r":[fmt_decimal(x) for x in d.pearson_r],
        "Spearman rho":[fmt_decimal(x) for x in d.spearman_rho],
        "Residual variance fraction":[fmt_decimal(x) for x in d.residual_var_fraction],
    }))


def table_s2():
    d=pd.read_csv(REPRO/"diagnostics"/"perceived_calorie_prediction_diagnostics"/"calorie_prediction_subject_specific_cv_summary.csv")
    rows=[]
    for p in PREDICTORS:
        g=d[d.predictor==p]
        rows.append({
            "Predictor":p,
            "CV R2, mean ± SD":fmt_mean_sd(g.cv_r2.mean(),g.cv_r2.std(ddof=1)),
            "Pearson r, mean ± SD":fmt_mean_sd(g.pearson_r.mean(),g.pearson_r.std(ddof=1)),
            "Spearman rho, mean ± SD":fmt_mean_sd(g.spearman_rho.mean(),g.spearman_rho.std(ddof=1)),
            "Residual variance fraction, mean ± SD":fmt_mean_sd(g.residual_var_fraction.mean(),g.residual_var_fraction.std(ddof=1)),
        })
    return write_table("S2",pd.DataFrame(rows))


def table_s3():
    d=pd.read_csv(REPRO/"diagnostics"/"perceived_calorie_prediction_diagnostics"/"calorie_rdm_diagnostics.csv").set_index("predictor").loc[PREDICTORS].reset_index()
    cols=[
        ("RawCal RDM vs features","corr_RDM_rawCalorie_predictorFeatures"),
        ("PredCal RDM vs features","corr_RDM_predCalorie_predictorFeatures"),
        ("ResCal RDM vs features","corr_RDM_resCalorie_predictorFeatures"),
        ("RawCal vs PredCal RDM","corr_RDM_rawCalorie_predCalorie"),
        ("RawCal vs ResCal RDM","corr_RDM_rawCalorie_resCalorie"),
        ("PredCal vs ResCal RDM","corr_RDM_predCalorie_resCalorie"),
    ]
    out={"Predictor":d.predictor}
    for lab,c in cols: out[lab]=[fmt_decimal(x) for x in d[c]]
    return write_table("S3",pd.DataFrame(out))


def _roi_inf(): return pd.read_csv(REPRO/"roi_inference"/"roi_model_r_joint_inference.csv")
def _roi_comp(): return pd.read_csv(REPRO/"roi_inference"/"roi_model_comparisons.csv")


def table_s4():
    d=_roi_inf(); rows=[]
    for roi in ROIS:
        for model in ["M0_visual","M1_visual_rawcal","M2_visual_predclip","M3_visual_resclip","M4_visual_pred_resclip"]:
            r=d[(d.roi==roi)&(d.model==model)].iloc[0]
            rows.append({"ROI":roi,"Model":MODEL_SHORT[model],"N":int(r.n_subjects),"Mean rjoint [95% CI]":fmt_ci(r.mean_r_joint,r.ci95_lo_r_joint,r.ci95_hi_r_joint),"p":fmt_p(r.p),"q":fmt_p(r.q_fdr_bh)})
    return write_table("S4",pd.DataFrame(rows))


def table_s4b():
    d=pd.read_csv(REPRO/"roi_inference"/"noise_ceiling_roi_values.csv"); rows=[]
    for roi in ROIS:
        v=d.loc[d.roi==roi,"nc_r"].to_numpy(float); lo,hi=ci95_t(v); rows.append({"ROI":roi,"N":len(v),"Noise ceiling r [95% CI]":fmt_ci(v.mean(),lo,hi)})
    return write_table("S4b",pd.DataFrame(rows))


def _comparison_rows(full=False):
    d=_roi_comp(); order=list(COMP_SHORT.keys()) if full else ["M1_rawcal_gt_M0_visual","M2_predclip_gt_M0_visual","M3_resclip_gt_M0_visual","M2_predclip_gt_M1_rawcal","M2_predclip_gt_M3_resclip"]
    rows=[]
    for roi in ROIS:
        for c in order:
            r=d[(d.roi==roi)&(d.comparison==c)].iloc[0]
            rows.append({"ROI":roi,"Contrast":COMP_SHORT[c].replace(">"," > "),"Delta rjoint [95% CI]":fmt_ci(r.mean_delta_r,r.ci95_lo_delta_r,r.ci95_hi_delta_r),"p":fmt_p(r.p),"q":fmt_p(r.q_fdr_bh)})
    return pd.DataFrame(rows)


def table_s5(): return write_table("S5",_comparison_rows(False))
def table_s5b(): return write_table("S5b",_comparison_rows(True))


def table_s5c():
    d=pd.read_csv(REPRO/"roi_inference"/"hierarchy_model_gain_slope_tests.csv"); rows=[]
    for c in ["M2_predclip_gt_M0_visual","M1_rawcal_gt_M0_visual"]:
        r=d[d.comparison==c].iloc[0]; rows.append({"Test":COMP_SHORT[c].replace(">","-")+" linear slope","N":int(r.n_subjects),"Estimate":f"{fmt_decimal(r.mean_slope_delta_r_per_step)} Delta rjoint per step","Cohen dz":fmt_decimal(r.cohens_dz_slope_z,2),"p":fmt_p(r.p_expected),"q":fmt_p(r.q_fdr_bh)})
    return write_table("S5c",pd.DataFrame(rows))


def table_s5d():
    d=pd.read_csv(REPRO/"roi_inference"/"hierarchy_model_gain_pairwise.csv"); d=d[d.comparison=="M2_predclip_gt_M0_visual"]; order=["Intermediate_gt_Early","HighLevelVTC_gt_Early","HighLevelVTC_gt_Intermediate"]; labels={"Intermediate_gt_Early":"IntermediateVisual > EarlyVisual","HighLevelVTC_gt_Early":"HighLevelVTC > EarlyVisual","HighLevelVTC_gt_Intermediate":"HighLevelVTC > IntermediateVisual"}; rows=[]
    for p in order:
        r=d[d.pair==p].iloc[0]; rows.append({"Contrast":labels[p],"Delta rjoint":fmt_decimal(r.mean_pair_delta_r_A_minus_B),"Cohen dz":fmt_decimal(r.cohens_dz,2),"p":fmt_p(r.p),"q":fmt_p(r.q_fdr_bh)})
    return write_table("S5d",pd.DataFrame(rows))


def table_s6():
    d=pd.read_csv(REPRO/"roi_inference"/"split_model_band_inference.csv"); rows=[]
    for roi in ROIS:
        for b in ["LowVis","HighVis","CaloriePredCLIP","CalorieResCLIP"]:
            r=d[(d.roi==roi)&(d.band==b)].iloc[0]; rows.append({"ROI":roi,"Band":{"CaloriePredCLIP":"PredCLIP","CalorieResCLIP":"ResCLIP"}.get(b,b),"rsplit [95% CI]":fmt_ci(r.mean_r_split,r.ci95_lo_r_split,r.ci95_hi_r_split),"p":fmt_p(r.p),"q":fmt_p(r.q_fdr_bh)})
    return write_table("S6",pd.DataFrame(rows))


def table_s7():
    d=pd.read_csv(REPRO/"roi_inference"/"split_model_hierarchy_relative_importance.csv"); rows=[]
    for roi in ROIS:
        row={"ROI":roi}
        for b,label in [("HighVis","HighVis"),("LowVis","LowVis"),("CaloriePredCLIP","PredCLIP"),("CalorieResCLIP","ResCLIP")]:
            v=float(d[(d.roi==roi)&(d.band==b)].mean_share.iloc[0]); row[label]=f"{100*v:.1f}%"
        rows.append(row)
    return write_table("S7",pd.DataFrame(rows))


def table_s7b():
    d=pd.read_csv(REPRO/"roi_inference"/"split_model_contrast_inference.csv"); rows=[]
    for roi in ROIS:
        r=d[d.roi==roi].iloc[0]; rows.append({"ROI":roi,"Contrast":"PredCLIP > ResCLIP","Delta rsplit [95% CI]":fmt_ci(r.mean_delta,r.ci95_lo_delta,r.ci95_hi_delta),"p":fmt_p(r.p),"q":fmt_p(r.q_fdr_bh)})
    return write_table("S7b",pd.DataFrame(rows))


def table_s8():
    d=pd.read_csv(REPRO/"robustness"/"permutation_null"/"permutation_null_summary.csv"); d=d[(d.perm_type=="full_shuffle")&d.quantity.isin(["M0","M2"])]; rows=[]
    for roi in ROIS:
        for q in ["M0","M2"]:
            r=d[(d.roi==roi)&(d.quantity==q)].iloc[0]; rows.append({"ROI":roi,"Quantity":q,"Null mean +/- SD":fmt_mean_sd(r.null_mean,r.null_std),"True mean":fmt_decimal(r.true_mean),"p(true > null)":fmt_p(r.true_gt_null_p)})
    return write_table("S8",pd.DataFrame(rows))


def table_s8b():
    d=pd.read_csv(REPRO/"robustness"/"permutation_null"/"calorie_specificity_summary.csv"); rows=[]
    for roi in ROIS:
        r=d[d.roi==roi].iloc[0]; rows.append({"ROI":roi,"N":int(r.n_subjects),"Mean specificity [95% CI]":fmt_ci(r.mean_specificity,r.ci95_lo,r.ci95_hi,4),"Cohen dz":fmt_decimal(r.cohens_dz,2),"% > 0":f"{100*r.frac_gt0:.1f}","p":fmt_p(r.p_signflip),"q":fmt_p(r.q_fdr_bh)})
    return write_table("S8b",pd.DataFrame(rows))


def rv_coefficient(X,Y):
    X=np.asarray(X,float); Y=np.asarray(Y,float)
    if X.ndim==1: X=X[:,None]
    if Y.ndim==1: Y=Y[:,None]
    X=X-X.mean(axis=0); Y=Y-Y.mean(axis=0); XX=X@X.T; YY=Y@Y.T
    return float(np.trace(XX@YY)/(np.sqrt(np.trace(XX@XX)*np.trace(YY@YY))+1e-12))


def table_s9():
    p = (
        REPRO / "diagnostics" / "feature_overlap_rv"
        / "rv_coefficient_matrix_manuscript_S9.csv"
    )
    d = pd.read_csv(p, index_col=0)
    names = [
        "Gabor", "Color", "AlexNetMid", "AlexNetHigh", "CORnetIT",
        "CLIP", "Palatability", "Calorie", "Health", "Familiarity",
    ]
    d = d.loc[names, names]
    out = d.copy()
    out.insert(0, "Feature space", out.index)
    out = out.reset_index(drop=True)
    for c in names:
        out[c] = [fmt_decimal(x) for x in out[c]]
    return write_table("S9", out)


def table_s10():
    b=REPRO/"roi_inference"; inf=pd.read_csv(b/"clipband_diagnostic_roi_model_r_joint_inference.csv"); comp=pd.read_csv(b/"clipband_diagnostic_roi_model_comparisons.csv"); rows=[]
    for roi in ROIS:
        for cname,label in [("D1_clipband_gt_D0_noclip","+CLIPband > NoCLIP"),("D2_predclip_gt_D1_clipband","+PredCLIP > CLIPband"),("D2_predclip_gt_D0_noclip","+CLIPband+PredCLIP > NoCLIP")]:
            r=comp[(comp.roi==roi)&(comp.comparison==cname)].iloc[0]
            rows.append({"ROI":"Early visual" if roi=="EarlyVisual" else "Intermediate visual" if roi=="IntermediateVisual" else "HighLevelVTC","Comparison":label,"N":int(r.n_subjects),"Model A r":f"{r.mean_r_A:.4f}","Model B r":f"{r.mean_r_B:.4f}","Delta r":f"{r.mean_delta_r:.4f}","95% CI Delta r":f"[{r.ci95_lo_delta_r:.4f}, {r.ci95_hi_delta_r:.4f}]","p":fmt_p(r.p),"q FDR":fmt_p(r.q_fdr_bh)})
    return write_table("S10",pd.DataFrame(rows))


def table_s11():
    d=pd.read_csv(REPRO/"diagnostics"/"resclip_reliability_and_recovery_diagnostics"/"resclip_step0_reliability_summary.csv").iloc[0]
    rows=[("Split-half Spearman correlation",d.resclip_split_spearman_mean),("Spearman-Brown reliability",d.resclip_spearman_brown_mean),("Correlation-scale ceiling",d.resclip_corr_ceiling_mean),("R2 ceiling",d.resclip_r2_ceiling_mean)]
    return write_table("S11",pd.DataFrame({"Metric":[a for a,_ in rows],"Value":[fmt_decimal(v,4) for _,v in rows]}))


def table_s12():
    d=pd.read_csv(REPRO/"diagnostics"/"resclip_reliability_and_recovery_diagnostics"/"resclip_step1_clip_recovery_summary.csv"); order=["CLIP50_linear_ridge","CLIP50_rbf_kernel_ridge","CLIP512_linear_ridge","CLIP512_discarded_PC51plus_ridge"]; d=d.set_index("model").loc[order].reset_index(); rows=[]
    for _,r in d.iterrows(): rows.append({"Model":r.model,"CV R2":fmt_decimal(r.cv_r2,4),"Pearson r":fmt_decimal(r.pearson_r,4),"Spearman rho":fmt_decimal(r.spearman_rho,4),"pperm CV R2":fmt_p(r.p_perm_cv_r2),"pperm Spearman":fmt_p(r.p_perm_spearman),"Positive CV R2 fraction of R2 ceiling":fmt_decimal(r.positive_cv_r2_fraction_of_r2_ceiling,4)})
    return write_table("S12",pd.DataFrame(rows))


def table_s13():
    d=pd.read_csv(REPRO/"diagnostics"/"resclip_reliability_and_recovery_diagnostics"/"resclip_step2A_item_anchor_correlations.csv").sort_values("abs_spearman_rho",ascending=False).head(20); rows=[]
    for _,r in d.iterrows(): rows.append({"Anchor":r.anchor,"Spearman rho":fmt_decimal(r.spearman_rho,4),"Pearson r":fmt_decimal(r.pearson_r,4),"pperm two-sided":fmt_decimal(r.p_perm_two_sided,4),"qFDR two-sided":fmt_decimal(r.q_fdr_bh_two_sided,4)})
    return write_table("S13",pd.DataFrame(rows))


def table_s14():
    d=pd.read_csv(REPRO/"diagnostics"/"resclip_reliability_and_recovery_diagnostics"/"resclip_step2B_rdm_anchor_correlations.csv"); order=["Calorie_mean","Color","Health_mean","CalorieObjective","Gabor","CORnet V4","SavorySweet","CORnet IT","Palatability_mean"]; rows=[]
    for a in order:
        r=d[d.anchor_rdm==a].iloc[0]; rows.append({"Anchor RDM":a,"Spearman rho":fmt_decimal(r.spearman_rho,4),"Pearson r":fmt_decimal(r.pearson_r,4),"pperm two-sided":fmt_decimal(r.p_perm_two_sided_itemlabel,4),"qFDR two-sided":fmt_decimal(r.q_fdr_bh_two_sided,4),"Direct R1 bridge":"1.0000" if bool(r.is_direct_r1_bridge) else "0.0000"})
    return write_table("S14",pd.DataFrame(rows))


def make_tables():
    funcs=[table_s1,table_s2,table_s3,table_s4,table_s4b,table_s5,table_s5b,table_s5c,table_s5d,table_s6,table_s7,table_s7b,table_s8,table_s8b,table_s9,table_s10,table_s11,table_s12,table_s13,table_s14]
    return [f() for f in funcs]


# =============================================================================
# Release entry point
# =============================================================================

# =============================================================================
# Canonical regenerated-output entry point
# =============================================================================

def resolve_repo_root(script_path: Path):
    """Expected location: <repo>/figure_reproduction/<this_script>."""
    candidate = script_path.resolve().parent.parent
    if (candidate / "reproduced_outputs").is_dir():
        return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "reproduced_outputs").is_dir():
        return cwd
    return candidate


def _required_inputs():
    """Files that must exist before manuscript regeneration starts."""
    roi = REPRO / "roi_inference"
    perc = (
        REPRO / "diagnostics"
        / "perceived_calorie_prediction_diagnostics"
    )
    pred = (
        REPRO / "diagnostics"
        / "predCLIP_axis_characterization_v2"
    )
    clip = (
        REPRO / "diagnostics"
        / "clip_openai_layerwise_processing_rsa"
    )
    res = (
        REPRO / "diagnostics"
        / "resclip_reliability_and_recovery_diagnostics"
    )
    perm = REPRO / "robustness" / "permutation_null"
    rv = REPRO / "diagnostics" / "feature_overlap_rv"

    req = [
        perc / "calorie_prediction_cv_summary.csv",
        perc / "calorie_prediction_subject_specific_cv_summary.csv",
        perc / "calorie_rdm_diagnostics.csv",
        pred / "single_text_probe_correlations.csv",
        clip / "clip_openai_layerwise_processing_rsa_results.csv",
        roi / "roi_model_r_joint_subject_values.csv",
        roi / "roi_model_r_joint_inference.csv",
        roi / "roi_model_comparisons.csv",
        roi / "hierarchy_model_gain_values.csv",
        roi / "hierarchy_model_gain_pairwise.csv",
        roi / "hierarchy_model_gain_slope_tests.csv",
        roi / "noise_ceiling_roi_values.csv",
        roi / "split_model_roi_subject_values.csv",
        roi / "split_model_band_inference.csv",
        roi / "split_model_contrast_inference.csv",
        roi / "split_model_hierarchy_relative_importance.csv",
        roi / "clipband_diagnostic_roi_model_r_joint_subject_values.csv",
        roi / "clipband_diagnostic_roi_model_r_joint_inference.csv",
        roi / "clipband_diagnostic_roi_model_comparisons.csv",
        roi / "clipband_diagnostic_hierarchy_model_gain_values.csv",
        perm / "permutation_null_summary.csv",
        perm / "calorie_specificity_summary.csv",
        rv / "rv_coefficient_matrix_manuscript_S9.csv",
        res / "resclip_step0_reliability_summary.csv",
        res / "resclip_step1_clip_recovery_summary.csv",
        res / "resclip_step2A_item_anchor_correlations.csv",
        res / "resclip_step2B_rdm_anchor_correlations.csv",
    ]
    req.extend(FIG3_INPUT / spec["filename"] for spec in FIG3_FILES.values())
    return req


def preflight(require_figure3=True):
    required = _required_inputs()
    if not require_figure3:
        figure3_paths = {
            FIG3_INPUT / spec["filename"] for spec in FIG3_FILES.values()
        }
        required = [p for p in required if p not in figure3_paths]

    missing = [p for p in required if not p.is_file()]
    if missing:
        lines = "\n".join(f"  MISSING: {p}" for p in missing)
        raise SystemExit(
            "Canonical manuscript regeneration preflight FAILED.\n"
            "The following required inputs are missing:\n" + lines
        )
    print(f"Preflight PASS: {len(required)} required inputs found.")


def clean_requested_outputs(figures=True, tables=True):
    """
    Remove manuscript products before regeneration so no stale historical
    figure/table remains mixed with the canonical rerun.
    """
    if figures and FIG_OUT.exists():
        shutil.rmtree(FIG_OUT)
    if tables and TAB_OUT.exists():
        shutil.rmtree(TAB_OUT)


def main():
    global REPO_ROOT, REPRO, OUT_ROOT, FIG_OUT, TAB_OUT, FIG3_INPUT

    parser = argparse.ArgumentParser(
        description=(
            "Recreate manuscript-facing figures and supplementary-table CSVs "
            "from canonical regenerated analysis outputs (steps 01-07)."
        )
    )
    default_repo = resolve_repo_root(Path(__file__))
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo,
        help="Repository root. Default: parent of figure_reproduction/.",
    )
    parser.add_argument(
        "--reproduced-root",
        type=Path,
        default=None,
        help="Override <repo>/reproduced_outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Override manuscript output directory. Default: "
            "<repo>/reproduced_outputs/manuscript."
        ),
    )
    parser.add_argument(
        "--figure3-input",
        type=Path,
        default=None,
        help=(
            "Override Figure 3 screenshot directory. Default: "
            "<repo>/figure_reproduction/static_inputs/figure3."
        ),
    )
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clear the requested manuscript output directories first.",
    )
    args = parser.parse_args()

    if args.figures_only and args.tables_only:
        raise SystemExit(
            "Choose at most one of --figures-only / --tables-only"
        )

    REPO_ROOT = args.repo_root.resolve()
    REPRO = (
        args.reproduced_root
        if args.reproduced_root is not None
        else REPO_ROOT / "reproduced_outputs"
    ).resolve()
    OUT_ROOT = (
        args.output_root
        if args.output_root is not None
        else REPRO / "manuscript"
    ).resolve()
    FIG_OUT = OUT_ROOT / "figures"
    TAB_OUT = OUT_ROOT / "tables"
    FIG3_INPUT = (
        args.figure3_input
        if args.figure3_input is not None
        else REPO_ROOT / "figure_reproduction" / "static_inputs" / "figure3"
    ).resolve()

    if not REPRO.is_dir():
        raise SystemExit(f"reproduced_outputs not found: {REPRO}")

    print("=" * 88)
    print("CANONICAL MANUSCRIPT OUTPUT REGENERATION")
    print("=" * 88)
    print(f"Repository root     : {REPO_ROOT}")
    print(f"Canonical inputs    : {REPRO}")
    print(f"Figure 3 screenshots: {FIG3_INPUT}")
    print(f"Manuscript outputs  : {OUT_ROOT}")

    make_figures = not args.tables_only
    make_tables_flag = not args.figures_only

    preflight(require_figure3=make_figures)

    if not args.no_clean:
        clean_requested_outputs(
            figures=make_figures,
            tables=make_tables_flag,
        )

    if make_figures:
        FIG_OUT.mkdir(parents=True, exist_ok=True)
        figure_1d()
        figure_2()
        figure_3()
        supplementary_s1()
        supplementary_s2()
        supplementary_s3_s4()
        supplementary_s6()

    if make_tables_flag:
        TAB_OUT.mkdir(parents=True, exist_ok=True)
        make_tables()

    print("\nDone.")
    if make_figures:
        print(f"Figures: {FIG_OUT}")
    if make_tables_flag:
        print(f"Tables : {TAB_OUT}")
    print(
        "\nThese products were generated from the canonical regenerated "
        "analysis outputs, not historical_outputs."
    )


if __name__ == "__main__":
    main()
