#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Top-level launcher for the encoding-analysis reproducibility repository.

The launcher contains no scientific calculations. It only calls the individual
stage scripts in the documented order using the current Python interpreter.

From the repository root, run:

    python run_pipeline.py --help
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

ANALYSIS_STEPS = {
    1: (
        "Feature extraction",
        Path("analysis/01_feature_extraction/extract_features.py"),
    ),
    2: (
        "Feature-band construction",
        Path("analysis/02_feature_bands/build_feature_bands.py"),
    ),
    3: (
        "Encoding-model fitting",
        Path("analysis/03_encoding_models/fit_encoding_models.py"),
    ),
    4: (
        "ROI inference and noise ceilings",
        Path("analysis/04_roi_inference/run_roi_inference.py"),
    ),
    5: (
        "Diagnostics and characterization",
        Path("analysis/05_diagnostics/run_all_diagnostics.py"),
    ),
    6: (
        "Permutation robustness controls",
        Path("analysis/06_robustness/run_permutation_controls.py"),
    ),
    7: (
        "Workbench surface-map generation",
        Path("analysis/07_surface_maps/run_surface_maps.py"),
    ),
}

DOWNLOADER = Path("data_download/download_public_data.py")
FIGURES_SNAPSHOT = Path("figure_reproduction/recreate_manuscript_outputs.py")
FIGURES_RERUN = Path(
    "figure_reproduction/recreate_manuscript_outputs_from_rerun.py"
)
REFRESH_SNAPSHOT = Path("refresh_historical_outputs_from_rerun.py")

FIG3_INPUTS = [
    Path("figure_reproduction/static_inputs/figure3/ROI.png"),
    Path("figure_reproduction/static_inputs/figure3/M0_r_joint_uncorr.png"),
    Path("figure_reproduction/static_inputs/figure3/M2_r_joint_uncorr.png"),
    Path("figure_reproduction/static_inputs/figure3/M2-M0_r_joint_uncorr.png"),
]


def _display_cmd(cmd):
    return " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd)


def _require_file(relpath: Path):
    path = ROOT / relpath
    if not path.is_file():
        raise SystemExit(f"Required script not found: {path}")
    return path


def run_script(relpath: Path, extra_args=None, dry_run=False):
    script = _require_file(relpath)
    cmd = [sys.executable, str(script)]
    if extra_args:
        cmd.extend(str(x) for x in extra_args)

    print("\n" + "=" * 96, flush=True)
    print(f"RUN: {relpath.as_posix()}", flush=True)
    print("=" * 96, flush=True)
    print(_display_cmd(cmd), flush=True)

    if dry_run:
        return

    subprocess.run(cmd, cwd=ROOT, check=True)


def cmd_download(args):
    extra = []
    if args.downloader_dry_run:
        extra.append("--dry-run")
    run_script(DOWNLOADER, extra, dry_run=args.dry_run)


def cmd_analysis(args):
    if not (1 <= args.start_step <= 7):
        raise SystemExit("--start-step must be between 1 and 7.")
    if not (1 <= args.stop_step <= 7):
        raise SystemExit("--stop-step must be between 1 and 7.")
    if args.start_step > args.stop_step:
        raise SystemExit("--start-step cannot be greater than --stop-step.")

    if args.download:
        run_script(DOWNLOADER, dry_run=args.dry_run)

    for step in range(args.start_step, args.stop_step + 1):
        label, relpath = ANALYSIS_STEPS[step]
        print(f"\n### STEP {step:02d}: {label}", flush=True)

        extra = []
        if step == 4:
            extra = ["--nc-mode", args.nc_mode]

        run_script(relpath, extra, dry_run=args.dry_run)

    print("\n" + "=" * 96)
    print("ANALYSIS PIPELINE COMPLETED")
    print("=" * 96)
    if args.stop_step >= 7:
        print(
            "\nStep 07 has generated the numerical Workbench surface maps.\n"
            "Before regenerating Figure 3 from a fresh rerun, render panels B-D "
            "in wb_view and replace the three PNGs under:\n"
            "  figure_reproduction/static_inputs/figure3/\n"
            "Panel A (ROI.png) is unchanged.\n"
            "\nThen run:\n"
            "  python run_pipeline.py figures-from-rerun"
        )


def _figure_args(args):
    extra = []
    if getattr(args, "output_root", None):
        extra += ["--output-root", str(Path(args.output_root))]
    if getattr(args, "figures_only", False):
        extra.append("--figures-only")
    if getattr(args, "tables_only", False):
        extra.append("--tables-only")
    if getattr(args, "no_clean", False):
        extra.append("--no-clean")
    return extra


def cmd_figures(args):
    run_script(
        FIGURES_SNAPSHOT,
        _figure_args(args),
        dry_run=args.dry_run,
    )


def cmd_figures_from_rerun(args):
    run_script(
        FIGURES_RERUN,
        _figure_args(args),
        dry_run=args.dry_run,
    )


def cmd_freeze_snapshot(args):
    extra = []
    if args.snapshot_dry_run:
        extra.append("--dry-run")
    elif not args.confirm:
        raise SystemExit(
            "Snapshot refresh replaces managed historical_outputs folders.\n"
            "Inspect first with:\n"
            "  python run_pipeline.py freeze-snapshot --snapshot-dry-run\n"
            "Then rerun with --confirm."
        )

    if args.no_backup:
        extra.append("--no-backup")

    run_script(REFRESH_SNAPSHOT, extra, dry_run=args.dry_run)



def _exists(relpath):
    return (ROOT / relpath).exists()


def cmd_status(_args):
    print("=" * 96)
    print("REPOSITORY STATUS")
    print("=" * 96)

    print("\nCore scripts:")
    for step, (label, relpath) in ANALYSIS_STEPS.items():
        mark = "OK" if _exists(relpath) else "MISSING"
        print(f"  [{mark:7s}] Step {step:02d} — {label}: {relpath}")

    for label, relpath in [
        ("Public-data downloader", DOWNLOADER),
        ("Fast manuscript reproduction", FIGURES_SNAPSHOT),
        ("Rerun manuscript reproduction", FIGURES_RERUN),
        ("Snapshot refresh tool", REFRESH_SNAPSHOT),
    ]:
        mark = "OK" if _exists(relpath) else "MISSING"
        print(f"  [{mark:7s}] {label}: {relpath}")

    print("\nMain data/output locations:")
    for label, relpath in [
        ("Downloaded GLMsingle data", Path("data/glmsingle")),
        ("Frozen canonical snapshot", Path("historical_outputs")),
        ("Feature extraction", Path("reproduced_outputs/feature_extraction")),
        ("Feature bands", Path("reproduced_outputs/feature_bands")),
        ("Encoding models", Path("reproduced_outputs/encoding_models")),
        ("ROI inference", Path("reproduced_outputs/roi_inference")),
        ("Diagnostics", Path("reproduced_outputs/diagnostics")),
        ("Robustness", Path("reproduced_outputs/robustness/permutation_null")),
        ("Surface maps", Path("reproduced_outputs/surface_maps/workbench_dscalars")),
        ("Manuscript outputs", Path("reproduced_outputs/manuscript")),
    ]:
        mark = "PRESENT" if _exists(relpath) else "ABSENT"
        print(f"  [{mark:7s}] {label}: {relpath}")

    print("\nFigure 3 static inputs:")
    for relpath in FIG3_INPUTS:
        mark = "OK" if _exists(relpath) else "MISSING"
        print(f"  [{mark:7s}] {relpath}")

    print("\nSee README.md for the intended fast and full-rerun workflows.")


def add_figure_flags(parser):
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--figures-only", action="store_true")
    grp.add_argument("--tables-only", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Top-level launcher for data download, analysis steps 01-07, "
            "manuscript reproduction, and snapshot maintenance."
        )
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("status", help="Show repository/input/output status.")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("download", help="Download the minimal public dataset.")
    s.add_argument(
        "--downloader-dry-run",
        action="store_true",
        help="Call the downloader's own --dry-run preflight.",
    )
    s.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launcher command without executing it.",
    )
    s.set_defaults(func=cmd_download)

    s = sub.add_parser(
        "analysis",
        help="Run the canonical analysis pipeline (steps 01-07).",
    )
    s.add_argument("--start-step", type=int, default=1)
    s.add_argument("--stop-step", type=int, default=7)
    s.add_argument(
        "--nc-mode",
        choices=["historical", "cached", "recompute"],
        default="historical",
        help=(
            "Noise-ceiling mode passed to Step 04. 'historical' is the exact "
            "manuscript-reproduction default; 'recompute' is the deepest "
            "public-data rerun."
        ),
    )
    s.add_argument(
        "--download",
        action="store_true",
        help="Run the public-data downloader before Step 01.",
    )
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_analysis)

    s = sub.add_parser(
        "figures",
        help=(
            "Fast reviewer-facing manuscript reproduction from the frozen "
            "canonical snapshot."
        ),
    )
    add_figure_flags(s)
    s.set_defaults(func=cmd_figures)

    s = sub.add_parser(
        "figures-from-rerun",
        help=(
            "Regenerate manuscript figures/tables directly from fresh outputs "
            "after Steps 01-07."
        ),
    )
    add_figure_flags(s)
    s.set_defaults(func=cmd_figures_from_rerun)

    s = sub.add_parser(
        "freeze-snapshot",
        help=(
            "Maintainer action: refresh historical_outputs from a validated "
            "full rerun."
        ),
    )
    s.add_argument(
        "--snapshot-dry-run",
        action="store_true",
        help="Show the refresh mapping without changing files.",
    )
    s.add_argument(
        "--confirm",
        action="store_true",
        help="Required for an actual snapshot refresh.",
    )
    s.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create the one-time original snapshot backup.",
    )
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_freeze_snapshot)


    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
