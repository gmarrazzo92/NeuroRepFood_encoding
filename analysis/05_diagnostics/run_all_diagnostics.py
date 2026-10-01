#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run diagnostic/characterization analyses in dependency-safe order.

This launcher contains no scientific analysis. Each step is executed as a
separate Python process.

Default:
    python analysis/05_diagnostics/run_all_diagnostics.py

Resume:
    python analysis/05_diagnostics/run_all_diagnostics.py --start-at 05c

Partial run:
    python analysis/05_diagnostics/run_all_diagnostics.py --stop-after 05b
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

STEPS = [
    ("05a", "Perceived-calorie prediction", "05a_perceived_calorie_prediction.py"),
    ("05b", "PredCLIP axis characterization", "05b_predclip_axis_characterization.py"),
    ("05c", "CLIP layer RSA", "05c_clip_layer_rsa.py"),
    ("05d", "ResCLIP reliability/recovery", "05d_resclip_reliability_and_recovery.py"),
    ("05e", "Feature-space RV overlap", "05e_feature_overlap_rv.py"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run diagnostic analyses in historical dependency order."
    )
    choices = [x[0] for x in STEPS]
    parser.add_argument("--start-at", choices=choices, default=choices[0])
    parser.add_argument("--stop-after", choices=choices, default=choices[-1])
    return parser.parse_args()


def main():
    args = parse_args()
    codes = [x[0] for x in STEPS]
    i0 = codes.index(args.start_at)
    i1 = codes.index(args.stop_after)

    if i1 < i0:
        raise SystemExit("--stop-after must not precede --start-at")

    selected = STEPS[i0:i1 + 1]

    print("=" * 88)
    print("DIAGNOSTIC / CHARACTERIZATION PIPELINE")
    print("=" * 88)
    print("Python:", sys.executable)
    print("Steps:")
    for code, label, _ in selected:
        print(f"  {code}: {label}")

    for code, label, filename in selected:
        script = SCRIPT_DIR / filename
        if not script.is_file():
            raise FileNotFoundError(script)

        print("\n" + "=" * 88)
        print(f"{code} — {label}")
        print("=" * 88)
        subprocess.run([sys.executable, str(script)], check=True)

    print("\n" + "=" * 88)
    print("ALL REQUESTED DIAGNOSTIC STEPS COMPLETED")
    print("=" * 88)


if __name__ == "__main__":
    main()
