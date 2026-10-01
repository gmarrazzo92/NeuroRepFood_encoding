#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run surface-map generation in historical order.

This launcher contains no scientific calculations.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

STEPS = [
    ("07a", "Generate group Workbench dscalars", "07a_generate_group_workbench_maps.py"),
    ("07b", "Apply historical broad visual display mask to r_joint maps",
     "07b_apply_broad_visual_display_mask_rjoint.py"),
    ("07c", "Apply historical broad visual display mask to gain maps",
     "07c_apply_broad_visual_display_mask_gains.py"),
]

print("=" * 88)
print("SURFACE-MAP GENERATION")
print("=" * 88)
print("Python:", sys.executable)

for code, label, filename in STEPS:
    print("\n" + "=" * 88)
    print(f"{code} — {label}")
    print("=" * 88)
    subprocess.run([sys.executable, str(SCRIPT_DIR / filename)], check=True)

print("\n" + "=" * 88)
print("SURFACE-MAP GENERATION COMPLETED")
print("=" * 88)
