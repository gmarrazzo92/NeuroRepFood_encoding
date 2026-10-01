# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:02:55 2026

@author: G.Marrazzo
"""

# -*- coding: utf-8 -*-
"""
Mask Workbench dscalar maps to VTC / HighLevelVTC for clean visualization.

Usage
-----
Set INPUT_DSCALAR to the file you want to mask.
Set OUTPUT_DSCALAR to the desired output file.
Choose whether to use the exact HighLevelVTC ROI or a broader VTC mask.
Choose whether outside-mask values become 0 or NaN.

Notes
-----
- This is for visualization only.
- Do NOT overwrite your original files.
- Works with HCP-MMP dlabel files in fsLR 32k / 91k CIFTI space.
"""

import os
import numpy as np
import nibabel as nib


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
WBDIR = os.path.join(REPO_ROOT, "reproduced_outputs", "surface_maps", "workbench_dscalars")

INPUT_DSCALAR = os.path.join(
    WBDIR, "wb_group_mean_r_joint_by_model.dscalar.nii"
)
OUTPUT_DSCALAR = os.path.join(
    WBDIR, "wb_group_mean_r_joint_by_model__HighLevelVTCmasked.dscalar.nii"
)
ATLAS_DIR = os.path.join(REPO_ROOT, "resources", "atlas")

HCP_DLABEL = os.path.join(
    ATLAS_DIR,
    "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors"
    ".32k_fs_LR.dlabel.nii",
)
# Historical display-mask settings.
# IMPORTANT: despite the legacy output filename ending in
# "__HighLevelVTCmasked", the executed historical configuration below uses the
# BROAD VISUAL CORTEX parcel list. This is visualization-only and does not alter
# the formal HighLevelVTC ROI used for inference.
# Choose one:
USE_EXACT_HighLevelVTC = False   # exact ROI from your main analyses
USE_BROAD_VTC = True      # broader ventral temporal mask

# What to put outside the mask:
#   "zero" -> set outside voxels/vertices to 0
#   "nan"  -> set outside voxels/vertices to NaN (usually cleaner for display)
OUTSIDE_MODE = "nan"

# =============================================================================
# VISUAL CORTEX MASK OPTIONS
# =============================================================================
# ROI_GROUPS = OrderedDict([
#     ("EarlyVisual",        ["V1", "V2", "V3", "V4"]),
#     ("IntermediateVisual", ["V8", "PIT", "LO1", "LO2", "LO3"]),
#     ("HighLevelVTC", ["FFC", "VVC", "TE1p", "TE2p"]),
  
# ])
# Option 1: conservative mask matching the main ROI hierarchy
CONSERVATIVE_VISUAL_CORTEX_PARCELS = [
    # Early visual
    "V1", "V2", "V3", "V4",

    # Intermediate visual
    "V8", "PIT", "LO1", "LO2", "LO3",

    # Food/VTC ROI
    "FFC", "VVC", "TE1p", "TE2p",
]

# Option 2: broader HCP-MMP visual cortex display mask
BROAD_VISUAL_CORTEX_PARCELS = [
    # Early / retinotopic visual cortex
    "V1", "V2", "V3", "V4",
    "V3A", "V3B", "V3CD",
    "V6", "V6A", "V7", "V8",

    # Lateral occipital / intermediate visual cortex
    "LO1", "LO2", "LO3",
    "PIT",

    # Motion / lateral temporal visual areas
    "MT", "MST", "FST",

    # Ventral temporal visual cortex
    "FFC", "VVC",
    "VMV1", "VMV2", "VMV3",
    "TE1a", "TE1m", "TE1p", "TE2p",

    # Posterior parahippocampal / scene-related visual cortex
    "PHA1", "PHA2", "PHA3",
    "PH",
]

# Choose which mask to use
USE_BROAD_VISUAL_CORTEX = True

if USE_BROAD_VISUAL_CORTEX:
    target_parcels = BROAD_VISUAL_CORTEX_PARCELS
    mask_name = "BroadVisualCortex"
else:
    target_parcels = CONSERVATIVE_VISUAL_CORTEX_PARCELS
    mask_name = "ConservativeVisualHierarchy"

# Optional: map indices to keep.
# If None, keep all maps in the input dscalar.
# Example: [1] to keep only the 2nd map, or [0, 2, 4]
KEEP_MAP_INDICES = None


# =============================================================================
# HELPERS
# =============================================================================

def normalize_hcp_name(name):
    """
    Normalize HCP parcel names to make matching robust.
    Examples:
        L_FFC_ROI -> FFC
        R_TE1p_ROI -> TE1p
    """
    out = str(name)

    if out.startswith("L_"):
        out = out[2:]
    if out.startswith("R_"):
        out = out[2:]
    if out.endswith("_ROI"):
        out = out[:-4]

    return out


def get_label_axis_dict(dlabel_img):
    """
    Return label dictionary from a CIFTI dlabel file.
    """
    label_axis = dlabel_img.header.get_axis(0)

    # Usually only one label map in the dlabel
    # label_axis.label[0] gives dict: int -> (name, rgba)
    if not hasattr(label_axis, "label"):
        raise RuntimeError("Could not access label axis label dictionary.")

    label_dict = label_axis.label[0]
    return label_dict


def build_parcel_mask_from_dlabel(dlabel_img, target_parcels):
    """
    Build a boolean mask over grayordinates selecting the requested HCP parcels.
    """
    label_dict = get_label_axis_dict(dlabel_img)

    label_data = np.asanyarray(dlabel_img.dataobj)
    if label_data.ndim != 2 or label_data.shape[0] < 1:
        raise RuntimeError(f"Unexpected dlabel data shape: {label_data.shape}")

    labels = label_data[0].astype(int)

    selected_label_ids = []
    found_names = []

    for lab_id, lab_info in label_dict.items():
        if lab_id == 0:
            continue  # unlabeled / medial wall

        # lab_info is usually (name, rgba)
        if isinstance(lab_info, tuple):
            lab_name = lab_info[0]
        else:
            lab_name = str(lab_info)

        clean_name = normalize_hcp_name(lab_name)

        if clean_name in target_parcels:
            selected_label_ids.append(lab_id)
            found_names.append(clean_name)

    selected_label_ids = sorted(set(selected_label_ids))
    found_names = sorted(set(found_names))

    missing = sorted(set(target_parcels) - set(found_names))

    print("\nRequested parcels:")
    print(target_parcels)

    print("\nFound parcels in atlas:")
    print(found_names)

    if missing:
        print("\nWARNING: some requested parcels were not found:")
        print(missing)

    mask = np.isin(labels, selected_label_ids)
    return mask, found_names, missing


def get_scalar_map_names(dscalar_img):
    """
    Return scalar map names from a dscalar file.
    """
    scalar_axis = dscalar_img.header.get_axis(0)
    names = []
    for i in range(len(scalar_axis)):
        try:
            elem = scalar_axis.get_element(i)
            if isinstance(elem, tuple):
                names.append(str(elem[0]))
            else:
                names.append(str(elem))
        except Exception:
            names.append(f"map_{i}")
    return names


# =============================================================================
# MAIN
# =============================================================================

def main():

    if not os.path.isfile(INPUT_DSCALAR):
        raise FileNotFoundError(INPUT_DSCALAR)

    if not os.path.isfile(HCP_DLABEL):
        raise FileNotFoundError(HCP_DLABEL)

    if USE_EXACT_HighLevelVTC and USE_BROAD_VTC:
        raise ValueError("Choose only one of USE_EXACT_HighLevelVTC or USE_BROAD_VTC.")

    if not USE_EXACT_HighLevelVTC and not USE_BROAD_VTC:
        raise ValueError("Set one of USE_EXACT_HighLevelVTC or USE_BROAD_VTC to True.")

    if OUTSIDE_MODE not in ["zero", "nan"]:
        raise ValueError("OUTSIDE_MODE must be 'zero' or 'nan'.")


    print("Loading dscalar:")
    print(INPUT_DSCALAR)
    dscalar_img = nib.load(INPUT_DSCALAR)
    dscalar_data = np.asanyarray(dscalar_img.dataobj).copy()

    print("\nLoading dlabel atlas:")
    print(HCP_DLABEL)
    dlabel_img = nib.load(HCP_DLABEL)

    mask, found_names, missing = build_parcel_mask_from_dlabel(dlabel_img, target_parcels)

    print(f"\nMask includes {int(mask.sum())} grayordinates.")

    if dscalar_data.ndim != 2:
        raise RuntimeError(f"Unexpected dscalar data shape: {dscalar_data.shape}")

    n_maps, n_grayords = dscalar_data.shape

    if mask.shape[0] != n_grayords:
        raise RuntimeError(
            f"Mask length ({mask.shape[0]}) does not match dscalar grayordinates ({n_grayords})."
        )

    map_names = get_scalar_map_names(dscalar_img)
    print("\nAvailable maps:")
    for i, name in enumerate(map_names):
        print(f"{i:2d}  {name}")

    # Keep subset of maps if requested
    if KEEP_MAP_INDICES is not None:
        dscalar_data = dscalar_data[KEEP_MAP_INDICES, :]
        selected_names = [map_names[i] for i in KEEP_MAP_INDICES]
        print("\nKeeping only maps:")
        for i, name in zip(KEEP_MAP_INDICES, selected_names):
            print(f"{i:2d}  {name}")
    else:
        selected_names = map_names

    # Apply mask
    masked_data = dscalar_data.astype(np.float32).copy()

    if OUTSIDE_MODE == "zero":
        outside_value = 0.0
    else:
        outside_value = np.nan

    masked_data[:, ~mask] = outside_value

    # Save
    new_img = nib.Cifti2Image(
        masked_data,
        header=dscalar_img.header,
        nifti_header=dscalar_img.nifti_header,
    )
    nib.save(new_img, OUTPUT_DSCALAR)

    print("\nSaved masked dscalar:")
    print(OUTPUT_DSCALAR)

    # Save a small note file
    note_path = os.path.splitext(os.path.splitext(OUTPUT_DSCALAR)[0])[0] + "_notes.txt"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("Masked Workbench dscalar for visualization only.\n")
        f.write(f"Input: {INPUT_DSCALAR}\n")
        f.write(f"Atlas: {HCP_DLABEL}\n")
        f.write(f"Mask type: {'HighLevelVTC' if USE_EXACT_HighLevelVTC else 'Broad visual cortex display mask'}\n")
        f.write(f"Parcels used: {', '.join(target_parcels)}\n")
        f.write(f"Parcels found: {', '.join(found_names)}\n")
        f.write(f"Missing parcels: {', '.join(missing) if missing else 'None'}\n")
        f.write(f"Outside mode: {OUTSIDE_MODE}\n")
        f.write(f"Mask grayordinates: {int(mask.sum())}\n")
        f.write("Formal inference was not altered; masking was for display only.\n")

    print("\nSaved note file:")
    print(note_path)


if __name__ == "__main__":
    main()