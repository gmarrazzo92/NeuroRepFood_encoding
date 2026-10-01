# Public data download

This folder contains the downloader for the minimal public data stream required
to reproduce the encoding analyses.

## Dataset

DataverseNL dataset:

- DOI: `10.34894/TVPLVR`
- pinned release version: `1.0`

The downloader uses the file-level Dataverse API and verifies every downloaded
file against the SHA-1 checksum recorded in `public_data_manifest.csv`.

## Run

From the repository root:

```bash
python data_download/download_public_data.py
```

The required folders are created automatically.

A non-downloading preflight is available with:

```bash
python data_download/download_public_data.py --dry-run
```

Interrupted downloads are retained as `.part` files and resumed when the
Dataverse server supports HTTP Range requests. Files already present with the
expected size and SHA-1 checksum are skipped.

## Data created locally

The public Dataverse download contains exactly 201 files (~11.23 GiB):

- 75 GLMsingle files: three files for each of the 25 included participants
  (`betas_trials.npy`, `stimorder.npy`, and
  `good_grayordinates_mask.npy`);
- 96 food stimulus images plus `ordered_stimuli.csv`;
- 25 subject-level food-rating files;
- three inherited visual-feature inputs used by the historical feature
  extraction (`gabor_v1_rsa_model.npz`, `V4_PCA_matrix.npy`,
  `IT_PCA_matrix.npy`);
- the exact HCP-MMP atlas used for ROI definitions.

The resulting tree is:

```text
data/
├── glmsingle/
│   └── sub-*/
│       ├── betas_trials.npy
│       ├── stimorder.npy
│       └── good_grayordinates_mask.npy
├── ratings/
└── stimuli/
    ├── images/
    ├── cornet_features/
    ├── ordered_stimuli.csv
    └── gabor_v1_rsa_model.npz

resources/
├── atlas/
└── derived_inputs/
    └── parent_rsa/
        └── model_rdm_vectors.csv
```

## Scope

This downloader intentionally starts from the public GLMsingle trial-level
derivatives. Re-running preprocessing and GLMsingle from raw BIDS data is a
deeper provenance route and is not required for reproduction of the encoding
analyses.
