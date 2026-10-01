# 01 — Feature extraction

This component reconstructs the feature matrices used by the encoding analyses.

## Exact reproduction strategy

Most feature arrays can be reconstructed exactly from the public stimuli,
ratings, and downloaded inherited feature inputs.

The deep-network arrays are a special case. Re-running the historical AlexNet
and OpenAI CLIP forward passes from the same images and public weights produced
the same representational geometry to high precision, but not bit-identical
arrays under the current PyTorch/CUDA environment. This was negligible for
AlexNet but was amplified by the historical 50-component global PCA for the
lower-variance CLIP components.

For exact downstream reproduction, the four small deep-network arrays used by
the executed analysis are therefore distributed as fixed derived inputs:

```text
resources/
└── derived_inputs/
    └── deep_features/
        ├── AlexNetMid.npy
        ├── AlexNetHigh.npy
        ├── CLIP.npy
        └── CLIP_full512.npy
```

Their total size is small (~250 KiB). They are copied unchanged into the
canonical feature output directory by `extract_features.py`.

`reextract_deep_features.py` separately preserves the original image-model
extraction procedure for provenance and transparency. Its outputs are written
to a separate diagnostic directory and are not used by the exact reproduction
pipeline.

## Inputs

First run:

```bash
python data_download/download_public_data.py
```

The canonical feature script then reads:

```text
data/
├── ratings/
│   └── {subject}_rating.txt
└── stimuli/
    ├── ordered_stimuli.csv
    ├── gabor_v1_rsa_model.npz
    ├── images/
    │   └── <96 food images>.jpg
    └── cornet_features/
        ├── V4_PCA_matrix.npy
        └── IT_PCA_matrix.npy

resources/
└── derived_inputs/
    └── deep_features/
        ├── AlexNetMid.npy
        ├── AlexNetHigh.npy
        ├── CLIP.npy
        └── CLIP_full512.npy
```

## Canonical run

From the repository root:

```bash
python analysis/01_feature_extraction/extract_features.py
```

Output:

```text
reproduced_outputs/
└── feature_extraction/
```

The script reconstructs the categorical, behavioral, Gabor, CORnet, and color
features and installs the exact four frozen deep-network feature arrays.

It also saves:

```text
rv_coefficient_matrix.csv
```

The historical feature script computed and printed this RV matrix but did not
save it. Saving it here makes Supplementary Table S9 directly reproducible.

## Optional deep-feature re-extraction

To reproduce the original AlexNet/CLIP extraction procedure from the stimulus
images:

```bash
python analysis/01_feature_extraction/reextract_deep_features.py
```

This writes to:

```text
reproduced_outputs/deep_feature_reextraction/
```

These files are diagnostic/provenance outputs only and are not substituted for
the fixed arrays used by the historical analysis.

This optional script requires PyTorch, torchvision, and the original OpenAI
CLIP implementation:

```bash
pip install git+https://github.com/openai/CLIP.git
```

## Historical definitions preserved

The provenance re-extraction script preserves the executed historical choices:

- AlexNet hooks at `features[6]`, `features[10]`, and `classifier[4]`;
- 50-component global PCA across all 96 stimuli for AlexNetMid and AlexNetHigh;
- OpenAI CLIP ViT-B/32;
- 50-component global PCA across all 96 unnormalized CLIP embeddings for
  `CLIP.npy`;
- row-wise L2 normalization of the 512-D CLIP embeddings for
  `CLIP_full512.npy`.

The canonical script preserves the historical non-deep feature construction,
including subject-wise behavioral z-scoring followed by group averaging,
StandardScaler transformations, and the original categorical definitions.

## Expected canonical outputs

```text
AlexNetHigh.npy             96 x 50
AlexNetMid.npy              96 x 50
CLIP.npy                    96 x 50
CLIP_full512.npy            96 x 512
CORnetIT.npy                96 x 93
CORnetV4.npy                96 x 93
Calorie.npy                 96 x 1
CalorieObjective.npy        96 x 1
Calorie_persubject.npy      96 x 25
Color.npy                   96 x 512
Familiarity.npy             96 x 1
Familiarity_persubject.npy  96 x 25
Gabor.npy                   96 x 384
Health.npy                  96 x 1
Health_persubject.npy       96 x 25
Palatability.npy            96 x 1
Palatability_persubject.npy 96 x 25
SavorySweet.npy             96 x 1
stimulus_names.npy          96

rv_coefficient_matrix.csv   13 x 13
```

The next pipeline component, `02_feature_bands`, consumes these canonical
feature arrays.
