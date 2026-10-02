# 01 — Feature extraction

This component reconstructs the feature matrices used by the encoding analyses
from the public stimulus images, behavioral ratings, and inherited CORnet/Gabor
inputs.

## Inputs

First run:

```bash
python data_download/download_public_data.py
```

The feature-extraction script reads:

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
```

## Run

From the repository root:

```bash
python analysis/01_feature_extraction/extract_features.py
```

Outputs are written to:

```text
reproduced_outputs/
└── feature_extraction/
```

The script reconstructs:

- categorical food labels;
- group and per-subject behavioral rating features;
- Gabor and color features;
- the supplied CORnet V4 and IT feature matrices;
- AlexNet mid- and high-level features from the stimulus images;
- OpenAI CLIP ViT-B/32 features from the stimulus images; and
- the full 512-dimensional normalized CLIP image embeddings used by later
  diagnostic analyses.

For AlexNet, the script uses hooks at `features[6]`, `features[10]`, and
`classifier[4]`. AlexNetMid is formed from concatenated conv3 and conv5
activations and reduced to 50 principal components; AlexNetHigh is derived from
fc6 and reduced to 50 principal components. CLIP uses ViT-B/32; `CLIP.npy` is
reduced to 50 principal components and `CLIP_full512.npy` contains row-wise
L2-normalized 512-dimensional image embeddings.

The script also computes and **prints** an RV-coefficient matrix across the
feature spaces as a diagnostic. It does not save that matrix to CSV.

## Deep-feature reference arrays

The repository additionally contains four small arrays under:

```text
resources/
└── derived_inputs/
    └── deep_features/
        ├── AlexNetMid.npy
        ├── AlexNetHigh.npy
        ├── CLIP.npy
        └── CLIP_full512.npy
```

These are the exact deep-network arrays from the executed historical analysis
and are retained as reference/provenance inputs for comparison with a fresh
re-extraction. They are **not** copied into `reproduced_outputs/` by
`extract_features.py` and are not substituted for newly extracted arrays in the
full rerun workflow.

Small numerical differences can arise when pretrained-network features are
re-extracted under different PyTorch, torchvision, CLIP, CUDA, or hardware
environments. The bundled arrays therefore provide a fixed record of the
historical feature matrices used for the reported analysis.

## Software note for CLIP

The full feature-extraction workflow requires the original OpenAI CLIP Python
package because both the 50-dimensional CLIP representation and the full
512-dimensional CLIP embeddings are extracted from the images:

```bash
pip install git+https://github.com/openai/CLIP.git
```

## Expected outputs

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
```

The next pipeline component, `02_feature_bands`, consumes these reconstructed
feature arrays.
