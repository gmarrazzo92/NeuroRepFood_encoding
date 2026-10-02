# Deep-network feature reference arrays

These four arrays are the exact deep-network feature matrices used by the
executed historical analysis:

- `AlexNetMid.npy`
- `AlexNetHigh.npy`
- `CLIP.npy`
- `CLIP_full512.npy`

They are retained as fixed reference/provenance arrays so that a newly
re-extracted feature set can be compared with the matrices used for the
reported results. They are not copied into `reproduced_outputs/` and are not
used in place of newly extracted features by the standard full rerun.

The extraction procedure is implemented directly in:

`analysis/01_feature_extraction/extract_features.py`

That script re-extracts AlexNet and OpenAI CLIP features from the public
stimulus images. Small numerical differences from these historical arrays can
occur across PyTorch, torchvision, CLIP, CUDA, or hardware environments.

SHA-256 checksums:

- `AlexNetMid.npy` (19,328 bytes): `83c981cf185190819fe891905a77683f68c9910b9f27ff311cabd1fccd46de33`
- `AlexNetHigh.npy` (19,328 bytes): `da7e8c0780e2038a673d2481d9d89bb1256f6429515007cf77998273592e2469`
- `CLIP.npy` (19,328 bytes): `51e1df01dfa261e5a9c0b1ee04154ef6d13434f44219b7147a3cec3b478f21a7`
- `CLIP_full512.npy` (196,736 bytes): `2a910a2a8c96e0380e06a03a8c17d6b3dd03d66b5aa0d7bf1cc51dda29811bed`
