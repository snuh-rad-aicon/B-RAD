# B-RAD-FULL

Region-aware BI-RADS classification. The lesion mask becomes a foveal attention
bias on the ViT token grid: lesion tokens keep a bias of 0 and background tokens
take a `-gamma` penalty (`gamma = 1.18`), so the frozen encoder down-weights the
background while still retaining it. The image embedding is fused with the
report-text embedding and an MLP head predicts BI-RADS {2, 3, 4, 5}. The bias is
applied at inference only; nothing in the encoder is trained.

## Setup

```bash
pip install -r requirements.txt
```

This pulls the shared repository-root environment. The encoder package `brad_clip/` is
vendored here so the module is self-contained. `../ckpts/b_rad_full.pt` bundles
the encoder and the classification heads (download with
[../ckpts/download_weights.sh](../ckpts/download_weights.sh)).

## Classification

```bash
bash scripts/inference.sh             # bundled sample
bash scripts/inference.sh --verbose   # also print intermediate shapes
```

`--mode concat` is the region-aware configuration used in the paper: the image
embedding, taken under the ROI foveal bias, is concatenated with the report
embedding into 1024-d. Pass `--image`, `--mask` and `--text-csv` to
`infer_birads.py` for any other case.

## Training the head

The encoder stays frozen; only the 5-fold MLP head is fitted.

```bash
ENCODER=../ckpts/b_rad_full.pt \
DATA_CSV=<image,mask,birads.csv> \
TEXT_CSV=<image_name,medgemma_txt.csv> \
DATA_ROOT=<root> bash scripts/train.sh
```

## Interpretability

`BRADClip.attention_map(img, attention_bias=...)` returns the CLS to patch
attention over the 14x14 grid from the same pass that produced the prediction
(last layer by default, `rollout=True` accumulates all layers).
`notebooks/brad_inference_demo.ipynb` overlays it on the image.

## Files

```text
brad_clip/         encoder package (shared with B-RAD-CLIP)
infer_birads.py    single-image ROI-guided BI-RADS classification
train.py           fit the 5-fold MLP head on a frozen encoder
scripts/           train.sh, inference.sh
```
