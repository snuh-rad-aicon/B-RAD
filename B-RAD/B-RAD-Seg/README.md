# B-RAD-Seg

Foundational lesion segmentation. UltraSam, a SAM variant implemented in
MMDetection, is prompted zero-shot with the B-RAD-Det box and returns a lesion
mask, scored with Dice/IoU against the ground truth when one is given.

## Setup

```bash
pip install -r requirements.txt   # openmim, then:
mim install mmengine "mmcv>=2.0.0,<2.2.0" "mmdet>=3.3.0" "mmpretrain>=1.2.0"
```

Validated with Python 3.8, torch 2.0 (cu118), mmdet 3.3, mmcv 2.1, mmpretrain 1.2.
This module needs its own environment. The custom model code is vendored under
`endosam/` and registered through the config's `custom_imports`. Place
`UltraSam.pth` at `../ckpts/UltraSam.pth` (download with
[../ckpts/download_weights.sh](../ckpts/download_weights.sh)).

## Segmentation

```bash
python infer_bbox.py \
  --image <image.png> \
  --bbox <x0,y0,x1,y1> \
  --ckpt ../ckpts/UltraSam.pth \
  --gt-mask <mask.png> \
  --out-mask <out.png>
```

`--bbox` is the B-RAD-Det prediction. It is expanded by `--margin` (default 3%)
and used as the SAM box prompt; the best-scoring mask is decoded, resized to the
image, and scored when `--gt-mask` is given. `B-RAD/scripts/inference_full.sh`
runs this step as part of the end-to-end pipeline.

## Files

```text
configs/        UltraSam config chain (UltraSAM_full/UltraSAM_box_refine.py + _base_)
endosam/        custom SAM detector, heads, prompt encoders, transforms
infer_bbox.py   single-image bbox-prompted segmentation with Dice/IoU
```
