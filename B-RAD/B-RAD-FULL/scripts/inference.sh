#!/usr/bin/env bash
# Predict BI-RADS for one image with its lesion mask as the ROI.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${CKPT:-../ckpts/b_rad_full.pt}"
IMAGE="${IMAGE:-../samples/images/usg_0059.png}"
MASK="${MASK:-../samples/masks/usg_0059_mask.png}"
TEXT_CSV="${TEXT_CSV:-../samples/text/medgemma_text.csv}"

python infer_birads.py \
  --ckpt "$CKPT" \
  --image "$IMAGE" --mask "$MASK" \
  --text-csv "$TEXT_CSV" \
  --mode concat "$@"
