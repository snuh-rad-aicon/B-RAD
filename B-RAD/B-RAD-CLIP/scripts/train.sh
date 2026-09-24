#!/usr/bin/env bash
# Train B-RAD-CLIP: 5-fold, unpaired BI-RADS-supervised contrastive learning.
# Required: CSV_SPLIT, TEXT_JSON, IMAGE_DIR. Extra flags are passed to train.py.
set -euo pipefail
cd "$(dirname "$0")/.."

CSV_SPLIT="${CSV_SPLIT:?set CSV_SPLIT=<csv with image_name,birads>}"
TEXT_JSON="${TEXT_JSON:?set TEXT_JSON=<JSONL with birads,natural_text>}"
IMAGE_DIR="${IMAGE_DIR:?set IMAGE_DIR=<root holding the images>}"
OUT_DIR="${OUT_DIR:-../ckpts}"

python train.py \
  --csv-split "$CSV_SPLIT" \
  --text-json "$TEXT_JSON" \
  --image-dir "$IMAGE_DIR" \
  --out-dir "$OUT_DIR" "$@"
