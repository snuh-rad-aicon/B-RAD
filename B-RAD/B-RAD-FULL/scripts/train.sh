#!/usr/bin/env bash
# Train the B-RAD-FULL BI-RADS head on a frozen B-RAD-CLIP encoder (5 folds).
# Required: DATA_CSV (image,mask,birads), TEXT_CSV, DATA_ROOT.
set -euo pipefail
cd "$(dirname "$0")/.."

ENCODER="${ENCODER:-../ckpts/b_rad_full.pt}"
DATA_CSV="${DATA_CSV:?set DATA_CSV=<csv with image,mask,birads>}"
TEXT_CSV="${TEXT_CSV:?set TEXT_CSV=<csv with image_name,medgemma_txt>}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT=<root the paths are relative to>}"
OUT="${OUT:-../ckpts/b_rad_full_retrained.pt}"

python train.py \
  --encoder "$ENCODER" \
  --data-csv "$DATA_CSV" \
  --text-csv "$TEXT_CSV" \
  --data-root "$DATA_ROOT" \
  --out "$OUT" "$@"
