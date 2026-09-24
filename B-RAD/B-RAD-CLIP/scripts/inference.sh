#!/usr/bin/env bash
# Retrieve the top-k references for one image (image + report fusion).
# Override any variable; extra flags are passed to infer_retrieval.py.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${CKPT:-../ckpts/b_rad_clip.pt}"
IMAGE="${IMAGE:-../samples/images/usg_0059.png}"
TEXT_CSV="${TEXT_CSV:-../samples/text/medgemma_text.csv}"
ALPHA="${ALPHA:-0.5}"
TOPK="${TOPK:-5}"
QID="$(basename "$IMAGE" .png)"

python infer_retrieval.py \
  --ckpt "$CKPT" \
  --bank assets/reference_bank.npz \
  --image "$IMAGE" --uid "$QID" \
  --text-csv "$TEXT_CSV" \
  --alpha "$ALPHA" --topk "$TOPK" "$@"
