#!/usr/bin/env bash
# End-to-end B-RAD inference on one bundled sample (default usg_0059, see QID):
#   B-RAD-CLIP (retrieval) -> B-RAD-Det (box) -> B-RAD-Seg (mask) -> B-RAD-FULL (BI-RADS).
#
# B-RAD-CLIP, B-RAD-Det and B-RAD-FULL share one environment; B-RAD-Seg needs its
# own (OpenMMLab), so it is launched with its own interpreter. Override the
# variables below for your setup. The detector runs zero-shot here; the paper's
# retrieval-guided few-shot numbers come from B-RAD-Det/scripts/evaluate.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAMPLES="${SAMPLES:-$ROOT/samples}"
OUT="${OUT:-$SAMPLES/predictions}"
mkdir -p "$OUT"

# Per-module interpreters (point these at the right conda envs).
CLIP_PY="${CLIP_PY:-python}"   # torch + transformers    (B-RAD-CLIP, B-RAD-FULL)
SEG_PY="${SEG_PY:-python}"     # mmdet + mmcv + mmpretrain (B-RAD-Seg)

# Checkpoints (in ckpts/; download with ckpts/download_weights.sh).
CLIP_CKPT="${CLIP_CKPT:-$ROOT/ckpts/b_rad_clip.pt}"
FULL_CKPT="${FULL_CKPT:-$ROOT/ckpts/b_rad_full.pt}"
SEG_CKPT="${SEG_CKPT:-$ROOT/ckpts/UltraSam.pth}"

# Which bundled sample to run; override with QID=<uid> (see samples/sample.csv).
QID="${QID:-usg_0059}"
SPEC="$("$CLIP_PY" - "$SAMPLES/sample.csv" "$QID" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
row = next((r for r in rows if r["uid"] == sys.argv[2]), None)
if row is None:
    raise SystemExit(f'uid {sys.argv[2]!r} not in sample.csv; have: '
                     + ", ".join(r["uid"] for r in rows))
print(f'IMG="{row["image"]}"; MASK="{row["mask"]}"; BIRADS={row["birads"]}')
PY
)" || exit 1
eval "$SPEC"
IMG="$SAMPLES/$IMG"; MASK="$SAMPLES/$MASK"
echo ">> Sample $QID  (GT BI-RADS $BIRADS)"

echo ">> [1/4] B-RAD-CLIP retrieval"
"$CLIP_PY" "$ROOT/B-RAD-CLIP/infer_retrieval.py" \
  --ckpt "$CLIP_CKPT" --bank "$ROOT/B-RAD-CLIP/assets/reference_bank.npz" \
  --image "$IMG" --uid "$QID" --topk 5 --alpha 0.5 \
  --text-csv "$SAMPLES/text/medgemma_text.csv" \
  --out "$OUT/${QID}_retrieval.json"

# Support cases for 4-shot detection: the top retrieved references. Their images
# ship with the demo; with the full corpus on disk any retrieved case resolves.
REFS=$("$CLIP_PY" - "$OUT/${QID}_retrieval.json" "$SAMPLES" "$QID" <<'REFPY'
import json, sys
from pathlib import Path
retrieved, samples, query = json.load(open(sys.argv[1])), Path(sys.argv[2]), sys.argv[3]
specs = []
for row in retrieved:
    image = samples / f"images/{row['uid']}.png"
    mask = samples / f"masks/{row['uid']}_mask.png"
    if image.exists() and mask.exists():
        specs.append(f"{image}:{mask}")
if not specs:      # nothing retrieved is on disk; fall back to the bundled cases
    specs = [f"{p}:{samples}/masks/{p.stem}_mask.png"
             for p in sorted(samples.glob("images/*.png"))
             if p.stem != query and (samples / f"masks/{p.stem}_mask.png").exists()]
print(" ".join(specs[:4]))
REFPY
)

echo ">> [2/4] B-RAD-Det retrieval-guided detection"
# shellcheck disable=SC2086
"$CLIP_PY" "$ROOT/B-RAD-Det/inference_single.py" \
  --image "$IMG" --ckpt "$ROOT/ckpts/b_rad_det.pt" \
  ${REFS:+--references $REFS} \
  --out "$OUT/${QID}_det.json"
PRED_BBOX=$("$CLIP_PY" -c "import json,sys; print(','.join(str(v) for v in json.load(open(sys.argv[1]))['bbox_xyxy']))" "$OUT/${QID}_det.json")

echo ">> [3/4] B-RAD-Seg segmentation"
"$SEG_PY" "$ROOT/B-RAD-Seg/infer_bbox.py" \
  --image "$IMG" --bbox "$PRED_BBOX" --ckpt "$SEG_CKPT" \
  --gt-mask "$MASK" --out-mask "$OUT/${QID}_seg.png" --out "$OUT/${QID}_seg.json"

echo ">> [4/4] B-RAD-FULL BI-RADS classification (ROI = predicted mask)"
"$CLIP_PY" "$ROOT/B-RAD-FULL/infer_birads.py" \
  --ckpt "$FULL_CKPT" \
  --image "$IMG" --mask "$OUT/${QID}_seg.png" \
  --text-csv "$SAMPLES/text/medgemma_text.csv" \
  --mode concat --out "$OUT/${QID}_birads.json"

echo ">> Done. Outputs in $OUT/"
