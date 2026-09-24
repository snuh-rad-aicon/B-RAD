#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/train_base.sh [GPU_ID] [FOLD] [additional arguments]"
  echo "Defaults: GPU_ID=0, FOLD=0"
  exit 0
fi

GPU_ID="${1:-0}"
FOLD="${2:-0}"
if (( $# >= 1 )); then shift; fi
if (( $# >= 1 )); then shift; fi

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]] || [[ ! "${FOLD}" =~ ^[0-4]$ ]]; then
  echo "Usage: bash scripts/train_base.sh [GPU_ID] [FOLD] [additional arguments]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "B-RAD-Det base training"
echo "Project: ${PROJECT_DIR}"
echo "Physical GPU: ${GPU_ID}"
echo "PyTorch device: cuda:0"
echo "Fold: ${FOLD}"
echo "Config: configs/train_base.yaml"
echo "Checkpoint directory: models/checkpoints/base"

exec python scripts/train_base.py --fold "${FOLD}" --device cuda:0 "$@"
