#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/evaluate_paper.sh [GPU_ID] [all|zero|top4]"
  echo "Defaults: GPU_ID=0, setting=all"
  echo "Optional: set BRAD_RUNTIME_ROOT to an alternate data/models runtime root"
  exit 0
fi

GPU_ID="${1:-0}"
SETTING="${2:-all}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/evaluate_paper.sh [GPU_ID] [all|zero|top4]" >&2
  exit 2
fi
if [[ "${SETTING}" != "all" && "${SETTING}" != "zero" && "${SETTING}" != "top4" ]]; then
  echo "Usage: bash scripts/evaluate_paper.sh [GPU_ID] [all|zero|top4]" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "B-RAD-Det Table 5 evaluation"
echo "Project: ${PROJECT_DIR}"
echo "Physical GPU: ${GPU_ID}"
echo "PyTorch device: cuda:0"
echo "Setting: ${SETTING}"

RUNTIME_ARGS=()
if [[ -n "${BRAD_RUNTIME_ROOT:-}" ]]; then
  RUNTIME_ARGS=(--runtime-root "${BRAD_RUNTIME_ROOT}")
  echo "Runtime root: ${BRAD_RUNTIME_ROOT}"
fi

if [[ "${SETTING}" == "zero" ]]; then
  SHOTS=(0)
elif [[ "${SETTING}" == "top4" ]]; then
  SHOTS=(4)
else
  SHOTS=(0 4)
fi

for target in id_validation ood_test buscot; do
  for fold in 0 1 2 3 4; do
    for shots in "${SHOTS[@]}"; do
      echo
      echo "Running target=${target} fold=${fold} shots=${shots}"
      python scripts/evaluate.py \
        --target "${target}" \
        --fold "${fold}" \
        --shots "${shots}" \
        --device cuda:0 \
        "${RUNTIME_ARGS[@]}"
    done
  done
done

echo
echo "Five-fold Table 5 summaries"
for shots in "${SHOTS[@]}"; do
  python scripts/summarize_evaluation.py --target all --shots "${shots}"
done
