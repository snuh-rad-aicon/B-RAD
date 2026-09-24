#!/usr/bin/env bash
# Download the released B-RAD-Det metadata from the public dataset repository.
#
#   https://huggingface.co/datasets/Subin94/B-RAD-Det-Dataset
#
# Two trees are written, both with repository-relative paths:
#
#   datasets/metadata/      reference tables (annotations, retrieval lists, folds)
#   data/metadata/splits/   the runtime train/valid/test manifests the configs read
#
# The runtime tree is what scripts/evaluate.py and the training entry points load
# directly, so downloading it skips the merge and split-construction steps.
# Original images and masks are not redistributed; see README.md for those.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${HF_DATASET_REPO:-Subin94/B-RAD-Det-Dataset}"
BASE="https://huggingface.co/datasets/${REPO}/resolve/main"

REF_DIR="${PROJECT}/datasets/metadata"
RUN_DIR="${PROJECT}/data/metadata/splits"

get() {  # get <remote-path> <local-path>
  local dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -s "$dest" ]]; then
    echo "  [skip] $dest"
    return
  fi
  echo "  [get ] $1"
  curl -sS --location --fail --retry 3 --retry-delay 2 -o "$dest" "${BASE}/$1"
}

echo ">> reference tables -> datasets/metadata/"
get metadata/annotations/brad_det_annotations.csv   "${REF_DIR}/brad_det_annotations.csv"
get metadata/retrieval/retrieval_top100.csv         "${REF_DIR}/retrieval_top100.csv"
get metadata/retrieval/fewshot_references.csv       "${REF_DIR}/fewshot_references.csv"
get metadata/folds/fewshot_splits.csv               "${REF_DIR}/fewshot_splits.csv"
get metadata/folds/paper_crawl.csv                  "${REF_DIR}/paper_crawl.csv"
get metadata/generated_text/medgemma_text.csv       "${REF_DIR}/medgemma_text.csv"
get metadata/folds/buscot_top100_tight_subset.csv   "${REF_DIR}/buscot_top100_tight_subset.csv"
get metadata/retrieval/buscot_retrieval_top100.csv  "${REF_DIR}/buscot_retrieval_top100.csv"

echo ">> runtime manifests -> data/metadata/splits/"
for fold in 0 1 2 3 4; do
  for role in train valid; do
    get "metadata/runtime/base/metadata_${role}_fold${fold}.csv" \
        "${RUN_DIR}/base/metadata_${role}_fold${fold}.csv"
    get "metadata/runtime/fewshot/metadata_${role}_fold${fold}.csv" \
        "${RUN_DIR}/fewshot/metadata_${role}_fold${fold}.csv"
  done
  get "metadata/runtime/evaluation/metadata_valid_fold${fold}.csv" \
      "${RUN_DIR}/evaluation/metadata_valid_fold${fold}.csv"
done
get metadata/runtime/metadata_test.csv "${RUN_DIR}/metadata_test.csv"

# train_fewshot.yaml reads the paper-crawl table from data/metadata/, evaluate.yaml
# from datasets/metadata/. Place it at both so either stage resolves it.
mkdir -p "${PROJECT}/data/metadata"
cp -f "${REF_DIR}/paper_crawl.csv" "${PROJECT}/data/metadata/paper_crawl.csv"

echo
echo "Done."
echo "  reference tables: ${REF_DIR}"
echo "  runtime splits  : ${RUN_DIR}"
