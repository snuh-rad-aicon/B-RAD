# B-RAD-Det

Retrieval-guided few-shot lesion detection module for B-RAD.

## End-to-end workflow

Create the validated environment from the B-RAD repository root and enter this
module:

```bash
conda env create -f environment.yml
conda activate b-rad
cd B-RAD/B-RAD-Det
```

Prepare the datasets using the download, extraction, conversion, metadata merge,
and split commands in [datasets/README.md](datasets/README.md). The resulting
runtime tree is rooted at `data/` and all metadata paths remain relative. BUS-CoT
requires the separately requested processed package described in
[datasets/BUSCOT_ACCESS.md](datasets/BUSCOT_ACCESS.md).

Place the fold-matched checkpoints under `models/checkpoints/base/` and
`models/checkpoints/fewshot/` as documented in
[models/README.md](models/README.md). Download URLs resolve to the public
B-RAD-Det model repository.

Base training is optional when the released checkpoints are used. To train fold
0 on the default visible GPU:

```bash
bash scripts/train_base.sh
```

The wrapper writes the best and final base checkpoints under
`models/checkpoints/base/`. Train all five base folds before starting few-shot
training when reproducing the full training pipeline.

After the matching base final checkpoint exists, retrieval-guided training for
fold 0 is:

```bash
bash scripts/train_fewshot.sh
```

It writes the stage-2 checkpoints under `models/checkpoints/fewshot/`. Both
training wrappers accept optional physical GPU and fold arguments; run them with
`--help` after those positional arguments to inspect the underlying Python CLI.

## Data preparation

Follow [datasets/README.md](datasets/README.md) for the complete reproducible
workflow: source-dataset download, archive placement, conversion to the B-RAD
image/mask layout, metadata merge, fold construction, and retrieval-table merge.
The released annotations, generated text, folds, retrieval lists, and runtime
tables are hosted in the
[B-RAD-Det dataset repository](https://huggingface.co/datasets/Subin94/B-RAD-Det-Dataset).

Start with the public source datasets:

```bash
python datasets/download.py all --output-dir data/downloads
```

Then run the provider-specific extraction and preparation commands in
[datasets/README.md](datasets/README.md). BUS-CoT is a separately held-out
evaluation cohort and is not downloaded by the public-data command; follow
[datasets/BUSCOT_ACCESS.md](datasets/BUSCOT_ACCESS.md) to request the processed
package from its corresponding authors.

Evaluate every configured target and aggregate all five folds with:

```bash
bash scripts/evaluate_paper.sh
```

Fold metrics are written to `results/*_metrics.json`. Per-image predictions are
not written by default; use `scripts/evaluate.py --save-predictions` only for
private validation when they are required.

## Reproduce the paper detection results

The commands below reproduce the B-RAD-Det rows of Table 5; the paper is the reference of record for the reported values. Create the release environment from the repository root, then run the commands from this module directory after preparing the datasets and checkpoints.

```bash
conda env create -f environment.yml
conda activate b-rad
cd B-RAD/B-RAD-Det
```

Prepare the relative data tree by following [datasets/README.md](datasets/README.md). Place the five `brad_det_fewshot_1shot_best_fold{0..4}.pth` checkpoints as described in [models/README.md](models/README.md). The evaluator uses one query per batch and reports bbox Dice (`mDSC`) and IoU (`mIoU`). In the top-4 setting it generates one candidate per retrieved reference and scores the maximum candidate, as specified in the paper.

### ID validation

Run all five folds for both settings:

```bash
for fold in 0 1 2 3 4; do
  python scripts/evaluate.py --target id_validation --shots 0 --fold "$fold"
  python scripts/evaluate.py --target id_validation --shots 4 --fold "$fold"
done

python scripts/summarize_evaluation.py --target id_validation --shots 0
python scripts/summarize_evaluation.py --target id_validation --shots 4
```

The release uses fold-matched base-best checkpoints for zero-shot evaluation and fold-matched one-shot-trained few-shot checkpoints for top-4 evaluation.

### OOD: BrEaST, BUS-STU, and UDIAT

The three cohorts are evaluated together as the 457-image OOD split used in Table 5.

```bash
for fold in 0 1 2 3 4; do
  python scripts/evaluate.py --target ood_test --shots 0 --fold "$fold"
  python scripts/evaluate.py --target ood_test --shots 4 --fold "$fold"
done

python scripts/summarize_evaluation.py --target ood_test --shots 0
python scripts/summarize_evaluation.py --target ood_test --shots 4
```

### BUS-CoT external evaluation

BUS-CoT is a separate held-out cohort and is not merged with the 457-image OOD split. Request the processed evaluation package from the corresponding authors as explained in [datasets/BUSCOT_ACCESS.md](datasets/BUSCOT_ACCESS.md), then place its merged metadata at `data/metadata/external/buscot_with_references.csv`.

```bash
for fold in 0 1 2 3 4; do
  python scripts/evaluate.py --target buscot --shots 0 --fold "$fold"
  python scripts/evaluate.py --target buscot --shots 4 --fold "$fold"
done

python scripts/summarize_evaluation.py --target buscot --shots 0
python scripts/summarize_evaluation.py --target buscot --shots 4
```

BUS-CoT numbers are reported after the radiologist-reviewed QC exclusions supplied with the processed package.

Fold-level aggregate JSON files are written under `results/`. Per-image prediction CSVs are private validation artifacts and are written only when `--save-predictions` is explicitly supplied.

## Other workflows

- Dataset preparation: [datasets/README.md](datasets/README.md)
- Checkpoint placement and downloads: [models/README.md](models/README.md)
