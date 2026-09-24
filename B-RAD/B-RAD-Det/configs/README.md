# Configuration

The configuration files preserve the active settings used by the original B-RAD-Det experiments.

- `model.yaml`: shared model, anchor, and normalization settings
- `train_base.yaml`: base detector training
- `train_fewshot.yaml`: single-reference fine-tuning
- `evaluate.yaml`: zero-shot and reference-guided evaluation targets
- `folds.yaml`: exact mapping from public fold numbers to metadata and checkpoint names

`seed` is intentionally `null` because the experiment scripts did not set a random seed. Dataset paths are repository-relative and the fold-specific source filenames are documented in `folds.yaml`.
