"""Configuration loading and validation for B-RAD-Det."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a B-RAD-Det configuration is missing or invalid."""


_STAGE_FILES = {
    "base": "train_base.yaml",
    "fewshot": "train_fewshot.yaml",
    "evaluate": "evaluate.yaml",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)

    if not isinstance(value, dict):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _format_fold(value: Any, fold: int) -> Any:
    if isinstance(value, str):
        return value.format(fold=fold)
    if isinstance(value, list):
        return [_format_fold(item, fold) for item in value]
    if isinstance(value, dict):
        return {key: _format_fold(item, fold) for key, item in value.items()}
    return value


def _require(config: Mapping[str, Any], path: str) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"Missing required configuration value: {path}")
        current = current[part]
    return current


def _require_positive(config: Mapping[str, Any], path: str) -> None:
    value = _require(config, path)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"Configuration value must be positive: {path}={value!r}")


def _validate_common(config: Mapping[str, Any]) -> None:
    _require(config, "model.backbone.name")
    _require_positive(config, "model.backbone.input_size")
    _require_positive(config, "model.backbone.patch_size")
    _require_positive(config, "model.detector.base_num_classes")
    _require_positive(config, "model.anchors.stride")

    scales = _require(config, "model.anchors.scales")
    ratios = _require(config, "model.anchors.aspect_ratios")
    if not isinstance(scales, list) or not scales or any(value <= 0 for value in scales):
        raise ConfigError("model.anchors.scales must be a non-empty list of positive values")
    if not isinstance(ratios, list) or not ratios or any(value <= 0 for value in ratios):
        raise ConfigError(
            "model.anchors.aspect_ratios must be a non-empty list of positive values"
        )


def _validate_training(config: Mapping[str, Any], stage: str) -> None:
    for path in (
        "training.epochs",
        "training.batch_size",
        "training.num_workers",
        "training.learning_rate",
        "training.min_learning_rate",
        "training.backbone_learning_rate",
        "training.weight_decay",
        "training.unfreeze_epoch",
        "training.num_unfreeze_blocks",
        "data.root",
        "data.train_metadata",
        "data.valid_metadata",
    ):
        if path in {"training.unfreeze_epoch", "training.num_workers"}:
            value = _require(config, path)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ConfigError(
                    f"Configuration value must be a non-negative integer: {path}={value!r}"
                )
        elif path.startswith("training"):
            _require_positive(config, path)
        else:
            _require(config, path)

    if stage == "fewshot":
        _require_positive(config, "training.reference_shots")
        _require(config, "checkpoint.reload")
        _require(config, "data.paper_crawl_metadata")


def _validate_evaluation(config: Mapping[str, Any]) -> None:
    _require_positive(config, "evaluation.dataloader_batch_size")
    _require_positive(config, "evaluation.anchor_batch_capacity")
    shots = _require(config, "evaluation.reference_shots")
    if not isinstance(shots, list) or not shots or any(value < 0 for value in shots):
        raise ConfigError(
            "evaluation.reference_shots must be a non-empty list of non-negative integers"
        )
    _require(config, "checkpoint.base_directory")
    _require(config, "checkpoint.fewshot_directory")
    _require(config, "reference_data.root")
    _require(config, "targets")


def load_config(
    stage: str,
    fold: int = 0,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load, merge, resolve, and validate a B-RAD-Det configuration.

    Args:
        stage: One of ``base``, ``fewshot``, or ``evaluate``.
        fold: Public fold index in the inclusive range 0 through 4.
        config_dir: Directory containing the B-RAD-Det YAML files. When omitted,
            the repository's ``configs`` directory is used.

    Returns:
        A new dictionary containing the resolved configuration and legacy fold
        provenance under ``legacy_fold``.
    """
    if stage not in _STAGE_FILES:
        choices = ", ".join(sorted(_STAGE_FILES))
        raise ConfigError(f"Unknown stage {stage!r}; expected one of: {choices}")
    if not isinstance(fold, int) or isinstance(fold, bool) or fold not in range(5):
        raise ConfigError(f"fold must be an integer from 0 through 4, got {fold!r}")

    default_dir = Path(__file__).resolve().parents[2] / "configs"
    root = Path(config_dir) if config_dir is not None else default_dir

    model_config = _read_yaml(root / "model.yaml")
    stage_config = _read_yaml(root / _STAGE_FILES[stage])
    fold_config = _read_yaml(root / "folds.yaml")

    folds = _require(fold_config, "folds")
    if fold not in folds:
        raise ConfigError(f"Missing legacy mapping for fold {fold}: {root / 'folds.yaml'}")

    config = _deep_merge(model_config, stage_config)
    config.setdefault("experiment", {})["fold"] = fold
    config["legacy_fold"] = deepcopy(folds[fold])
    config = _format_fold(config, fold)

    _validate_common(config)
    if stage in {"base", "fewshot"}:
        _validate_training(config, stage)
    else:
        _validate_evaluation(config)
    return config
