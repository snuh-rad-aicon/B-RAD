"""Checkpoint selection rules preserved from B-RAD-Det training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class BestCheckpointState:
    best_validation_loss: float = float("inf")
    best_validation_iou: float = 0.0


def maybe_save_base_checkpoint(
    model,
    path,
    state: BestCheckpointState,
    validation_loss: float,
    tight_iou: float,
) -> bool:
    """Save when tight IoU improves, matching base training."""
    if tight_iou <= state.best_validation_iou:
        return False
    state.best_validation_loss = validation_loss
    state.best_validation_iou = tight_iou
    _save_state_dict(model, path)
    return True


def maybe_save_fewshot_checkpoint(
    model,
    path,
    state: BestCheckpointState,
    validation_loss: float,
    tight_iou: float,
    reference_best_iou: float,
) -> bool:
    """Preserve the historical few-shot comparison/store asymmetry."""
    if reference_best_iou <= state.best_validation_iou:
        return False
    state.best_validation_loss = validation_loss
    state.best_validation_iou = (reference_best_iou + tight_iou) / 2
    _save_state_dict(model, path)
    return True


def save_final_checkpoint(model, path) -> None:
    """Write the final model state after the last epoch."""
    _save_state_dict(model, path)


def _save_state_dict(model, path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
