"""Learning-rate and matching-threshold schedules for B-RAD-Det."""

from __future__ import annotations

import torch


def build_cosine_scheduler(optimizer, num_epochs: int, min_learning_rate: float):
    """Construct the epoch-level cosine scheduler used by both training stages."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=min_learning_rate,
    )


def step_learning_rates(scheduler, optimizer, backbone_learning_rate: float) -> None:
    """Step cosine decay and restore the backbone group's fixed learning rate."""
    scheduler.step()
    head_learning_rate = scheduler.get_last_lr()[0]
    for parameter_group in optimizer.param_groups:
        if parameter_group.get("name") != "backbone":
            parameter_group["lr"] = head_learning_rate
        else:
            parameter_group["lr"] = backbone_learning_rate


def update_matching_thresholds(
    iou_threshold: float,
    ema_best_validation_iou: float,
    best_validation_iou: float,
) -> tuple[float, float, float]:
    """Apply the original EMA-based IoU and ignore-threshold update."""
    ema_best_validation_iou = (
        0.5 * ema_best_validation_iou + 0.5 * best_validation_iou
    )
    iou_threshold = max(iou_threshold, ema_best_validation_iou)
    ignore_threshold = iou_threshold - 0.35
    return iou_threshold, ignore_threshold, ema_best_validation_iou
