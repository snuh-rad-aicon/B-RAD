"""Optimizer and backbone unfreezing behavior used by B-RAD-Det training."""

from __future__ import annotations

import torch


def freeze_backbone_for_head_training(model) -> None:
    """Freeze the backbone and leave the detection head trainable."""
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True


def build_head_optimizer(model, learning_rate: float, weight_decay: float = 1e-4):
    """Create the initial head-only AdamW optimizer.

    The named parameter group and its position are checkpoint-relevant training
    state and intentionally match the original scripts.
    """
    return torch.optim.AdamW(
        [
            {
                "params": model.head.parameters(),
                "lr": learning_rate,
                "name": "head",
            }
        ],
        weight_decay=weight_decay,
    )


def add_backbone_parameter_group(
    model,
    optimizer,
    learning_rate: float,
    num_unfreeze_blocks: int = 12,
) -> None:
    """Unfreeze the last ViT blocks and append the original backbone group.

    The original implementation adds all backbone parameters to the optimizer,
    including parameters outside the selected blocks if fewer than twelve blocks
    are requested. Their ``requires_grad`` flags determine whether they update.
    """
    total_blocks = len(model.backbone.blocks)
    first_block = total_blocks - num_unfreeze_blocks
    if num_unfreeze_blocks <= 0 or first_block < 0:
        raise ValueError(
            "num_unfreeze_blocks must be between 1 and the number of backbone blocks"
        )

    for index in range(first_block, total_blocks):
        for parameter in model.backbone.blocks[index].parameters():
            parameter.requires_grad = True

    optimizer.add_param_group(
        {
            "params": model.backbone.parameters(),
            "lr": learning_rate,
            "name": "backbone",
        }
    )
