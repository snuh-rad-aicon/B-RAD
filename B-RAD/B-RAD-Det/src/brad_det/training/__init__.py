"""Training utilities that preserve the original B-RAD-Det update order."""

from .optimizer import (
    add_backbone_parameter_group,
    build_head_optimizer,
    freeze_backbone_for_head_training,
)
from .schedules import (
    build_cosine_scheduler,
    step_learning_rates,
    update_matching_thresholds,
)
from .engine import train_base_batch, train_fewshot_batch, train_one_epoch
from .checkpointing import (
    BestCheckpointState,
    maybe_save_base_checkpoint,
    maybe_save_fewshot_checkpoint,
    save_final_checkpoint,
)
from .setup import build_model_and_anchors, build_training_loaders
from .validation import evaluate_base_loss_and_iou, evaluate_fewshot_loss_and_iou

__all__ = [
    "add_backbone_parameter_group",
    "build_head_optimizer",
    "freeze_backbone_for_head_training",
    "build_cosine_scheduler",
    "step_learning_rates",
    "update_matching_thresholds",
    "train_base_batch",
    "train_fewshot_batch",
    "train_one_epoch",
    "BestCheckpointState",
    "maybe_save_base_checkpoint",
    "maybe_save_fewshot_checkpoint",
    "save_final_checkpoint",
    "build_model_and_anchors",
    "build_training_loaders",
    "evaluate_base_loss_and_iou",
    "evaluate_fewshot_loss_and_iou",
]
