#!/usr/bin/env python3
"""Train the base B-RAD detector with the validated release schedule."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brad_det import load_config
from brad_det.training import (
    BestCheckpointState,
    add_backbone_parameter_group,
    build_cosine_scheduler,
    build_head_optimizer,
    build_model_and_anchors,
    build_training_loaders,
    evaluate_base_loss_and_iou,
    freeze_backbone_for_head_training,
    maybe_save_base_checkpoint,
    save_final_checkpoint,
    step_learning_rates,
    train_one_epoch,
    update_matching_thresholds,
)


def resolve_paths(config):
    for key in ("root", "train_metadata", "valid_metadata", "test_metadata"):
        config["data"][key] = str(PROJECT_ROOT / config["data"][key])
    config["output"]["checkpoint_dir"] = str(
        PROJECT_ROOT / config["output"]["checkpoint_dir"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "configs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config("base", fold=args.fold, config_dir=args.config_dir)
    resolve_paths(config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model, anchors, _ = build_model_and_anchors(config, device)
    freeze_backbone_for_head_training(model)
    train_loader, valid_loader = build_training_loaders(config, "base")
    training = config["training"]
    optimizer = build_head_optimizer(
        model,
        training["learning_rate"],
        training["weight_decay"],
    )
    scheduler = build_cosine_scheduler(
        optimizer,
        training["epochs"],
        training["min_learning_rate"],
    )

    legacy_final_name = config["legacy_fold"]["base_final_checkpoint"]
    final_path = Path(config["output"]["checkpoint_dir"]) / legacy_final_name
    best_path = final_path.with_name(legacy_final_name.replace("_final_", "_best_"))
    checkpoint_state = BestCheckpointState()
    iou_threshold = training["initial_iou_threshold"]
    ignore_threshold = training["initial_ignore_threshold"]
    ema_best_iou = 0.0

    for epoch in range(training["epochs"]):
        if epoch == training["unfreeze_epoch"]:
            add_backbone_parameter_group(
                model,
                optimizer,
                training["backbone_learning_rate"],
                training["num_unfreeze_blocks"],
            )
        train_losses = train_one_epoch(
            model,
            train_loader,
            anchors,
            optimizer,
            epoch,
            iou_threshold,
            ignore_threshold,
            device,
        )
        validation_losses, validation_ious = evaluate_base_loss_and_iou(
            model,
            valid_loader,
            anchors,
            iou_threshold,
            ignore_threshold,
            device,
            input_size=config["model"]["backbone"]["input_size"],
        )
        step_learning_rates(
            scheduler,
            optimizer,
            training["backbone_learning_rate"],
        )
        saved = maybe_save_base_checkpoint(
            model,
            best_path,
            checkpoint_state,
            validation_losses["total"],
            validation_ious.get("tight", 0.0),
        )
        iou_threshold, ignore_threshold, ema_best_iou = update_matching_thresholds(
            iou_threshold,
            ema_best_iou,
            checkpoint_state.best_validation_iou,
        )
        print(
            f"epoch={epoch + 1}/{training['epochs']} "
            f"train={train_losses['total']:.6f} "
            f"valid={validation_losses['total']:.6f} "
            f"tight_iou={validation_ious.get('tight', 0.0):.6f} "
            f"head_lr={optimizer.param_groups[0]['lr']:.8g} "
            f"saved_best={saved}",
            flush=True,
        )

    save_final_checkpoint(model, final_path)
    print(f"final checkpoint: {final_path}", flush=True)


if __name__ == "__main__":
    main()
