#!/usr/bin/env python3
"""Train one-shot reference-guided B-RAD-Det."""

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
    evaluate_fewshot_loss_and_iou,
    freeze_backbone_for_head_training,
    maybe_save_fewshot_checkpoint,
    save_final_checkpoint,
    step_learning_rates,
    train_one_epoch,
    update_matching_thresholds,
)


def resolve_paths(config):
    for key in (
        "root",
        "train_metadata",
        "valid_metadata",
        "ood_metadata",
        "paper_crawl_metadata",
    ):
        config["data"][key] = str(PROJECT_ROOT / config["data"][key])
    config["checkpoint"]["reload"] = str(
        PROJECT_ROOT / config["checkpoint"]["reload"]
    )
    config["output"]["checkpoint_dir"] = str(
        PROJECT_ROOT / config["output"]["checkpoint_dir"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "configs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config("fewshot", fold=args.fold, config_dir=args.config_dir)
    resolve_paths(config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model, anchors, _ = build_model_and_anchors(config, device)
    base_checkpoint = (
        Path(config["checkpoint"]["reload"])
        / config["legacy_fold"]["base_final_checkpoint"]
    )
    state_dict = torch.load(base_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    freeze_backbone_for_head_training(model)
    train_loader, valid_loader = build_training_loaders(config, "fewshot")
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

    legacy_best_name = config["legacy_fold"]["fewshot_best_checkpoint"]
    best_path = Path(config["output"]["checkpoint_dir"]) / legacy_best_name
    final_path = best_path.with_name(legacy_best_name.replace("_best_", "_final_"))
    checkpoint_state = BestCheckpointState()
    iou_threshold = training["initial_iou_threshold"]
    ignore_threshold = training["initial_ignore_threshold"]
    ema_best_iou = 0.0
    reference_shots = training["reference_shots"]

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
            num_ref_shot=reference_shots,
        )
        validation_losses, validation_ious = evaluate_fewshot_loss_and_iou(
            model,
            valid_loader,
            anchors,
            iou_threshold,
            ignore_threshold,
            reference_shots,
            device,
            input_size=config["model"]["backbone"]["input_size"],
        )
        step_learning_rates(
            scheduler,
            optimizer,
            training["backbone_learning_rate"],
        )
        saved = maybe_save_fewshot_checkpoint(
            model,
            best_path,
            checkpoint_state,
            validation_losses["total"],
            validation_ious.get("tight", 0.0),
            validation_ious.get("ref_best", 0.0),
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
            f"ref_best_iou={validation_ious.get('ref_best', 0.0):.6f} "
            f"head_lr={optimizer.param_groups[0]['lr']:.8g} "
            f"saved_best={saved}",
            flush=True,
        )

    save_final_checkpoint(model, final_path)
    print(f"final checkpoint: {final_path}", flush=True)


if __name__ == "__main__":
    main()
