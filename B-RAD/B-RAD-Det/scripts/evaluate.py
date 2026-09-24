#!/usr/bin/env python3
"""Evaluate a fold for the B-RAD paper detector results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brad_det import load_config
from brad_det.datasets import EvaluationDetectionDataset
from brad_det.evaluation import evaluate_detector
from brad_det.training import build_model_and_anchors


def resolve(project_root, value):
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("id_validation", "ood_test", "buscot"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--shots", type=int, choices=(0, 4), default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "configs")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        help="Optional local root containing data/ and models/ without changing release paths",
    )
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    config = load_config("evaluate", fold=args.fold, config_dir=args.config_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    runtime_root = args.runtime_root.resolve() if args.runtime_root else PROJECT_ROOT

    def resolve_input(value):
        runtime_path = resolve(runtime_root, value)
        if runtime_path.exists():
            return runtime_path
        return resolve(PROJECT_ROOT, value)

    metadata = resolve_input(config["targets"][args.target]["metadata"])
    data_root = resolve_input(config["reference_data"]["root"])
    reference_metadata = [
        resolve_input(config["reference_data"][name])
        for name in ("valid_metadata", "ood_metadata", "train_metadata", "paper_crawl_metadata")
    ]
    reference_columns = None
    if args.target == "buscot":
        reference_columns = [f"top{rank}_path" for rank in range(1, 101)]
    dataset = EvaluationDetectionDataset(
        metadata=metadata,
        image_root=data_root,
        input_size=config["model"]["backbone"]["input_size"],
        long_bbox=True,
        margin_bbox_ratio=config["model"]["detector"]["margin_box_ratio"],
        num_ref_shot=args.shots,
        reference_metadata=reference_metadata,
        is_inference=True,
        reference_path_columns=reference_columns,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    config["training"] = {"batch_size": config["evaluation"]["anchor_batch_capacity"]}
    model, anchors, _ = build_model_and_anchors(config, device)
    if args.shots == 0:
        checkpoint_name = config["legacy_fold"]["base_best_checkpoint"]
        checkpoint_directory = config["checkpoint"]["base_directory"]
    else:
        checkpoint_name = config["legacy_fold"]["fewshot_best_checkpoint"]
        checkpoint_directory = config["checkpoint"]["fewshot_directory"]
    checkpoint = resolve_input(checkpoint_directory) / checkpoint_name
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    summary, rows = evaluate_detector(
        model, loader, anchors, device,
        input_size=config["model"]["backbone"]["input_size"],
        reference_shots=args.shots,
        progress_description=(
            f"{args.target} fold={args.fold} shots={args.shots}"
        ),
    )
    for row in rows:
        row["image_path"] = str(Path(row["image_path"]).relative_to(data_root))
    output_dir = resolve(PROJECT_ROOT, config["output"]["result_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.target}_{args.shots}shot_fold{args.fold}"
    summary_path = output_dir / f"{stem}_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.save_predictions:
        pd.DataFrame(rows).to_csv(output_dir / f"{stem}_predictions.csv", index=False)
    metric_prefix = "zero" if args.shots == 0 else "kshot"
    setting = "0-shot" if args.shots == 0 else "k-shot (k=4)"
    print("\nB-RAD-Det: fold result")
    print("| Task | Setting | Model | Target | Fold | n | mDSC | mIoU |")
    print("|---|---|---|---|---:|---:|---:|---:|")
    print(
        f"| Detection | {setting} | B-RAD-Det | {args.target} | "
        f"{args.fold} | {summary['count']} | "
        f"{summary[metric_prefix + '_mDSC']:.4f} | "
        f"{summary[metric_prefix + '_mIoU']:.4f} |",
        flush=True,
    )
    print(f"\nSaved JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
