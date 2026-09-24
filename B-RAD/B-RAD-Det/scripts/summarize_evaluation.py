#!/usr/bin/env python3
"""Aggregate five fold-level B-RAD-Det metric files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGETS = ("id_validation", "ood_test", "buscot")
TARGET_LABELS = {
    "id_validation": "ID",
    "ood_test": "BrEaST/STU/UDIAT",
    "buscot": "BUS-CoT",
}


def summarize_target(result_dir, target, shots):
    names = ("zero_mDSC", "zero_mIoU") if shots == 0 else ("kshot_mDSC", "kshot_mIoU")
    fold_values = []
    for fold in range(5):
        path = result_dir / f"{target}_{shots}shot_fold{fold}_metrics.json"
        fold_values.append(json.loads(path.read_text(encoding="utf-8")))
    return {
        name: {
            "mean": float(np.mean([row[name] for row in fold_values])),
            "std": float(np.std([row[name] for row in fold_values], ddof=1)),
        }
        for name in names
    }


def paper_value(metric):
    return f"{metric['mean']:.3f} ± {metric['std']:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=(*TARGETS, "all"), required=True)
    parser.add_argument("--shots", type=int, choices=(0, 4), required=True)
    parser.add_argument("--result-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    result_dir = (
        args.result_dir
        if args.result_dir.is_absolute()
        else PROJECT_ROOT / args.result_dir
    )
    targets = TARGETS if args.target == "all" else (args.target,)
    result = {
        target: summarize_target(result_dir, target, args.shots)
        for target in targets
    }
    result["folds"] = 5
    output = result_dir / f"{args.target}_{args.shots}shot_5fold_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    metric_prefix = "zero" if args.shots == 0 else "kshot"
    setting = "0-shot" if args.shots == 0 else "k-shot (k=4)"
    print("\nB-RAD-Det: Table 5 format (5-fold mean ± std)")
    print("| Task | Setting | Model | Cohort | mDSC | mIoU |")
    print("|---|---|---|---|---:|---:|")
    for target in targets:
        metrics = result[target]
        print(
            f"| Detection | {setting} | B-RAD-Det | {TARGET_LABELS[target]} | "
            f"{paper_value(metrics[metric_prefix + '_mDSC'])} | "
            f"{paper_value(metrics[metric_prefix + '_mIoU'])} |"
        )
    print(f"\nSaved JSON: {output}")


if __name__ == "__main__":
    main()
