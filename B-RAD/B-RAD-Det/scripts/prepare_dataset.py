#!/usr/bin/env python3
"""Prepare a public ultrasound dataset for B-RAD-Det."""

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from brad_det.data import DATASET_NAMES, prepare_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-metadata", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    prepared = prepare_dataset(
        dataset=args.dataset,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        manifest=args.manifest,
        copy_files=not args.validate_only,
    )
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output_metadata, index=False)
    def normalize_box(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        return ast.literal_eval(value) if isinstance(value, str) else value

    mismatches = prepared[
        prepared.apply(
            lambda row: normalize_box(row["bbox_xyxy"])
            != normalize_box(row["computed_bbox_xyxy"]),
            axis=1,
        )
    ]
    print(f"Prepared {len(prepared)} rows for {args.dataset}")
    print(f"BBox mismatches requiring review: {len(mismatches)}")


if __name__ == "__main__":
    main()
