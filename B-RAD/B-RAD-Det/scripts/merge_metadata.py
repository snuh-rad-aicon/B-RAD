#!/usr/bin/env python3
"""Merge converted samples, paper splits, and released MedGemma text."""

import argparse
import ast
from pathlib import Path

import pandas as pd


def read_unique_table(path, keys, value_columns, name):
    frame = pd.read_csv(path, usecols=[*keys, *value_columns])
    conflicts = frame.groupby(keys, dropna=False)[value_columns].nunique(dropna=False)
    conflicts = conflicts[(conflicts > 1).any(axis=1)]
    if not conflicts.empty:
        raise ValueError(f"Conflicting {name} rows for {len(conflicts)} keys")
    return frame.drop_duplicates(subset=keys)


def validate_split(value):
    try:
        entries = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError) as error:
        raise ValueError(f"Invalid split_multi value: {value!r}") from error
    if not isinstance(entries, list) or not all(isinstance(item, str) for item in entries):
        raise ValueError(f"split_multi must be a list of strings: {value!r}")
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--medgemma-text", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    samples = pd.read_csv(args.samples)
    required = {"image_path", "mask_path", "bbox_xyxy", "source_dataset"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"Samples table is missing columns: {sorted(missing)}")
    if samples.duplicated(["image_path", "mask_path"]).any():
        raise ValueError("Samples table contains duplicate image_path/mask_path pairs")

    splits = read_unique_table(args.splits, ["image_path"], ["split_multi"], "split")
    text = read_unique_table(
        args.medgemma_text,
        ["image_path", "mask_path"],
        ["medgemma_txt"],
        "MedGemma text",
    )
    merged = samples.merge(splits, on="image_path", how="left", validate="many_to_one")
    merged = merged.merge(
        text,
        on=["image_path", "mask_path"],
        how="left",
        validate="one_to_one",
    )

    missing_split = int(merged["split_multi"].isna().sum())
    missing_text = int(merged["medgemma_txt"].isna().sum())
    if missing_split or missing_text:
        raise ValueError(
            f"Merge is incomplete: missing split={missing_split}, text={missing_text}"
        )
    merged["split_multi"].map(validate_split)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {len(merged)} rows to {args.output}")


if __name__ == "__main__":
    main()
