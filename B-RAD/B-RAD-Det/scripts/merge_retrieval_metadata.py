#!/usr/bin/env python3
"""Restore the retrieval columns expected by the original B-RAD code."""

import argparse
from pathlib import Path

import pandas as pd


def validate_references(references, key_columns, expected_ranks=100):
    required = {*key_columns, "rank", "reference_path"}
    missing = required - set(references.columns)
    if missing:
        raise ValueError(f"Reference table is missing columns: {sorted(missing)}")
    if references.duplicated([*key_columns, "rank"]).any():
        raise ValueError("Reference table contains duplicate query-key/rank rows")
    expected = list(range(1, expected_ranks + 1))
    for sample_id, group in references.groupby(key_columns, sort=False):
        ranks = sorted(group["rank"].astype(int).tolist())
        if ranks != expected:
            raise ValueError(f"Invalid ranks for {sample_id}: {ranks}")


def merge_standard(metadata, references):
    keys = ["image_path", "mask_path"]
    paths = (
        references.sort_values([*keys, "rank"])
        .groupby(keys, dropna=False)["reference_path"]
        .apply(list)
        .rename("seg_ref_imgs")
        .reset_index()
    )
    return metadata.merge(paths, on=keys, how="left", validate="one_to_one")


def merge_buscot(metadata, references):
    output = metadata.copy()
    grouped = references.sort_values(["sample_id", "rank"]).groupby("sample_id")
    output["seg_ref_imgs"] = output["sample_id"].map(
        grouped["reference_path"].apply(list)
    )
    indexed = references.set_index(["sample_id", "rank"])
    for rank in range(1, 101):
        paths = indexed.xs(rank, level="rank")["reference_path"]
        output[f"top{rank}_path"] = output["sample_id"].map(paths)
        if "similarity" in references:
            similarities = indexed.xs(rank, level="rank")["similarity"]
            output[f"top{rank}_sim"] = output["sample_id"].map(similarities)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--format", choices=("standard", "buscot"), required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    references = pd.read_csv(args.references)
    key_columns = ["image_path", "mask_path"] if args.format == "standard" else ["sample_id"]
    validate_references(references, key_columns)
    output = (
        merge_standard(metadata, references)
        if args.format == "standard"
        else merge_buscot(metadata, references)
    )
    if output.isna().any(axis=None):
        new_columns = set(output.columns) - set(metadata.columns)
        if output[list(new_columns)].isna().any(axis=None):
            raise ValueError("Retrieval merge produced missing values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
