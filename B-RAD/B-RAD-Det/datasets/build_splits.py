#!/usr/bin/env python3
"""Materialize the paper train, validation, and OOD test CSV files."""

import argparse
import ast
from pathlib import Path

import pandas as pd


def memberships(value):
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise ValueError(f"Invalid split_multi value: {value!r}")
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--stage",
        choices=("base", "fewshot", "both"),
        default="both",
    )
    parser.add_argument(
        "--fewshot-splits",
        type=Path,
        help="Few-shot split manifest; defaults beside --metadata",
    )
    parser.add_argument(
        "--fewshot-references",
        type=Path,
        help="Fold-specific selected references; defaults beside --metadata",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.metadata)
    frame = frame[frame["label"].str.lower() != "normal"].copy().reset_index(drop=True)
    duplicate_uids = frame["uid"].value_counts()
    duplicate_uids = set(duplicate_uids[duplicate_uids > 1].index)
    deduplicated = frame[~frame["uid"].isin(duplicate_uids)].reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"base", "both"}:
        base_output = args.output_dir / "base" if args.stage == "both" else args.output_dir
        base_output.mkdir(parents=True, exist_ok=True)
        for fold in range(5):
            for role in ("train", "valid"):
                fold_frame = frame if (fold == 0 and role == "train") else deduplicated
                split_lists = fold_frame["split_multi"].map(memberships)
                split_name = f"fold{fold}_{role}"
                selected = fold_frame[
                    split_lists.map(lambda values: split_name in values)
                ]
                output = base_output / f"metadata_{role}_fold{fold}.csv"
                selected.to_csv(output, index=False)
                print(f"base {split_name}: {len(selected)} rows -> {output}")

    if args.stage in {"fewshot", "both"}:
        split_path = args.fewshot_splits or args.metadata.with_name(
            "fewshot_splits.csv"
        )
        split_frame = pd.read_csv(split_path)
        split_frame["split_multi"] = split_frame["split_multi"].map(memberships)
        joined = frame.drop(columns=["split_multi"]).merge(
            split_frame,
            on=["image_path", "mask_path"],
            how="inner",
            validate="one_to_one",
        )
        fewshot_output = (
            args.output_dir / "fewshot" if args.stage == "both" else args.output_dir
        )
        fewshot_output.mkdir(parents=True, exist_ok=True)
        evaluation_output = args.output_dir / "evaluation"
        evaluation_output.mkdir(parents=True, exist_ok=True)
        reference_path = args.fewshot_references or args.metadata.with_name(
            "fewshot_references.csv"
        )
        reference_frame = pd.read_csv(reference_path)
        for fold in range(5):
            for role in ("train", "valid"):
                split_name = f"fold{fold}_{role}"
                selected = joined[
                    joined["split_multi"].map(
                        lambda values: split_name in values
                    )
                ]
                references = reference_frame[
                    reference_frame["split"] == split_name
                ]
                selected = selected.drop(
                    columns=["seg_ref_imgs"], errors="ignore"
                ).merge(
                    references[["image_path", "mask_path", "reference_path"]],
                    on=["image_path", "mask_path"],
                    how="left",
                    validate="one_to_one",
                )
                if selected["reference_path"].isna().any():
                    raise ValueError(
                        f"Missing few-shot reference for {split_name}"
                    )
                selected["seg_ref_imgs"] = selected["reference_path"].map(
                    lambda value: repr([value])
                )
                selected = selected.drop(columns=["reference_path"])
                output = fewshot_output / f"metadata_{role}_fold{fold}.csv"
                selected.to_csv(output, index=False)
                print(f"fewshot {split_name}: {len(selected)} rows -> {output}")
                if role == "valid":
                    evaluation = joined[
                        joined["split_multi"].map(
                            lambda values: split_name in values
                        )
                    ]
                    evaluation_path = (
                        evaluation_output / f"metadata_valid_fold{fold}.csv"
                    )
                    evaluation.to_csv(evaluation_path, index=False)
                    print(
                        f"evaluation {split_name}: {len(evaluation)} rows "
                        f"-> {evaluation_path}"
                    )

    split_lists = frame["split_multi"].map(memberships)
    test = frame[split_lists.map(lambda values: any(x.startswith("test_") for x in values))]
    output = args.output_dir / "metadata_test.csv"
    test.to_csv(output, index=False)
    print(f"test: {len(test)} rows -> {output}")


if __name__ == "__main__":
    main()
