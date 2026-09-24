#!/usr/bin/env python3
"""Separate query metadata from ranked retrieval references."""

import argparse
import ast
from pathlib import Path

import pandas as pd


def require_relative(value, field):
    path = Path(str(value))
    if path.is_absolute():
        raise ValueError(f"{field} must be relative, got: {value}")
    return path.as_posix()


def split_standard(frame):
    records = []
    for _, row in frame.iterrows():
        references = ast.literal_eval(row["seg_ref_imgs"])
        if len(references) != 100:
            raise ValueError(f"{row['uid']} has {len(references)} references, expected 100")
        for rank, reference in enumerate(references, start=1):
            records.append(
                {
                    "image_path": require_relative(row["image_path"], "image_path"),
                    "mask_path": require_relative(row["mask_path"], "mask_path"),
                    "rank": rank,
                    "reference_path": require_relative(reference, "reference_path"),
                }
            )
    return frame.drop(columns=["seg_ref_imgs"]), pd.DataFrame(records)


def split_buscot(frame):
    records = []
    reference_columns = []
    for rank in range(1, 101):
        path_column = f"top{rank}_path"
        similarity_column = f"top{rank}_sim"
        reference_columns.extend([path_column, similarity_column])
        for _, row in frame.iterrows():
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "rank": rank,
                    "reference_path": require_relative(row[path_column], path_column),
                    "similarity": row[similarity_column],
                }
            )
    drop_columns = ["seg_ref_imgs", *reference_columns]
    return frame.drop(columns=drop_columns), pd.DataFrame(records)


def validate_query_paths(frame):
    for column in ("image_path", "mask_path", "bbox_path"):
        if column not in frame:
            continue
        for value in frame[column].dropna():
            require_relative(value, column)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--references-output", required=True, type=Path)
    parser.add_argument("--format", choices=("standard", "buscot"), required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    validate_query_paths(frame)
    if args.format == "standard":
        metadata, references = split_standard(frame)
    else:
        metadata, references = split_buscot(frame)

    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.references_output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.metadata_output, index=False)
    references.to_csv(args.references_output, index=False)
    print(f"Wrote {len(metadata)} query rows to {args.metadata_output}")
    print(f"Wrote {len(references)} reference rows to {args.references_output}")


if __name__ == "__main__":
    main()
