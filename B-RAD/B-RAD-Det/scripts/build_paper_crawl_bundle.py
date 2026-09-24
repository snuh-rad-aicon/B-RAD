#!/usr/bin/env python3
"""Build the redistributable paper-crawl bundle from its public manifest."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path

import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_relative_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Manifest path must be project-relative: {value}")
    if not path.parts or path.parts[0] != "paper_crawl":
        raise ValueError(f"Unexpected paper-crawl path: {value}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    paths = []
    for column in ("image_path", "mask_path"):
        paths.extend(validate_relative_path(value) for value in metadata[column])
    paths = list(dict.fromkeys(paths))
    missing = [path for path in paths if not (args.data_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} manifest files: {missing[0]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as bundle:
        for index, relative_path in enumerate(paths, start=1):
            bundle.add(
                args.data_root / relative_path,
                arcname=relative_path.as_posix(),
                recursive=False,
            )
            if index % 500 == 0 or index == len(paths):
                print(f"packed: {index:,}/{len(paths):,}", flush=True)
    print(f"archive: {args.output}", flush=True)
    print(f"bytes: {args.output.stat().st_size:,}", flush=True)
    print(f"sha256: {sha256(args.output)}", flush=True)


if __name__ == "__main__":
    main()
