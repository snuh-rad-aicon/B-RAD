#!/usr/bin/env python3
"""Dataset-local entry point for restoring original retrieval columns."""

import runpy
from pathlib import Path


runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "merge_retrieval_metadata.py"),
    run_name="__main__",
)
