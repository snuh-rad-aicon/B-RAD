#!/usr/bin/env python3
"""Dataset-local entry point for separating retrieval references."""

import runpy
from pathlib import Path


runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "split_retrieval_metadata.py"),
    run_name="__main__",
)
