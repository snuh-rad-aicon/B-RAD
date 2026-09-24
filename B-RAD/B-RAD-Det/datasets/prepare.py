#!/usr/bin/env python3
"""Dataset-local entry point for converting one original dataset."""

import runpy
from pathlib import Path


runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_dataset.py"),
    run_name="__main__",
)
