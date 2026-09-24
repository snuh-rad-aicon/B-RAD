#!/usr/bin/env python3
"""Dataset-local entry point for the verbose public-data downloader."""

import runpy
from pathlib import Path


runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "download_datasets.py"),
    run_name="__main__",
)
