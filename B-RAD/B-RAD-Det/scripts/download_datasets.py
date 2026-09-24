#!/usr/bin/env python3
"""Download the public datasets used by B-RAD-Det with visible progress."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from urllib.parse import urlparse


DOWNLOADS = {
    "bus_bra": {
        "url": "https://zenodo.org/records/8231412/files/BUSBRA.zip?download=1",
        "filename": "BUSBRA.zip",
        "sha256": "ba3e6ed19cc37c682d8d39e25435bbf8a555a12cb7e641b5f2117685c95580ff",
    },
    "busi": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/sabahesaraki/breast-ultrasound-images-dataset",
        "filename": "Dataset_BUSI.zip",
    },
    "bus_uclm": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/orvile/bus-uclm-breast-ultrasound-dataset",
        "filename": "BUS-UCLM.zip",
        "sha256": "88ac46278b72477a55aafe7814ba5475c0e01c51df1a996958e58c4f55f7459b",
    },
    "busi_whu": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/orvile/busi-whu-breast-cancer-ultrasound-image-dataset",
        "filename": "BUS-WHU.zip",
        "sha256": "44d97241fe154e24670503b6fc363b9c6ec6a6ec6b73456a2d2b66a49f2c5d20",
    },
    "breast_usg": {
        "url": "https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip",
        "filename": "BrEaST-Lesions_USG.zip",
        "sha256": "c32e49cdfa54042e065582d0cb35978b5fc5e5b5e461b33bf773fb50cfdad355",
    },
    "udiat": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/jarintasnim090/udiat-data",
        "filename": "udiat-data.zip",
        "sha256": "12a6e04565494fecdd13904ce153f5dbd865d7a361589f48efef0f40752b7def",
    },
}

STU_REPOSITORY = "https://github.com/xbhlk/STU-Hospital.git"


def run(command):
    shown = list(command)
    if "--user" in shown:
        shown[shown.index("--user") + 1] = "<kaggle-credentials>"
    print("\n$", " ".join(str(part) for part in shown), flush=True)
    subprocess.run(command, check=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kaggle_credentials():
    """Return ``user:key`` for the Kaggle dataset API, or None when unset."""
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        return f"{user}:{key}"
    token = Path.home() / ".kaggle" / "kaggle.json"
    if token.is_file():
        data = json.loads(token.read_text(encoding="utf-8"))
        if data.get("username") and data.get("key"):
            return f'{data["username"]}:{data["key"]}'
    return None


def fetch(url, target):
    print(f"source: {url}", flush=True)
    print(f"target: {target}", flush=True)
    command = [
        "curl",
        "--location",
        "--fail",
        "--continue-at",
        "-",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "--progress-bar",
        "--output",
        str(target),
        url,
    ]
    is_kaggle = "kaggle.com/api" in url
    credentials = kaggle_credentials() if is_kaggle else None
    if credentials is not None:
        command[1:1] = ["--user", credentials]
    try:
        run(command)
    except subprocess.CalledProcessError:
        if not is_kaggle:
            raise
        raise SystemExit(
            f"Download failed: {url}\n"
            "Some Kaggle datasets reject unauthenticated requests. Export\n"
            "KAGGLE_USERNAME and KAGGLE_KEY, or place ~/.kaggle/kaggle.json, or\n"
            f"download the archive manually to {target}."
        ) from None
    print(f"downloaded: {target.stat().st_size:,} bytes", flush=True)


def extract_tar_gz(archive, extract_dir):
    root = extract_dir.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            destination = (root / member.name).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(f"Unsafe archive member: {member.name}")
        print(f"archive members: {len(members):,}", flush=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
        bundle.extractall(extract_dir)
    print(f"extracted: {extract_dir}", flush=True)


def download_tar_bundle(name, url, output_dir, extract_dir, expected_sha256, default_filename):
    parsed_name = Path(urlparse(url).path).name
    filename = parsed_name if parsed_name.endswith((".tar.gz", ".tgz")) else default_filename
    target = output_dir / filename
    print(f"\n[{name}]", flush=True)
    fetch(url, target)
    actual = sha256(target)
    print(f"sha256: {actual}", flush=True)
    if expected_sha256 and actual.lower() != expected_sha256.lower():
        raise ValueError(f"Checksum mismatch for {target}")
    extract_tar_gz(target, extract_dir)


def download(name, output_dir, verify):
    if name == "bus_stu":
        target = output_dir / "STU-Hospital"
        if target.exists():
            print(f"[skip] {target} already exists", flush=True)
        else:
            run(["git", "clone", "--progress", STU_REPOSITORY, str(target)])
        return

    spec = DOWNLOADS[name]
    target = output_dir / spec["filename"]
    print(f"\n[{name}]", flush=True)
    fetch(spec["url"], target)
    if verify:
        actual = sha256(target)
        expected = spec.get("sha256")
        print(f"sha256: {actual}", flush=True)
        if expected and actual != expected:
            raise ValueError(f"Checksum mismatch for {target}")
        run(["unzip", "-tq", str(target)])


def main():
    choices = [*DOWNLOADS, "bus_stu"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=[*choices, "ood_bundle", "paper_crawl_bundle", "all"],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/downloads"))
    parser.add_argument("--ood-url", help="URL of the hosted B-RAD-Det OOD tar.gz bundle")
    parser.add_argument("--ood-sha256", help="Expected SHA-256 of the OOD bundle")
    parser.add_argument("--ood-extract-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--paper-crawl-url", help="URL of the hosted paper-crawl tar.gz bundle")
    parser.add_argument("--paper-crawl-sha256", help="Expected SHA-256 of the paper-crawl bundle")
    parser.add_argument("--paper-crawl-extract-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    if shutil.which("curl") is None or shutil.which("unzip") is None:
        raise RuntimeError("curl and unzip are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = choices if "all" in args.datasets else args.datasets
    for name in selected:
        if name == "ood_bundle":
            if not args.ood_url:
                parser.error("--ood-url is required when ood_bundle is selected")
            download_tar_bundle(
                "ood_bundle",
                args.ood_url,
                args.output_dir,
                args.ood_extract_dir,
                None if args.no_verify else args.ood_sha256,
                "B-RAD-Det-OOD.tar.gz",
            )
        elif name == "paper_crawl_bundle":
            if not args.paper_crawl_url:
                parser.error(
                    "--paper-crawl-url is required when paper_crawl_bundle is selected"
                )
            download_tar_bundle(
                "paper_crawl_bundle",
                args.paper_crawl_url,
                args.output_dir,
                args.paper_crawl_extract_dir,
                None if args.no_verify else args.paper_crawl_sha256,
                "B-RAD-Det-paper-crawl.tar.gz",
            )
        else:
            download(name, args.output_dir, verify=not args.no_verify)


if __name__ == "__main__":
    main()
