"""Convert original public datasets to the paths used by B-RAD metadata."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


DATASET_NAMES = (
    "bus_bra",
    "busi",
    "bus_uclm",
    "busi_whu",
    "udiat",
    "bus_stu",
    "breast_usg",
)
SOURCE_NAMES = {
    "bus_bra": "BUSBRA",
    "busi": "BUSI",
    "bus_uclm": "BUS-UCLM",
    "busi_whu": "BUS-WHU",
    "udiat": "UDIAT-DatasetB",
    "bus_stu": "Breast-STU",
    "breast_usg": "Breast-Lesions-USG",
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def natural_key(path):
    """Return a stable natural-sort key for dataset filenames."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def image_files(root):
    return sorted(
        (
            path
            for path in Path(root).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=natural_key,
    )


def is_mask(path):
    name = path.stem.lower()
    parts = {part.lower() for part in path.parts}
    return (
        "mask" in parts
        or "masks" in parts
        or "ground truth" in parts
        or "ground_truth" in parts
        or name.startswith("mask_")
        or "_mask" in name
        or re.search(r"_(tumor|other\d*)$", name) is not None
        or name.endswith("_anno")
    )


def build_basename_index(paths):
    index = {}
    for path in paths:
        index.setdefault(path.name.lower(), []).append(path)
    return index


def unique_basename(index, filename, kind):
    matches = index.get(filename.lower(), [])
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {kind} named {filename!r}, found {len(matches)}"
        )
    return matches[0]


def bbox_from_mask(mask_path, maximum_exclusive=False):
    """Calculate the inclusive-maximum xyxy box used by the paper metadata."""
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        color = mask[..., :3] if mask.shape[-1] == 4 else mask
        foreground = np.any(color != 0, axis=-1)
    else:
        foreground = mask != 0
    ys, xs = np.nonzero(foreground)
    if len(xs) == 0:
        return None
    offset = 1 if maximum_exclusive else 0
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() + offset),
        int(ys.max() + offset),
    ]


def write_standard_mask(source, target):
    """Write the existing B-RAD annotation format: uint8 L with values 0/255."""
    mask = np.asarray(Image.open(source))
    if mask.ndim == 3:
        color = mask[..., :3] if mask.shape[-1] == 4 else mask
        foreground = np.any(color != 0, axis=-1)
    else:
        foreground = mask != 0
    Image.fromarray(foreground.astype(np.uint8) * 255, mode="L").save(target)


def materialize_pair(dataset, source_image, source_mask, target_image, target_mask):
    """Write files using the exact preprocessing used by the current merged tree."""
    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_mask.parent.mkdir(parents=True, exist_ok=True)
    if source_mask is None:
        if dataset == "busi":
            Image.open(source_image).convert("L").save(target_image)
        elif not target_image.exists():
            shutil.copy2(source_image, target_image)
        width, height = Image.open(source_image).size
        Image.new("L", (width, height), color=0).save(target_mask)
        return
    if dataset == "bus_stu":
        Image.open(source_image).resize((256, 256), Image.BILINEAR).save(target_image)
        Image.open(source_mask).resize((256, 256), Image.NEAREST).save(target_mask)
        return
    if dataset in {"breast_usg", "bus_uclm", "udiat"}:
        Image.open(source_image).convert("L").save(target_image)
        write_standard_mask(source_mask, target_mask)
        return
    if dataset == "busi":
        Image.open(source_image).convert("L").save(target_image)
        if source_mask is None:
            width, height = Image.open(source_image).size
            Image.new("L", (width, height), color=0).save(target_mask)
        else:
            write_standard_mask(source_mask, target_mask)
        return
    if not target_image.exists():
        shutil.copy2(source_image, target_image)
    write_standard_mask(source_mask, target_mask)


def parse_target(path):
    return Path(str(path)).name


def resolve_bus_bra(row, image_index, mask_index, _catalog):
    image_name = str(row["filename"])
    image = unique_basename(image_index, image_name, "BUS-BRA image")
    candidates = [
        re.sub(r"^bus_", "mask_", image_name, flags=re.IGNORECASE),
        f"{Path(image_name).stem}_mask{Path(image_name).suffix}",
    ]
    mask = next(
        (
            mask_index[name.lower()][0]
            for name in candidates
            if len(mask_index.get(name.lower(), [])) == 1
        ),
        None,
    )
    if mask is None:
        raise FileNotFoundError(f"Mask not found for BUS-BRA image {image_name!r}")
    return image, mask


def resolve_busi_whu(row, image_index, mask_index, _catalog):
    image_name = str(row["filename"])
    image = unique_basename(image_index, image_name, "BUSI-WHU image")
    stem, suffix = Path(image_name).stem, Path(image_name).suffix
    mask = unique_basename(mask_index, f"{stem}_anno{suffix}", "BUSI-WHU mask")
    return image, mask


def resolve_bus_uclm(row, image_index, mask_index, _catalog):
    target = parse_target(row["image_path"])
    match = re.fullmatch(r"busuclm_(.+)", Path(target).stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Unexpected BUS-UCLM target name: {target}")
    source_name = f"{match.group(1)}{Path(target).suffix}"
    image = unique_basename(image_index, source_name, "BUS-UCLM image")
    mask = unique_basename(mask_index, source_name, "BUS-UCLM mask")
    return image, mask


def busi_catalog(paths):
    by_label = {"benign": [], "malignant": [], "normal": []}
    pattern = re.compile(
        r"^(benign|malignant|normal)\s*\((\d+)\)$",
        re.IGNORECASE,
    )
    for path in paths:
        match = pattern.fullmatch(path.stem)
        if match and not is_mask(path):
            by_label[match.group(1).lower()].append(path)
    catalog = {}
    for label, label_paths in by_label.items():
        for index, path in enumerate(sorted(label_paths, key=lambda item: item.name)):
            catalog[(label, index)] = path
    return catalog


def resolve_busi(row, _image_index, mask_index, catalog):
    target = Path(parse_target(row["image_path"])).stem
    match = re.fullmatch(r"busi_(benign|malignant|normal)_(\d+)", target)
    if match is None:
        raise ValueError(f"Unexpected BUSI target name: {target}")
    label, zero_based_index = match.group(1), int(match.group(2))
    image = catalog.get((label, zero_based_index))
    if image is None:
        raise FileNotFoundError(
            f"BUSI source image not found for {label} index {zero_based_index}"
        )

    if label == "normal":
        return image, None

    target_mask = Path(parse_target(row["mask_path"])).stem
    mask_match = re.search(r"_mask(\d+)$", target_mask)
    mask_index_value = int(mask_match.group(1)) if mask_match else 0
    source_stem = image.stem
    candidates = (
        [f"{source_stem}_mask{image.suffix}"]
        if mask_index_value == 0
        else [f"{source_stem}_mask_{mask_index_value}{image.suffix}"]
    )
    mask = next(
        (
            mask_index[name.lower()][0]
            for name in candidates
            if len(mask_index.get(name.lower(), [])) == 1
        ),
        None,
    )
    if mask is None:
        raise FileNotFoundError(f"BUSI mask not found for {image.name!r}")
    return image, mask


def resolve_bus_stu(row, image_index, mask_index, _catalog):
    target = Path(parse_target(row["image_path"])).stem
    match = re.fullmatch(r"breast_stu_(\d+)", target, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Unexpected BUS-STU target name: {target}")
    number = match.group(1)
    image = unique_basename(
        image_index,
        f"Test_Image_{number}.png",
        "BUS-STU image",
    )
    mask = unique_basename(mask_index, f"mask_{number}.png", "BUS-STU mask")
    return image, mask


def resolve_udiat(row, _image_index, _mask_index, catalog):
    filename = str(row["filename"])
    label = str(row["label"]).lower()
    image_dir = "Benign" if label == "benign" else "Malignant"
    image = catalog["root"] / image_dir / filename
    mask = catalog["root"] / "all_masks" / filename
    if not image.is_file():
        raise FileNotFoundError(f"UDIAT image not found: {image}")
    if not mask.is_file():
        raise FileNotFoundError(f"UDIAT mask not found: {mask}")
    return image, mask


def resolve_breast_usg(row, image_index, mask_index, _catalog):
    match = re.fullmatch(r"usg_(\d+)", str(row["uid"]), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Unexpected BrEaST uid: {row['uid']}")
    case_number = int(match.group(1)) + 1
    image = unique_basename(
        image_index,
        f"case{case_number:03d}.png",
        "BrEaST image",
    )
    mask = unique_basename(
        mask_index,
        f"case{case_number:03d}_tumor.png",
        "BrEaST tumor mask",
    )
    return image, mask


RESOLVERS = {
    "bus_bra": resolve_bus_bra,
    "busi": resolve_busi,
    "bus_uclm": resolve_bus_uclm,
    "busi_whu": resolve_busi_whu,
    "udiat": resolve_udiat,
    "bus_stu": resolve_bus_stu,
    "breast_usg": resolve_breast_usg,
}


def prepare_dataset(dataset, raw_dir, output_dir, manifest, copy_files=True):
    """Materialize one dataset according to the released B-RAD paper manifest."""
    if dataset not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {DATASET_NAMES}")

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    frame = pd.read_csv(manifest)
    frame = frame[
        frame["source_dataset"].str.lower() == SOURCE_NAMES[dataset].lower()
    ].copy()
    if frame.empty:
        raise ValueError(f"No {SOURCE_NAMES[dataset]} rows found in {manifest}")

    files = image_files(raw_dir)
    images = [path for path in files if not is_mask(path)]
    masks = [path for path in files if is_mask(path)]
    image_index = build_basename_index(images)
    mask_index = build_basename_index(masks)
    if dataset == "busi":
        catalog = busi_catalog(images)
    elif dataset == "udiat":
        catalog = {"root": raw_dir}
    else:
        catalog = None

    records = []
    resolver = RESOLVERS[dataset]
    for _, row in frame.iterrows():
        source_image, source_mask = resolver(
            row,
            image_index,
            mask_index,
            catalog,
        )
        target_image = output_dir / str(row["image_path"])
        target_mask = output_dir / str(row["mask_path"])
        if copy_files:
            materialize_pair(
                dataset,
                source_image,
                source_mask,
                target_image,
                target_mask,
            )

        records.append(
            {
                **row.to_dict(),
                "source_image_path": str(source_image.relative_to(raw_dir)),
                "source_mask_path": (
                    str(source_mask.relative_to(raw_dir))
                    if source_mask is not None
                    else None
                ),
                "computed_bbox_xyxy": (
                    bbox_from_mask(
                        target_mask if copy_files else source_mask,
                        maximum_exclusive=dataset == "bus_stu",
                    )
                    if copy_files or source_mask is not None
                    else None
                ),
            }
        )

    return pd.DataFrame(records)
