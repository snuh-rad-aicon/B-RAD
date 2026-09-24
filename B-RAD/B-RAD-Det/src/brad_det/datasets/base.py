"""Base B-RAD-Det dataset."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import augment_image_and_bbox_ultrasound, crop_resize_pad_aug


def parse_sequence(value):
    """Parse a serialized Python list or tuple from public metadata."""
    return ast.literal_eval(value) if isinstance(value, str) else value


def parse_bbox(value, image_size):
    """Parse an ``xyxy`` box, falling back to the full image as in the original."""
    width, height = image_size
    try:
        bbox = parse_sequence(value)
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError
        return bbox
    except (ValueError, SyntaxError, TypeError):
        return [0, 0, width, height]


def prepare_image_tensor(image, bbox, input_size):
    image_tensor, bbox_tensor = crop_resize_pad_aug(
        image,
        bbox,
        output_size=(input_size, input_size),
        crop=False,
    )
    if image_tensor.max() > 1:
        image_tensor = image_tensor / 255.0
    if image_tensor.shape[0] != 3:
        image_tensor = image_tensor.permute(2, 0, 1)
    return image_tensor - 0.5, bbox_tensor


def build_auxiliary_boxes(image, bbox, input_size, long_bbox, margin_bbox_ratio):
    """Build the posterior-shadow and expanded-margin box targets."""
    width, height = image.size
    output = {}
    if long_bbox:
        shadow_bbox = [bbox[0], bbox[1], bbox[2], image.height]
        _, output["shadow_bbox"] = crop_resize_pad_aug(
            image,
            shadow_bbox,
            output_size=(input_size, input_size),
            crop=False,
        )

    if margin_bbox_ratio > 0:
        delta_x = (bbox[2] - bbox[0]) * margin_bbox_ratio
        delta_y = (bbox[3] - bbox[1]) * margin_bbox_ratio
        margin_bbox = [
            max(0, bbox[0] - delta_x),
            max(0, bbox[1] - delta_y),
            min(width, bbox[2] + delta_x),
            min(height, bbox[3] + delta_y),
        ]
        _, output["margin_bbox"] = crop_resize_pad_aug(
            image,
            margin_bbox,
            output_size=(input_size, input_size),
            crop=False,
        )
    return output


class BaseDetectionDataset(Dataset):
    """Load query images and tight, shadow, and margin box targets."""

    def __init__(
        self,
        metadata,
        image_root=None,
        input_size=224,
        apply_aug=True,
        source_list=None,
        long_bbox=False,
        margin_bbox_ratio=0,
    ):
        self.df = (
            pd.read_csv(metadata)
            if isinstance(metadata, (str, Path))
            else metadata.copy()
        )
        self.image_root = Path(image_root) if image_root is not None else Path()
        self.input_size = input_size
        self.apply_aug = apply_aug
        self.long_bbox = long_bbox
        self.margin_bbox_ratio = margin_bbox_ratio

        if source_list is not None and "source_dataset" in self.df.columns:
            sources = [source.lower() for source in source_list]
            self.df = self.df[
                self.df["source_dataset"].str.lower().isin(sources)
            ]

    def __len__(self):
        return len(self.df)

    def _load_query(self, row):
        image_path = self.image_root / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        bbox = parse_bbox(row["bbox_xyxy"], image.size)
        if self.apply_aug:
            image, bbox = augment_image_and_bbox_ultrasound(image, bbox)
        image_tensor, bbox_tensor = prepare_image_tensor(
            image,
            bbox,
            self.input_size,
        )
        return image_path, image, bbox, image_tensor, bbox_tensor

    def __getitem__(self, index):
        row = self.df.iloc[index]
        _, image, bbox, image_tensor, bbox_tensor = self._load_query(row)
        sample = {
            "image": image_tensor,
            "bbox": bbox_tensor,
            "label": 0,
        }
        sample.update(
            build_auxiliary_boxes(
                image,
                bbox,
                self.input_size,
                self.long_bbox,
                self.margin_bbox_ratio,
            )
        )
        return sample

