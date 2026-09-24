"""Reference-guided B-RAD-Det datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from .base import (
    BaseDetectionDataset,
    build_auxiliary_boxes,
    parse_bbox,
    parse_sequence,
    prepare_image_tensor,
)


class FewShotDetectionDataset(BaseDetectionDataset):
    """Add retrieved support images and boxes to each query sample."""

    def __init__(
        self,
        metadata,
        image_root=None,
        input_size=224,
        apply_aug=True,
        source_list=None,
        long_bbox=False,
        margin_bbox_ratio=0,
        num_ref_shot=0,
        birads_class=None,
        filter_birads=False,
        reference_metadata=(),
        is_inference=False,
        exclude_crawled_references=False,
        reference_path_columns=None,
        restrict_references_to_metadata=False,
    ):
        super().__init__(
            metadata=metadata,
            image_root=image_root,
            input_size=input_size,
            apply_aug=apply_aug,
            source_list=source_list,
            long_bbox=long_bbox,
            margin_bbox_ratio=margin_bbox_ratio,
        )
        self.num_ref_shot = num_ref_shot
        self.is_inference = is_inference
        self.exclude_crawled_references = exclude_crawled_references
        self.reference_path_columns = reference_path_columns
        self.restrict_references_to_metadata = restrict_references_to_metadata

        if filter_birads:
            self.df = self.df[self.df["birads"] == birads_class]

        frames = [self.df]
        for metadata_path in reference_metadata:
            frame = (
                pd.read_csv(metadata_path)
                if isinstance(metadata_path, (str, Path))
                else metadata_path.copy()
            )
            frames.append(frame)
        self.reference_df = pd.concat(frames).drop_duplicates(
            subset=["image_path"]
        ).reset_index(drop=True)
        self.reference_frames = [self.reference_df, *frames]

    def _reference_candidates(self, row, query_path):
        if self.reference_path_columns is None:
            candidates = list(parse_sequence(row["seg_ref_imgs"]))
        else:
            candidates = [row[column] for column in self.reference_path_columns]
        if query_path in candidates:
            candidates.remove(query_path)
        candidates = [path for path in candidates if "normal" not in path.lower()]
        if self.exclude_crawled_references:
            candidates = [
                path for path in candidates if "crawled" not in path.lower()
            ]
        if self.restrict_references_to_metadata:
            known_paths = set(self.reference_df["image_path"].values)
            candidates = [path for path in candidates if path in known_paths]
        return candidates[: self.num_ref_shot]

    def _lookup_reference_bbox(self, relative_path, image_size):
        for frame in self.reference_frames:
            try:
                value = frame[frame["image_path"] == relative_path]["bbox_xyxy"].item()
                bbox = parse_sequence(value)
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    raise ValueError
                return bbox
            except (ValueError, KeyError, SyntaxError, TypeError):
                continue
        width, height = image_size
        return [0, 0, width, height]

    def _load_references(self, row, query_relative_path):
        candidates = self._reference_candidates(row, query_relative_path)
        paths = []
        images = []
        boxes = []
        labels = []
        shadow_boxes = []
        margin_boxes = []

        for index in range(self.num_ref_shot):
            relative_path = candidates[index]
            image_path = self.image_root / relative_path
            image = Image.open(image_path).convert("RGB")
            bbox = self._lookup_reference_bbox(relative_path, image.size)
            image_tensor, bbox_tensor = prepare_image_tensor(
                image,
                bbox,
                self.input_size,
            )

            paths.append(str(image_path))
            images.append(image_tensor)
            boxes.append(bbox_tensor)
            labels.append(0)

            if self.is_inference:
                auxiliary = build_auxiliary_boxes(
                    image,
                    bbox,
                    self.input_size,
                    long_bbox=True,
                    margin_bbox_ratio=self.margin_bbox_ratio,
                )
                shadow_boxes.append(auxiliary["shadow_bbox"])
                if self.margin_bbox_ratio > 0:
                    margin_boxes.append(auxiliary["margin_bbox"])

        reference = {
            "text": row["medgemma_txt"],
            "image_path": paths,
            "image": torch.stack(images),
            "bbox": torch.stack(boxes),
            "label": torch.tensor(labels, dtype=torch.long),
        }
        if self.is_inference:
            reference["shadow_bbox"] = torch.stack(shadow_boxes)
            reference["margin_bbox"] = torch.stack(margin_boxes)
        return reference

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path, image, bbox, image_tensor, bbox_tensor = self._load_query(row)
        sample = {
            "path": str(image_path),
            "image": image_tensor,
            "bbox": bbox_tensor,
            "label": 0,
        }
        if self.num_ref_shot != 0:
            sample["reference"] = self._load_references(row, str(image_path))
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


class EvaluationDetectionDataset(FewShotDetectionDataset):
    """Few-shot dataset with the reference filtering used by final evaluation."""

    def __init__(self, *args, **kwargs):
        kwargs["apply_aug"] = False
        kwargs["exclude_crawled_references"] = True
        kwargs.setdefault("is_inference", False)
        kwargs.setdefault("restrict_references_to_metadata", True)
        super().__init__(*args, **kwargs)
