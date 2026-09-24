"""Tensor operations for B-RAD-Det."""

from .anchors import (
    decode_box_deltas,
    generate_anchors,
    pool_anchor_region_features_roi,
)

__all__ = [
    "decode_box_deltas",
    "generate_anchors",
    "pool_anchor_region_features_roi",
]

