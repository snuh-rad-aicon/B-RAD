"""Bounding-box metrics and inverse resize/pad geometry."""

from __future__ import annotations

from PIL import Image


def bbox_iou(box1, box2):
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area1 = max(float(box1[2]) - float(box1[0]), 0.0) * max(
        float(box1[3]) - float(box1[1]), 0.0
    )
    area2 = max(float(box2[2]) - float(box2[0]), 0.0) * max(
        float(box2[3]) - float(box2[1]), 0.0
    )
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def bbox_dice(box1, box2):
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area1 = max(float(box1[2]) - float(box1[0]), 0.0) * max(
        float(box1[3]) - float(box1[1]), 0.0
    )
    area2 = max(float(box2[2]) - float(box2[0]), 0.0) * max(
        float(box2[3]) - float(box2[1]), 0.0
    )
    denominator = area1 + area2
    return 2.0 * intersection / denominator if denominator > 0 else 0.0


def invert_resize_pad_box(image_path, box, output_size=224):
    """Map an ``xyxy`` box in the padded model frame to the source image."""
    with Image.open(image_path) as image:
        width, height = image.size
    scale = min(output_size / width, output_size / height)
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    pad_left = (output_size - resized_width) // 2
    pad_top = (output_size - resized_height) // 2
    result = [
        (float(box[0]) - pad_left) / scale,
        (float(box[1]) - pad_top) / scale,
        (float(box[2]) - pad_left) / scale,
        (float(box[3]) - pad_top) / scale,
    ]
    return [
        max(0.0, min(result[0], width)),
        max(0.0, min(result[1], height)),
        max(0.0, min(result[2], width)),
        max(0.0, min(result[3], height)),
    ]
