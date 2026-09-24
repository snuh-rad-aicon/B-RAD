"""Image and bounding-box transforms used by B-RAD-Det datasets."""

import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


def resize_pad_only(image, bbox, mask=None, output_size=(224, 224)):
    """Resize with preserved aspect ratio, pad, and transform an ``xyxy`` box."""
    width, height = image.size
    scale = min(output_size[0] / width, output_size[1] / height)
    resized = image.resize((int(width * scale), int(height * scale)))

    pad_width = output_size[0] - resized.size[0]
    pad_height = output_size[1] - resized.size[1]
    padding = (
        pad_width // 2,
        pad_height // 2,
        pad_width - pad_width // 2,
        pad_height - pad_height // 2,
    )
    padded = TF.pad(TF.to_tensor(resized), padding, fill=0)
    adjusted_bbox = torch.tensor(
        [
            bbox[0] * scale + padding[0],
            bbox[1] * scale + padding[1],
            bbox[2] * scale + padding[0],
            bbox[3] * scale + padding[1],
        ],
        dtype=torch.float32,
    )

    if mask is not None:
        mask_resized = mask.resize(resized.size, resample=Image.NEAREST)
        mask_padded = TF.pad(TF.to_tensor(mask_resized), padding, fill=0)
        return padded, adjusted_bbox, mask_padded
    return padded, adjusted_bbox


def crop_resize_pad_aug(
    image,
    bbox,
    mask=None,
    output_size=(224, 224),
    margin_ratio=1,
    jitter=0,
    crop=True,
):
    """Optionally crop around a box, then resize and pad the image and box."""
    if not crop:
        return resize_pad_only(image, bbox, mask, output_size)

    width, height = image.size
    x_min, y_min, x_max, y_max = bbox
    box_width, box_height = x_max - x_min, y_max - y_min
    margin_x = int(box_width * margin_ratio)
    margin_y = int(box_height * margin_ratio)
    center_x = (x_min + x_max) // 2 + int(
        random.uniform(-jitter, jitter) * box_width
    )
    center_y = (y_min + y_max) // 2 + int(
        random.uniform(-jitter, jitter) * box_height
    )
    crop_x0 = max(0, center_x - (box_width // 2 + margin_x))
    crop_y0 = max(0, center_y - (box_height // 2 + margin_y))
    crop_x1 = min(width, center_x + (box_width // 2 + margin_x))
    crop_y1 = min(height, center_y + (box_height // 2 + margin_y))

    cropped = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    scale = min(
        output_size[0] / cropped.size[0],
        output_size[1] / cropped.size[1],
    )
    resized = cropped.resize(
        (int(cropped.size[0] * scale), int(cropped.size[1] * scale))
    )
    pad_width = output_size[0] - resized.size[0]
    pad_height = output_size[1] - resized.size[1]
    padding = (
        pad_width // 2,
        pad_height // 2,
        pad_width - pad_width // 2,
        pad_height - pad_height // 2,
    )
    padded = TF.pad(TF.to_tensor(resized), padding, fill=0)
    adjusted_bbox = torch.tensor(
        [
            (x_min - crop_x0) * scale + padding[0],
            (y_min - crop_y0) * scale + padding[1],
            (x_max - crop_x0) * scale + padding[0],
            (y_max - crop_y0) * scale + padding[1],
        ],
        dtype=torch.float32,
    )
    return padded, adjusted_bbox


def rotate_bbox(bbox, angle, image_width, image_height):
    """Rotate an ``xyxy`` box around the image center and return its enclosure."""
    center_x, center_y = image_width / 2, image_height / 2
    radians = -np.radians(angle)

    x_min, y_min, x_max, y_max = bbox
    corners = np.array(
        [
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_min],
            [x_max, y_max],
        ],
        dtype=np.float32,
    )
    corners -= np.array([center_x, center_y], dtype=np.float32)
    rotation_matrix = np.array(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ],
        dtype=np.float32,
    )
    rotated = np.dot(corners, rotation_matrix.T)
    rotated += np.array([center_x, center_y], dtype=np.float32)

    x_coords = rotated[:, 0]
    y_coords = rotated[:, 1]
    return [x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()]


def augment_image_and_bbox_ultrasound(
    image,
    bbox,
    flip_prob=0.5,
    noise_prob=0.3,
    rotate_prob=0.4,
    rotation_angle=30.0,
    noise_std=5.0,
):
    """Apply the original ultrasound flip, rotation, and noise augmentation."""
    width, height = image.size
    x_min, y_min, x_max, y_max = bbox
    bbox_out = [x_min, y_min, x_max, y_max]

    if random.random() < flip_prob:
        image = TF.hflip(image)
        bbox_out = [
            width - bbox_out[2],
            bbox_out[1],
            width - bbox_out[0],
            bbox_out[3],
        ]

    if random.random() < rotate_prob:
        angle = random.uniform(-rotation_angle, rotation_angle)
        image = TF.rotate(image, angle)
        bbox_out = rotate_bbox(bbox_out, angle, width, height)

    if random.random() < noise_prob:
        image_array = np.array(image).astype(np.float32)
        noise = np.random.normal(0, noise_std, image_array.shape)
        image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(image_array)

    return image, bbox_out

