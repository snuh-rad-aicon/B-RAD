"""Anchor generation, ROI pooling, and box decoding."""

import itertools

import torch
from torchvision.ops import roi_align


def generate_anchors(
    feature_map_size,
    stride,
    scales,
    aspect_ratios,
    device="cpu",
):
    """Generate anchors for every location in a feature map."""
    height, width = feature_map_size
    shifts_x = torch.arange(
        0,
        width * stride,
        step=stride,
        dtype=torch.float32,
        device=device,
    )
    shifts_y = torch.arange(
        0,
        height * stride,
        step=stride,
        dtype=torch.float32,
        device=device,
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    center_xy = torch.stack((shift_x, shift_y), dim=-1).reshape(-1, 2)

    anchor_sizes = []
    for scale, ratio in itertools.product(scales, aspect_ratios):
        area = (stride * scale) ** 2
        anchor_width = (area * ratio) ** 0.5
        anchor_height = area / anchor_width
        anchor_sizes.append((anchor_width, anchor_height))

    anchor_sizes = torch.tensor(
        anchor_sizes,
        dtype=torch.float32,
        device=device,
    )
    num_anchors = anchor_sizes.size(0)
    centers = center_xy[:, None, :].expand(-1, num_anchors, -1)
    sizes = anchor_sizes[None, :, :].expand(center_xy.size(0), -1, -1)
    anchors = torch.cat(
        [centers - 0.5 * sizes, centers + 0.5 * sizes],
        dim=-1,
    )
    return anchors, anchors.size(1)


def pool_anchor_region_features_roi(
    features,
    anchors,
    img_size,
    patch_size,
    output_size=(1, 1),
    mode="mean",
):
    """Pool ViT feature regions aligned to anchors using ROIAlign."""
    batch_size, height, width, dim = features.shape
    _, num_locations, num_anchors, _ = anchors.shape
    spatial_scale = 1.0 / patch_size
    output_height, output_width = output_size

    features = features.permute(0, 3, 1, 2).contiguous()
    anchors_flat = anchors.view(
        batch_size * num_locations * num_anchors,
        4,
    )
    batch_indices = torch.arange(
        batch_size,
        device=anchors.device,
    ).repeat_interleave(num_locations * num_anchors).unsqueeze(1)
    rois = torch.cat([batch_indices, anchors_flat], dim=1).float()

    aligned = roi_align(
        input=features,
        boxes=rois,
        output_size=output_size,
        spatial_scale=spatial_scale,
        aligned=True,
    )
    aligned = aligned.view(
        batch_size,
        num_locations * num_anchors,
        dim,
        output_height,
        output_width,
    )

    if mode == "mean":
        return aligned.mean(dim=(-1, -2))
    if mode == "max":
        return aligned.amax(dim=(-1, -2))
    if mode == "flatten":
        return aligned.flatten(start_dim=3).flatten(start_dim=2)
    raise ValueError(f"Unsupported mode: {mode}")


def decode_box_deltas(deltas, anchors, eps=1e-6):
    """Decode ``(tx, ty, tw, th)`` deltas into ``xyxy`` boxes."""
    anchor_x = (anchors[..., 0] + anchors[..., 2]) * 0.5
    anchor_y = (anchors[..., 1] + anchors[..., 3]) * 0.5
    anchor_width = (anchors[..., 2] - anchors[..., 0]).clamp(min=eps)
    anchor_height = (anchors[..., 3] - anchors[..., 1]).clamp(min=eps)

    tx, ty, tw, th = (
        deltas[..., 0],
        deltas[..., 1],
        deltas[..., 2],
        deltas[..., 3],
    )
    center_x = tx * anchor_width + anchor_x
    center_y = ty * anchor_height + anchor_y
    width = torch.exp(tw.clamp(max=10.0)) * anchor_width
    height = torch.exp(th.clamp(max=10.0)) * anchor_height

    return torch.stack(
        [
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        ],
        dim=-1,
    )

