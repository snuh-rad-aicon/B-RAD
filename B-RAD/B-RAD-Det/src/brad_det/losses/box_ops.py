"""Active bounding-box losses used by B-RAD-Det training."""

import torch
from torchvision.ops import box_iou


def weighted_box_l1_loss(pred, tgt, weights=None, reduction="mean", tight=True):
    """Compute the pairwise L1 loss between predicted and target boxes."""
    if pred.shape[-1] != 4 or tgt.shape[-1] != 4:
        raise ValueError(f"Expected box shape (..., 4), got {pred.shape}, {tgt.shape}")

    pred = pred.unsqueeze(2)
    tgt = tgt.unsqueeze(1)

    if tight:
        abs_diff = torch.abs(pred - tgt)
    else:
        diff = pred - tgt
        xy1_diff, xy2_diff = torch.split(diff, 2, dim=-1)
        xy1 = torch.minimum(xy1_diff, torch.zeros_like(xy1_diff))
        xy2 = torch.maximum(xy2_diff, torch.zeros_like(xy2_diff))
        abs_diff = torch.abs(torch.cat([xy1, xy2], dim=-1))

    if weights is not None:
        abs_diff = abs_diff * weights.unsqueeze(-1)

    if reduction == "mean":
        return abs_diff.mean(dim=-1)
    if reduction is None:
        return abs_diff
    raise ValueError(f"Unknown reduction: {reduction}")


def bbox_center(box):
    return (box[..., :2] + box[..., 2:]) / 2


def bbox_size(box):
    return box[..., 2:] - box[..., :2]


def ciou_loss(pred, target):
    """Compute Complete IoU loss for aligned ``xyxy`` box pairs."""
    pred_center = bbox_center(pred)
    target_center = bbox_center(target)
    pred_size = bbox_size(pred)
    target_size = bbox_size(target)

    iou = box_iou(pred, target).diag()
    center_distance = ((pred_center - target_center) ** 2).sum(dim=-1)

    x1 = torch.min(pred[:, 0], target[:, 0])
    y1 = torch.min(pred[:, 1], target[:, 1])
    x2 = torch.max(pred[:, 2], target[:, 2])
    y2 = torch.max(pred[:, 3], target[:, 3])
    diagonal_length = (x2 - x1) ** 2 + (y2 - y1) ** 2

    aspect_penalty = (4 / (torch.pi**2)) * torch.pow(
        torch.atan(target_size[:, 0] / target_size[:, 1])
        - torch.atan(pred_size[:, 0] / pred_size[:, 1]),
        2,
    )
    with torch.no_grad():
        alpha = aspect_penalty / (1 - iou + aspect_penalty + 1e-7)

    ciou = (
        iou
        - center_distance / (diagonal_length + 1e-7)
        - alpha * aspect_penalty
    )
    return 1 - ciou

