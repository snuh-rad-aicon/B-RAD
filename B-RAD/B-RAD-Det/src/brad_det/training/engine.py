"""Per-batch training operations for base and few-shot B-RAD-Det stages."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from brad_det.losses import base_detection_loss, fewshot_detection_loss


def _to_device(value, device):
    return value.to(device, non_blocking=True)


def select_batch_anchors(anchors, batch_size):
    """Select the anchor prefix used for a possibly incomplete final batch."""
    if anchors.shape[0] != batch_size:
        return anchors[:batch_size]
    return anchors


def build_training_targets(batch, labels, images, device, include_reference=False):
    """Build the target dictionary in the original insertion order."""
    batch_size, _, height, width = images.shape
    targets = {
        "labels": F.one_hot(labels, num_classes=4).float().unsqueeze(1),
        "boxes": _to_device(batch["bbox"], device).unsqueeze(1),
        "size": torch.tensor(
            [[width, height]] * batch_size,
            device=device,
        ),
    }
    if include_reference:
        targets["reference"] = batch["reference"]
    if "shadow_bbox" in batch:
        targets["shadow_boxes"] = _to_device(
            batch["shadow_bbox"], device
        ).unsqueeze(1)
    if "margin_bbox" in batch:
        targets["margin_boxes"] = _to_device(
            batch["margin_bbox"], device
        ).unsqueeze(1)
    return targets


def train_base_batch(
    model,
    batch,
    anchors,
    optimizer,
    epoch,
    iou_threshold,
    ignore_threshold,
    device,
    loss_function=base_detection_loss,
):
    """Run the active base-training update for one batch."""
    optimizer.zero_grad()
    images = _to_device(batch["image"], device)
    labels = _to_device(batch["label"], device)
    anchors_batch = select_batch_anchors(anchors, images.shape[0])
    outputs = model(images, labels=labels, anchors=anchors_batch)
    targets = build_training_targets(batch, labels, images, device)
    losses = loss_function(
        outputs,
        targets,
        anchors_batch,
        epoch,
        iou_threshold=iou_threshold,
        ignore_threshold=ignore_threshold,
    )
    loss = sum(losses.values())
    loss.backward()
    optimizer.step()
    return losses, loss


def train_fewshot_batch(
    model,
    batch,
    anchors,
    optimizer,
    epoch,
    iou_threshold,
    ignore_threshold,
    num_ref_shot,
    device,
    loss_function=fewshot_detection_loss,
):
    """Run the active reference-guided update for one batch."""
    optimizer.zero_grad()
    images = _to_device(batch["image"], device)
    labels = _to_device(batch["label"], device)
    anchors_batch = select_batch_anchors(anchors, images.shape[0])
    outputs = model(images, labels=labels, anchors=anchors_batch)
    targets = build_training_targets(
        batch,
        labels,
        images,
        device,
        include_reference=True,
    )

    optimizer.zero_grad()
    reference_images = _to_device(targets["reference"]["image"].squeeze(), device)
    reference_labels = _to_device(targets["reference"]["label"].squeeze(), device)
    with torch.no_grad():
        model.eval()
        reference_outputs = model(
            reference_images,
            labels=[
                reference_labels.unsqueeze(-1),
                reference_labels.unsqueeze(-1) + 1,
                reference_labels.unsqueeze(-1) + 2,
            ],
            anchors=anchors[: reference_images.shape[0]],
        )
        model.train()

    losses = loss_function(
        outputs,
        targets,
        anchors_batch,
        epoch,
        iou_threshold=iou_threshold,
        ignore_threshold=ignore_threshold,
        num_ref_shot=num_ref_shot,
        ref_outputs=reference_outputs,
    )
    loss = sum(losses.values())
    loss.backward()
    optimizer.step()
    return losses, loss


def train_one_epoch(
    model,
    loader,
    anchors,
    optimizer,
    epoch,
    iou_threshold,
    ignore_threshold,
    device,
    num_ref_shot=0,
):
    """Run one base or few-shot epoch and return original-style averages."""
    model.train()
    epoch_losses = defaultdict(float)
    num_batches = 0
    for batch in loader:
        if num_ref_shot == 0:
            losses, loss = train_base_batch(
                model,
                batch,
                anchors,
                optimizer,
                epoch,
                iou_threshold,
                ignore_threshold,
                device,
            )
        else:
            losses, loss = train_fewshot_batch(
                model,
                batch,
                anchors,
                optimizer,
                epoch,
                iou_threshold,
                ignore_threshold,
                num_ref_shot,
                device,
            )
        for name, value in losses.items():
            epoch_losses[name] += value.item()
        epoch_losses["total"] += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Training loader produced no batches")
    return {
        name: value / num_batches
        for name, value in epoch_losses.items()
    }
