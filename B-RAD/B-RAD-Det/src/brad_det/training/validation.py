"""Validation used during base B-RAD-Det training."""

from __future__ import annotations

from collections import defaultdict

import torch
from torchvision.ops import box_iou

from brad_det.losses import base_detection_loss, fewshot_detection_loss
from brad_det.ops.anchors import decode_box_deltas
from .engine import build_training_targets, select_batch_anchors


def _box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_base_loss_and_iou(
    model,
    loader,
    anchors,
    iou_threshold,
    ignore_threshold,
    device,
    input_size=224,
):
    model.eval()
    total_losses = defaultdict(float)
    iou_per_type = defaultdict(list)
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            batch_size, _, height, width = images.shape
            anchors_batch = select_batch_anchors(anchors, batch_size)
            outputs = model(images, labels=labels, anchors=anchors_batch)
            targets = build_training_targets(batch, labels, images, device)
            losses = base_detection_loss(
                outputs,
                targets,
                anchors_batch,
                epoch=30,
                iou_threshold=iou_threshold,
                ignore_threshold=ignore_threshold,
            )
            total_losses["total"] += sum(
                value.item() for value in losses.values()
            )
            for name, value in losses.items():
                total_losses[name] += value.item()
            num_batches += 1

            logits = outputs["pred_logits"].sigmoid()
            deltas = outputs["pred_boxes"]
            top_indices = {
                "tight": logits[:, :, 0].argmax(-1),
                "shadow": logits[:, :, 1].argmax(-1),
                "margin": logits[:, :, 2].argmax(-1),
            }
            anchors_expanded = anchors_batch.view(batch_size, -1, 4)
            num_anchors = anchors_expanded.shape[1]
            scale = torch.tensor(
                [input_size, input_size, input_size, input_size],
                dtype=anchors_expanded.dtype,
                device=anchors_expanded.device,
            ).view(1, 1, 4).expand(batch_size, num_anchors, 4)
            decoded = decode_box_deltas(deltas, anchors_expanded / scale)
            rows = torch.arange(batch_size, device=decoded.device)
            predictions = {
                name: decoded[rows, indices].cpu()
                for name, indices in top_indices.items()
            }
            ground_truth = {"tight": batch["bbox"]}
            if "shadow_bbox" in batch:
                ground_truth["shadow"] = batch["shadow_bbox"]
            if "margin_bbox" in batch:
                ground_truth["margin"] = batch["margin_bbox"]
            image_scale = torch.tensor(
                [[width, height, width, height]] * batch_size
            )
            for sample_index in range(batch_size):
                for box_type, boxes in ground_truth.items():
                    prediction = (
                        predictions[box_type][sample_index] * image_scale[sample_index]
                    ).clamp(min=0, max=max(width, height))
                    iou_per_type[box_type].append(
                        _box_iou(
                            prediction.tolist(),
                            boxes[sample_index].tolist(),
                        )
                    )

    if num_batches == 0:
        raise ValueError("Validation loader produced no batches")
    averaged_losses = {
        name: value / num_batches for name, value in total_losses.items()
    }
    mean_ious = {
        name: sum(values) / len(values) if values else 0.0
        for name, values in iou_per_type.items()
    }
    return averaged_losses, mean_ious


def evaluate_fewshot_loss_and_iou(
    model,
    loader,
    anchors,
    iou_threshold,
    ignore_threshold,
    num_ref_shot,
    device,
    input_size=224,
):
    """Run the one-reference validation used to select stage-2 checkpoints."""
    model.eval()
    total_losses = defaultdict(float)
    iou_per_type = defaultdict(list)
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            batch_size, _, height, width = images.shape
            anchors_batch = select_batch_anchors(anchors, batch_size)
            reference_anchors = anchors[:num_ref_shot]
            outputs = model(images, labels=labels, anchors=anchors_batch)
            target_features = outputs["class_embeddings"]
            targets = build_training_targets(batch, labels, images, device)

            reference_images = batch["reference"]["image"][0].to(device)
            reference_labels = batch["reference"]["label"][0].to(device)
            reference_boxes = batch["reference"]["bbox"][0].to(device)
            reference_outputs = model(
                reference_images,
                labels=reference_labels,
                anchors=reference_anchors,
            )
            reference_features = reference_outputs["class_embeddings"]
            reference_deltas = reference_outputs["pred_boxes"]
            reference_logits = reference_outputs["pred_logits"].sigmoid()

            reference_anchors_expanded = reference_anchors.view(
                num_ref_shot, -1, 4
            )
            reference_count = reference_anchors_expanded.shape[1]
            reference_scale = torch.tensor(
                [input_size, input_size, input_size, input_size],
                dtype=reference_anchors_expanded.dtype,
                device=device,
            ).view(1, 1, 4).expand(
                num_ref_shot, reference_count, 4
            )
            decoded_reference_boxes = decode_box_deltas(
                reference_deltas,
                reference_anchors_expanded / reference_scale,
            ) * reference_scale
            tight_logits = reference_logits[:, :, 0]
            shadow_logits = reference_logits[:, :, 1]
            margin_logits = reference_logits[:, :, 2]
            tight_mask = (
                (tight_logits > 0.8)
                & ((tight_logits - shadow_logits) > 0.4)
                & ((tight_logits - margin_logits) > 0.4)
            )

            selected_target_indices = []
            for reference_index in range(num_ref_shot):
                mask = tight_mask[reference_index]
                if mask.sum() > 0:
                    candidate_boxes = decoded_reference_boxes[reference_index][mask]
                    candidate_features = reference_features[reference_index][mask]
                else:
                    top_index = tight_logits[reference_index].argmax()
                    candidate_boxes = decoded_reference_boxes[reference_index][
                        top_index
                    ].unsqueeze(0)
                    candidate_features = reference_features[reference_index][
                        top_index
                    ].unsqueeze(0)
                reference_ious = box_iou(
                    candidate_boxes,
                    reference_boxes[reference_index].unsqueeze(0),
                )
                _, best_candidate = reference_ious.max(dim=0)
                selected_feature = candidate_features[best_candidate]
                selected_feature = selected_feature / selected_feature.norm(
                    dim=-1, keepdim=True
                )
                similarities = torch.einsum(
                    "ij,kj->ik",
                    target_features.squeeze(),
                    selected_feature,
                )
                selected_target_indices.append(
                    similarities.topk(k=1, largest=True, dim=0).indices
                )
            selected_target_indices = torch.stack(
                selected_target_indices
            ).squeeze()

            losses = fewshot_detection_loss(
                outputs,
                targets,
                anchors_batch,
                epoch=30,
                iou_threshold=iou_threshold,
                ignore_threshold=ignore_threshold,
            )
            total_losses["total"] += sum(
                value.item() for value in losses.values()
            )
            for name, value in losses.items():
                total_losses[name] += value.item()
            num_batches += 1

            logits = outputs["pred_logits"].sigmoid()
            deltas = outputs["pred_boxes"]
            top_indices = {
                "tight": logits[:, :, 0].argmax(-1),
                "shadow": logits[:, :, 1].argmax(-1),
                "margin": logits[:, :, 2].argmax(-1),
            }
            selected_deltas = deltas[:, selected_target_indices, :]
            anchors_expanded = anchors_batch.view(batch_size, -1, 4)
            anchor_count = anchors_expanded.shape[1]
            scale = torch.tensor(
                [input_size, input_size, input_size, input_size],
                dtype=anchors_expanded.dtype,
                device=device,
            ).view(1, 1, 4).expand(batch_size, anchor_count, 4)
            anchors_normalized = anchors_expanded / scale
            decoded = decode_box_deltas(deltas, anchors_normalized)
            selected_boxes = decode_box_deltas(
                selected_deltas,
                anchors_normalized[:, selected_target_indices, :],
            ).cpu()
            rows = torch.arange(batch_size, device=decoded.device)
            predictions = {
                name: decoded[rows, indices].cpu()
                for name, indices in top_indices.items()
            }
            ground_truth = {"tight": batch["bbox"]}
            if "shadow_bbox" in batch:
                ground_truth["shadow"] = batch["shadow_bbox"]
            if "margin_bbox" in batch:
                ground_truth["margin"] = batch["margin_bbox"]
            image_scale = torch.tensor(
                [[width, height, width, height]] * batch_size
            )
            for sample_index in range(batch_size):
                for box_type, boxes in ground_truth.items():
                    prediction = (
                        predictions[box_type][sample_index]
                        * image_scale[sample_index]
                    ).clamp(min=0, max=max(width, height))
                    iou_per_type[box_type].append(
                        _box_iou(
                            prediction.tolist(),
                            boxes[sample_index].tolist(),
                        )
                    )

                reference_ious = []
                for reference_index in range(num_ref_shot):
                    if selected_boxes.ndim == 2:
                        prediction = selected_boxes[sample_index]
                    else:
                        prediction = selected_boxes[sample_index][reference_index]
                    prediction = (
                        prediction * image_scale[sample_index]
                    ).clamp(min=0, max=max(width, height))
                    reference_ious.append(
                        _box_iou(
                            prediction.tolist(),
                            ground_truth["tight"][sample_index].tolist(),
                        )
                    )
                iou_per_type["ref_best"].append(max(reference_ious))

    if num_batches == 0:
        raise ValueError("Validation loader produced no batches")
    averaged_losses = {
        name: value / num_batches for name, value in total_losses.items()
    }
    mean_ious = {
        name: sum(values) / len(values) if values else 0.0
        for name, values in iou_per_type.items()
    }
    return averaged_losses, mean_ious
