"""Base and reference-guided B-RAD-Det training losses."""

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from brad_det.losses.box_ops import ciou_loss, weighted_box_l1_loss
from brad_det.ops.anchors import decode_box_deltas


def _compute_detection_losses(
    outputs,
    targets,
    anchors,
    iou_threshold,
    ignore_threshold,
):
    pred_boxes = outputs["pred_boxes"]
    pred_logits = outputs["pred_logits"]
    batch_size = pred_logits.shape[0]

    total_class_loss = 0
    total_iou_loss = 0
    total_l1_loss = 0
    best_pred_targets = []

    boxes = [targets["boxes"]]
    labels = [targets["labels"].argmax(dim=-1)]
    if "shadow_boxes" in targets:
        boxes.append(targets["shadow_boxes"])
        labels.append(torch.full_like(labels[0], fill_value=1))
    if "margin_boxes" in targets:
        boxes.append(targets["margin_boxes"])
        labels.append(torch.full_like(labels[0], fill_value=2))

    merged_boxes = torch.cat(boxes, dim=1)
    merged_labels = torch.cat(labels, dim=1)
    anchors_flat = anchors.view(batch_size, -1, 4)

    for batch_index in range(batch_size):
        gt_boxes = merged_boxes[batch_index]
        gt_labels = merged_labels[batch_index]
        num_gt = gt_boxes.shape[0]
        image_width, image_height = targets["size"][batch_index]
        scale = torch.tensor(
            [image_width, image_height, image_width, image_height],
            dtype=pred_boxes.dtype,
            device=pred_boxes.device,
        )
        gt_boxes = gt_boxes / scale
        anchors_norm = anchors_flat[batch_index] / scale

        if num_gt == 0:
            class_target = torch.zeros_like(pred_logits[batch_index])
            total_class_loss += F.binary_cross_entropy_with_logits(
                pred_logits[batch_index],
                class_target,
            )
            best_pred_targets.append(None)
            continue

        ious = box_iou(anchors_norm, gt_boxes)
        max_ious, matched_gt_index = ious.max(dim=1)
        positive_mask = max_ious > iou_threshold
        negative_mask = max_ious <= ignore_threshold

        top_k = 5
        _, top_indices = ious.topk(k=min(top_k, ious.size(0)), dim=0)
        for gt_index in range(gt_boxes.size(0)):
            anchor_indices = top_indices[:, gt_index]
            positive_mask[anchor_indices] = True
            matched_gt_index[anchor_indices] = gt_index

        class_target = torch.zeros_like(pred_logits[batch_index])
        matched_labels = gt_labels[matched_gt_index[positive_mask]]
        matched_indices = torch.nonzero(
            positive_mask,
            as_tuple=False,
        ).squeeze(1)
        class_target[matched_indices, matched_labels] = 1.0

        class_loss = F.binary_cross_entropy_with_logits(
            pred_logits[batch_index][positive_mask | negative_mask],
            class_target[positive_mask | negative_mask],
            reduction="mean",
        )
        best_pred_targets.append(
            outputs["class_embeddings"][
                batch_index,
                pred_logits[batch_index][:, 0].argmax(),
            ]
        )

        decoded_boxes = decode_box_deltas(
            pred_boxes[batch_index],
            anchors_norm,
        )
        if positive_mask.any():
            pred_positive_boxes = decoded_boxes[positive_mask]
            target_positive_boxes = gt_boxes[matched_gt_index[positive_mask]]
            iou_loss = ciou_loss(
                pred_positive_boxes,
                target_positive_boxes,
            ).mean()
            l1_loss = weighted_box_l1_loss(
                pred_positive_boxes,
                target_positive_boxes,
                reduction="mean",
                tight=True,
            ).mean()
        else:
            iou_loss = torch.tensor(0.0, device=pred_boxes.device)
            l1_loss = torch.tensor(0.0, device=pred_boxes.device)

        total_class_loss += class_loss
        total_iou_loss += iou_loss
        total_l1_loss += l1_loss

    losses = {
        "loss_class": total_class_loss / batch_size,
        "loss_iou": total_iou_loss / batch_size,
        "loss_l1": total_l1_loss / batch_size,
    }
    return losses, best_pred_targets, anchors_flat


def base_detection_loss(
    outputs,
    targets,
    anchors,
    epoch,
    iou_threshold=0.5,
    ignore_threshold=0.2,
):
    """Compute the loss used by base detector training."""
    losses, _, _ = _compute_detection_losses(
        outputs,
        targets,
        anchors,
        iou_threshold,
        ignore_threshold,
    )
    return losses


def fewshot_detection_loss(
    outputs,
    targets,
    anchors,
    epoch,
    iou_threshold=0.5,
    ignore_threshold=0.2,
    num_ref_shot=0,
    ref_outputs=None,
    input_size=224,
):
    """Compute the loss used by single-reference fine-tuning."""
    losses, best_pred_targets, anchors_flat = _compute_detection_losses(
        outputs,
        targets,
        anchors,
        iou_threshold,
        ignore_threshold,
    )
    pred_boxes = outputs["pred_boxes"]
    batch_size = pred_boxes.shape[0]
    total_ref_loss = 0

    if ref_outputs is not None:
        ref_pred_boxes = ref_outputs["pred_boxes"]
        ref_anchor_features = ref_outputs["class_embeddings"]

        for batch_index in range(batch_size):
            best_pred_target = best_pred_targets[batch_index]
            if best_pred_target is None:
                continue

            gt_ref_box = targets["reference"]["bbox"][batch_index].to(
                ref_pred_boxes.device
            )
            ref_pred_box = ref_pred_boxes[batch_index]
            ref_anchors = anchors[:num_ref_shot].view(num_ref_shot, -1, 4)
            scale = torch.tensor(
                [input_size, input_size, input_size, input_size],
                dtype=ref_anchors.dtype,
                device=ref_anchors.device,
            )
            gt_ref_box = gt_ref_box / scale
            anchors_norm = anchors_flat[batch_index] / scale
            decoded_ref_boxes = decode_box_deltas(ref_pred_box, anchors_norm)
            ref_ious = box_iou(
                decoded_ref_boxes,
                gt_ref_box.to(ref_pred_box.device),
            )
            max_ref_iou, matched_ref_index = ref_ious.max(dim=0)

            if max_ref_iou < 0.7:
                ref_loss = torch.tensor(0.0, device=ref_pred_boxes.device)
            else:
                ref_feature = ref_anchor_features[batch_index][matched_ref_index]
                ref_feature = ref_feature / ref_feature.norm(dim=-1, keepdim=True)
                ref_loss = 1 - F.cosine_similarity(
                    best_pred_target,
                    ref_feature,
                    dim=-1,
                ).mean()
            total_ref_loss += ref_loss

    if total_ref_loss == 0:
        total_ref_loss = torch.tensor(0.0, device=pred_boxes.device)
    losses["loss_ref"] = 0.1 * total_ref_loss / batch_size
    return losses

