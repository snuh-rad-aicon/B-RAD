"""Zero-shot and retrieval-guided evaluation kernel used for paper results."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm.auto import tqdm

from brad_det.ops.anchors import decode_box_deltas
from .geometry import bbox_dice, bbox_iou


def _decode(outputs, anchors, input_size):
    batch_size = outputs["pred_boxes"].shape[0]
    flat = anchors[:batch_size].reshape(batch_size, -1, 4)
    scale = torch.as_tensor(
        [input_size] * 4, dtype=flat.dtype, device=flat.device
    ).view(1, 1, 4)
    return decode_box_deltas(outputs["pred_boxes"], flat / scale) * scale


def _reference_query_indices(model, reference, anchors, target_features, device, input_size):
    reference_images = reference["image"][0].to(device)
    reference_boxes = reference["bbox"][0].to(device)
    count = reference_images.shape[0]
    outputs = model(reference_images, anchors=anchors[:count])
    decoded = _decode(outputs, anchors[:count], input_size)
    logits = outputs["pred_logits"].sigmoid()
    features = outputs["class_embeddings"]
    selected = []
    for index in range(count):
        tight = logits[index, :, 0]
        mask = (tight > 0.8) & ((tight - logits[index, :, 1]) > 0.4) & (
            (tight - logits[index, :, 2]) > 0.4
        )
        if mask.any():
            boxes = decoded[index][mask]
            candidates = features[index][mask]
        else:
            top = tight.argmax()
            boxes = decoded[index][top].unsqueeze(0)
            candidates = features[index][top].unsqueeze(0)
        best = box_iou(boxes, reference_boxes[index].unsqueeze(0)).argmax()
        reference_feature = F.normalize(candidates[best], dim=-1)
        similarities = target_features[0] @ reference_feature
        selected.append(int(similarities.argmax()))
    return selected


def evaluate_detector(
    model,
    loader,
    anchors,
    device,
    input_size=224,
    reference_shots=4,
    progress_description=None,
):
    """Return per-case and aggregate metrics used by the paper's detector rows."""
    model.eval()
    rows = []
    values = defaultdict(list)
    with torch.no_grad():
        iterator = tqdm(
            loader,
            total=len(loader),
            desc=progress_description,
            unit="image",
            dynamic_ncols=True,
            disable=progress_description is None,
        )
        for batch in iterator:
            images = batch["image"].to(device)
            outputs = model(images, anchors=anchors[: images.shape[0]])
            decoded = _decode(outputs, anchors, input_size)[0]
            logits = outputs["pred_logits"].sigmoid()[0]
            zero_box = decoded[logits[:, 0].argmax()].clamp(0, input_size)
            gt = batch["bbox"][0]
            zero_iou = bbox_iou(zero_box.cpu(), gt)
            zero_dice = bbox_dice(zero_box.cpu(), gt)
            candidate_boxes = []
            if reference_shots:
                target_features = F.normalize(outputs["class_embeddings"], dim=-1)
                indices = _reference_query_indices(
                    model, batch["reference"], anchors, target_features,
                    device, input_size,
                )
                candidate_boxes = [decoded[index].clamp(0, input_size) for index in indices]
            candidate_ious = [bbox_iou(box.cpu(), gt) for box in candidate_boxes]
            candidate_dice = [bbox_dice(box.cpu(), gt) for box in candidate_boxes]
            best_iou = max(candidate_ious) if candidate_ious else zero_iou
            best_dice = max(candidate_dice) if candidate_dice else zero_dice
            path = batch.get("path", [""])[0]
            rows.append({
                "image_path": path,
                "zero_bbox_xyxy": zero_box.cpu().tolist(),
                "candidate_bboxes_xyxy": [box.cpu().tolist() for box in candidate_boxes],
                "zero_mDSC": zero_dice,
                "zero_mIoU": zero_iou,
                "kshot_mDSC": best_dice,
                "kshot_mIoU": best_iou,
            })
            values["zero_mDSC"].append(zero_dice)
            values["zero_mIoU"].append(zero_iou)
            values["kshot_mDSC"].append(best_dice)
            values["kshot_mIoU"].append(best_iou)
            prefix = "zero" if reference_shots == 0 else "kshot"
            iterator.set_postfix(
                mDSC=f"{sum(values[prefix + '_mDSC']) / len(rows):.4f}",
                mIoU=f"{sum(values[prefix + '_mIoU']) / len(rows):.4f}",
                refresh=False,
            )
    if not rows:
        raise ValueError("Evaluation loader produced no samples")
    summary = {key: sum(series) / len(series) for key, series in values.items()}
    summary["count"] = len(rows)
    return summary, rows
