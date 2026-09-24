"""Lesion detection on a single image, zero-shot or retrieval-guided few-shot.

Runs the released B-RAD-Det checkpoint on one ultrasound image and prints the
tight lesion box. With `--references`, the support cases retrieved by B-RAD-CLIP
guide the box the same way `evaluation/engine.py` does for the paper's few-shot
rows; without them the detector runs zero-shot off its learned class query.
`scripts/evaluate.py` reproduces the paper's detector table and needs the full
metadata splits; this entry point exists so the end-to-end demo can run the
detector on an arbitrary image.

    python inference_single.py --image <img.png> --ckpt ../ckpts/b_rad_det.pt \
        --references ref1.png:ref1_mask.png ref2.png:ref2_mask.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch.nn.functional as F                                # noqa: E402
from torchvision.ops import box_iou                             # noqa: E402

from brad_det import load_config                               # noqa: E402
from brad_det.datasets.base import prepare_image_tensor        # noqa: E402
from brad_det.ops.anchors import decode_box_deltas             # noqa: E402
from brad_det.training import build_model_and_anchors          # noqa: E402

CLASSES = ["tight", "margin", "shadow"]


def bbox_from_mask(mask_path):
    """Tight xyxy box around the lesion mask."""
    import numpy as np
    from PIL import Image as _Image
    mask = np.asarray(_Image.open(mask_path).convert("L"))
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def decode_boxes(outputs, anchors, count, input_size):
    """Anchor deltas -> xyxy boxes in the padded input frame (engine.py `_decode`)."""
    flat = anchors[:count].reshape(count, -1, 4)
    scale = torch.as_tensor([input_size] * 4, dtype=flat.dtype,
                            device=flat.device).view(1, 1, 4)
    return decode_box_deltas(outputs["pred_boxes"], flat / scale) * scale


def reference_query_indices(model, refs, anchors, target_features, input_size, device):
    """Port of `evaluation/engine.py::_reference_query_indices`.

    For each support case, keep the anchors the detector calls a tight lesion,
    take the one that best overlaps the support's own box, and return the target
    anchor whose class embedding is most similar to it.
    """
    images = torch.cat([r["tensor"] for r in refs]).to(device)
    boxes = torch.tensor([r["bbox"] for r in refs], dtype=torch.float32, device=device)
    count = images.shape[0]
    outputs = model(images, anchors=anchors[:count])
    decoded = decode_boxes(outputs, anchors, count, input_size)
    logits = outputs["pred_logits"].sigmoid()
    features = outputs["class_embeddings"]

    selected = []
    for index in range(count):
        tight = logits[index, :, 0]
        mask = (tight > 0.8) & ((tight - logits[index, :, 1]) > 0.4) & (
            (tight - logits[index, :, 2]) > 0.4
        )
        if mask.any():
            candidate_boxes, candidates = decoded[index][mask], features[index][mask]
        else:
            top = tight.argmax()
            candidate_boxes = decoded[index][top].unsqueeze(0)
            candidates = features[index][top].unsqueeze(0)
        best = box_iou(candidate_boxes, boxes[index].unsqueeze(0)).argmax()
        reference_feature = F.normalize(candidates[best], dim=-1)
        selected.append(int((target_features[0] @ reference_feature).argmax()))
    return selected


def letterbox_params(size, input_size):
    """Scale and padding that `resize_pad_only` applies, so boxes can be mapped back."""
    width, height = size
    scale = min(input_size / width, input_size / height)
    pad_x = (input_size - int(width * scale)) // 2
    pad_y = (input_size - int(height * scale)) // 2
    return scale, pad_x, pad_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", default="../ckpts/b_rad_det.pt")
    ap.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "configs")
    ap.add_argument("--class-index", type=int, default=0,
                    help="0 = tight lesion box, 1 = margin box, 2 = shadow box")
    ap.add_argument("--references", nargs="*", default=[], metavar="IMAGE:MASK",
                    help="Support cases retrieved by B-RAD-CLIP; enables few-shot")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    config = load_config("evaluate", fold=0, config_dir=args.config_dir)
    # The evaluate stage carries no `training` section; anchors are replicated per
    # batch entry, so this must cover the query plus every support reference.
    config.setdefault("training", {"batch_size": max(1, len(args.references))})
    input_size = config["model"]["backbone"]["input_size"]

    device = args.device if torch.cuda.is_available() else "cpu"
    model, anchors, _ = build_model_and_anchors(config, device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    model.eval()

    image = Image.open(args.image).convert("RGB")
    tensor, _ = prepare_image_tensor(image, [0, 0, 1, 1], input_size)
    tensor = tensor.unsqueeze(0).to(device)

    refs = []
    for spec in args.references:
        image_path, _, mask_path = spec.partition(":")
        bbox = bbox_from_mask(mask_path) if mask_path else None
        if bbox is None:
            print(f"[warn] no lesion in {mask_path or image_path}; skipping reference")
            continue
        ref_image = Image.open(image_path).convert("RGB")
        ref_tensor, ref_bbox = prepare_image_tensor(ref_image, bbox, input_size)
        refs.append({"tensor": ref_tensor.unsqueeze(0), "bbox": ref_bbox.tolist()})

    with torch.no_grad():
        outputs = model(tensor, anchors=anchors[:1])
        decoded = decode_boxes(outputs, anchors, 1, input_size)[0]
        scores = outputs["pred_logits"].sigmoid()[0][:, args.class_index]
        if refs:
            target_features = F.normalize(outputs["class_embeddings"], dim=-1)
            indices = reference_query_indices(model, refs, anchors, target_features,
                                              input_size, device)
            # Keep the support-guided anchor the detector scores highest.
            best = max(indices, key=lambda i: float(scores[i]))
            mode = f"few-shot ({len(refs)} references)"
        else:
            best = int(scores.argmax())
            mode = "zero-shot"
        box = decoded[best].clamp(0, input_size).cpu().tolist()

    # Undo the letterbox so the box is in the original image's pixel coordinates.
    scale, pad_x, pad_y = letterbox_params(image.size, input_size)
    x0, y0, x1, y1 = box
    bbox = [round((x0 - pad_x) / scale, 1), round((y0 - pad_y) / scale, 1),
            round((x1 - pad_x) / scale, 1), round((y1 - pad_y) / scale, 1)]
    bbox = [max(0.0, min(bbox[0], image.width)), max(0.0, min(bbox[1], image.height)),
            max(0.0, min(bbox[2], image.width)), max(0.0, min(bbox[3], image.height))]

    name = CLASSES[args.class_index] if args.class_index < len(CLASSES) else str(args.class_index)
    print(f"Image: {Path(args.image).name}   ({image.width}x{image.height})   [{mode}]")
    print(f"Predicted {name} box (xyxy): [{', '.join(f'{v:.1f}' for v in bbox)}]")
    print(f"  score {float(scores[best]):.4f}   anchor #{best} of {len(scores)}")

    result = {"image": Path(args.image).name, "class": name, "mode": mode,
              "bbox_xyxy": bbox, "score": float(scores[best])}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
