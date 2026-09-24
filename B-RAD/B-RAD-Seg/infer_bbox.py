"""B-RAD-Seg: bbox-prompted lesion segmentation with UltraSam (zero-shot).

Given an image and a box prompt (the B-RAD-Det prediction), UltraSam produces a
lesion mask. If a ground-truth mask is supplied, Dice and IoU are reported.
Run inside the `ultrasam` environment (mmdet/mmcv/mmpretrain).
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py"


def apply_margin(bbox, pct, w, h):
    x0, y0, x1, y1 = bbox
    mw, mh = (x1 - x0) * pct / 100.0, (y1 - y0) * pct / 100.0
    return [max(0, x0 - mw), max(0, y0 - mh), min(w, x1 + mw), min(h, y1 + mh)]


def build_coco(image_path, bbox_xyxy, margin, out_json):
    h, w = cv2.imread(str(image_path)).shape[:2]
    x0, y0, x1, y1 = apply_margin(bbox_xyxy, margin, w, h)
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    coco = {
        "images": [{"id": 0, "file_name": str(image_path), "width": w, "height": h}],
        "annotations": [{
            "id": 0, "image_id": 0, "category_id": 1, "iscrowd": 0,
            "bbox": [x0, y0, bw, bh], "area": bw * bh,
            "segmentation": [[x0, y0, x0 + bw, y0, x0 + bw, y0 + bh, x0, y0 + bh]],
        }],
        "categories": [{"id": 1, "name": "object"}],
    }
    json.dump(coco, open(out_json, "w"))
    return (h, w)


def run_mim(config, checkpoint, ann_json, work_dir, device):
    env = dict(os.environ, PYTHONPATH=f"{MODULE_DIR}:{os.environ.get('PYTHONPATH', '')}")
    # mmdet test.py exposes no --device flag; steer the device via CUDA_VISIBLE_DEVICES.
    if device.startswith("cuda"):
        env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else "0"
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU fallback in mmengine.get_device()
    cmd = [
        sys.executable, "-m", "mim", "test", "mmdet", str(config),
        "--checkpoint", str(checkpoint),
        "--cfg-options",
        "test_dataloader.dataset.data_root=/",
        f"test_dataloader.dataset.ann_file={ann_json}",
        "test_dataloader.dataset.data_prefix.img=",
        f"test_evaluator.ann_file={ann_json}",
        "--work-dir", str(work_dir),
        "--out", str(Path(work_dir) / "results.pkl"),
    ]
    proc = subprocess.run(cmd, cwd=MODULE_DIR, env=env, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f"UltraSam inference failed (exit {proc.returncode})")
    return Path(work_dir) / "results.pkl"


def extract_mask(result, shape):
    h, w = shape
    pi = result["pred_instances"] if isinstance(result, dict) else result.pred_instances
    masks = pi["masks"] if isinstance(pi, dict) else pi.masks
    if hasattr(masks, "cpu"):
        masks = masks.cpu().numpy()
    if len(masks) == 0:
        return np.zeros((h, w), np.uint8)
    m = masks[0]  # best-scoring
    if isinstance(m, dict):
        m = mask_utils.decode(m)
    m = np.asarray(m)
    if m.ndim == 3:
        m = m[0]
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def dice_iou(pred, gt):
    p, g = (pred > 0).astype(np.uint8), (gt > 0).astype(np.uint8)
    inter = int((p * g).sum())
    denom, union = int(p.sum() + g.sum()), int(p.sum() + g.sum() - inter)
    return (2 * inter / denom if denom else 0.0), (inter / union if union else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--bbox", required=True, help="Prompt box 'x0,y0,x1,y1' (B-RAD-Det prediction)")
    ap.add_argument("--ckpt", required=True, help="UltraSam.pth")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--gt-mask", default=None, help="Optional GT mask for Dice/IoU")
    ap.add_argument("--out-mask", required=True, help="Output predicted mask PNG")
    ap.add_argument("--out", default=None, help="Optional JSON metrics path")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    bbox = [float(v) for v in args.bbox.split(",")]
    image = Path(args.image).resolve()  # absolute paths required: data_root is "/"
    out_mask = Path(args.out_mask).resolve()
    work_dir = (Path(args.work_dir) if args.work_dir else out_mask.parent / "_seg_tmp").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    ann = work_dir / "prompt.json"

    shape = build_coco(image, bbox, args.margin, ann)
    pkl = run_mim(Path(args.config).resolve(), Path(args.ckpt).resolve(), ann, work_dir, args.device)
    result = pickle.load(open(pkl, "rb"))[0]
    pred = extract_mask(result, shape)

    out_mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_mask), pred * 255)

    out = {"image": image.name, "prompt_bbox": bbox, "mask": str(out_mask)}
    if args.gt_mask:
        gt = cv2.imread(str(args.gt_mask), cv2.IMREAD_GRAYSCALE)
        if gt.shape != pred.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        d, i = dice_iou(pred, gt)
        out["dice"], out["iou"] = round(d, 4), round(i, 4)
        print(f"{image.name}  Dice={d:.4f}  IoU={i:.4f}")
    else:
        print(f"{image.name}  mask -> {out_mask}")

    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
