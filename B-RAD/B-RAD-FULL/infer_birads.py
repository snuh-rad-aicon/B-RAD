"""B-RAD-FULL: ROI-guided BI-RADS classification.

Embeds the image with B-RAD-CLIP under an ROI foveal-attention bias (the
segmentation mask's per-patch coverage), fuses it with the report-text embedding, and predicts a
BI-RADS {2,3,4,5} probability with the 5-fold MLP ensemble bundled in the checkpoint.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from brad_clip import BRADClip, foveal_bias_from_mask

BIRADS = [2, 3, 4, 5]
# Representative configuration (region-aware / concat), matching the paper eval.
MODE_CFG = {
    "concat": dict(input_dim=1024, hidden=[128, 64], dropout=0.80),
    "image_only": dict(input_dim=512, hidden=[512, 256], dropout=0.45),
}


class FusionMLP(nn.Module):
    def __init__(self, input_dim, hidden, dropout, num_classes=4):
        super().__init__()
        layers, prev = [nn.Dropout(dropout * 0.5)], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def trace(label, tensor=None, note=""):
    """Print the shape of an intermediate quantity, tagged with its symbol in the paper."""
    shape = tuple(tensor.shape) if tensor is not None else "--"
    print(f"  {label:<34} {str(shape):<18} {note}")


def roi_bias(model, mask_path, verbose=False):
    """Foveal attention bias from the segmentation mask (paper Eq. 5)."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or (mask > 0).sum() == 0:
        return None
    bias = foveal_bias_from_mask(mask, model.grid_side,
                                 n_prefix_tokens=1 + model.n_prompt, device=model.device)
    if verbose:
        n_prefix = 1 + model.n_prompt
        row = bias[0, 0, 0]
        lesion = int((row[n_prefix:] > 0).sum())
        trace("mask M", torch.as_tensor(mask), "binary lesion mask, image resolution")
        trace("B_foveal  (Eq. 5)", bias,
              f"CLS row; {lesion}/{model.grid_side ** 2} lesion tokens, max {float(row.max()):.2f}")
    return bias


def embed(model, image_path, mode, mask_path=None, text=None, verbose=False):
    img = model.load_image(image_path)
    if verbose:
        n_prefix = 1 + model.n_prompt
        print("\n[trace] encoder")
        trace("input image", img, "after resize + normalize")
        trace("VPT prompt p", model.clip.vpt.prompt,
              f"tokens N = 1 CLS + {model.n_prompt} VPT + {model.grid_side ** 2} patches "
              f"= {n_prefix + model.grid_side ** 2}")
    if mode == "concat":
        bias = roi_bias(model, mask_path, verbose=verbose) if mask_path else None
        f_img = model.embed_image(img, attention_bias=bias, normalize=False)
        f_txt = model.embed_text(text or "ultrasonography of a breast lesion", normalize=False)
        fused = torch.cat([f_img, f_txt], dim=1)
        if verbose:
            trace("z_v = g_v(h_v^RoI)", f_img, "2-layer MLP head, ROI-biased attention")
            trace("z_t = g_t(h_t^CLS)", f_txt, "2-layer MLP head, BioLinkBERT [CLS]")
            trace("[z_v ; z_t]", fused, "fused multimodal context vector")
        return fused.cpu().numpy()
    return model.embed_image(img, normalize=True).cpu().numpy()


def classify(embedding, heads, cfg, verbose=False):
    """Average the softmax over the five MLP heads bundled in the checkpoint."""
    if not heads:
        raise SystemExit("Checkpoint has no 'mlp_heads'; re-download b_rad_full.pt.")
    probs = []
    for ck in heads:
        if embedding.shape[1] != len(ck["scaler_mean"]):
            raise SystemExit(
                f"Embedding dim {embedding.shape[1]} does not match the MLP head "
                f"({len(ck['scaler_mean'])}-d). The released heads are for --mode concat "
                f"(1024-d); --mode image_only needs 512-d heads.")
        x = (embedding - ck["scaler_mean"]) / ck["scaler_scale"]
        net = FusionMLP(cfg["input_dim"], cfg["hidden"], cfg["dropout"]).eval()
        net.load_state_dict(ck["mlp"])
        with torch.no_grad():
            probs.append(F.softmax(net(torch.tensor(x, dtype=torch.float32)), dim=1).numpy())
    out = np.mean(probs, axis=0)
    if verbose:
        print("\n[trace] classifier")
        trace("standardized input", torch.as_tensor(x), "per-fold StandardScaler")
        trace("FusionMLP logits", torch.zeros(1, len(BIRADS)),
              f"shallow MLP {cfg['input_dim']} -> {' -> '.join(map(str, cfg['hidden']))} -> {len(BIRADS)}")
        trace("P(y | v, t_gen, M)", torch.as_tensor(out), f"softmax averaged over {len(heads)} folds")
        print()
    return out[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Fine-tuned B-RAD-CLIP (ordinal) checkpoint")
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", default=None, help="Lesion mask for ROI (required for concat mode)")
    ap.add_argument("--text", default=None, help="Report text; or use --text-csv")
    ap.add_argument("--text-csv", default=None, help="CSV with image_name,medgemma_txt")
    ap.add_argument("--mode", choices=list(MODE_CFG), default="concat")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verbose", action="store_true",
                    help="Print the shape of each intermediate quantity with its symbol in the paper")
    args = ap.parse_args()

    text = args.text
    if text is None and args.text_csv:
        df = pd.read_csv(args.text_csv)
        row = df[df["image_name"] == Path(args.image).name]
        if len(row):
            text = str(row.iloc[0]["medgemma_txt"])

    cfg = MODE_CFG[args.mode]
    model = BRADClip(args.ckpt, device=args.device, load_text=(args.mode == "concat"))
    emb = embed(model, args.image, args.mode, mask_path=args.mask, text=text, verbose=args.verbose)
    probs = classify(emb, model.mlp_heads, cfg, verbose=args.verbose)
    pred = BIRADS[int(probs.argmax())]

    print(f"Image: {Path(args.image).name}   mode: {args.mode}")
    print(f"Predicted BI-RADS: {pred}")
    for b, p in zip(BIRADS, probs):
        print(f"  BI-RADS {b}: {p:.4f}")

    result = {"image": Path(args.image).name, "pred_birads": pred,
              **{f"prob_birads{b}": float(p) for b, p in zip(BIRADS, probs)}}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
