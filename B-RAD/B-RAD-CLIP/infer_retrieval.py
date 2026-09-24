"""B-RAD-CLIP retrieval: embed a query case and return the top-k references.

The encoder is trained with unpaired, BI-RADS-supervised contrastive learning, so
image and report embeddings share one space. Retrieval fuses both similarities:

    S = alpha * cos(image) + (1 - alpha) * cos(report)

with the report being the MedGemma pseudo-report of the query. Set --alpha 1.0
for image-only retrieval. Exact self-matches (same uid) are excluded.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from brad_clip import BRADClip


def trace(label, tensor=None, note=""):
    """Print the shape of an intermediate quantity, tagged with its symbol in the paper."""
    shape = tuple(tensor.shape) if tensor is not None else "--"
    print(f"  {label:<30} {str(shape):<16} {note}")


def retrieve(query_emb, bank, topk, exclude_uid=None, query_text_emb=None, alpha=0.5,
             verbose=False):
    sims = bank["embeddings"] @ query_emb
    if verbose:
        print("\n[trace] retrieval")
        trace("bank image embeddings", bank["embeddings"], "L2-normalized z_v")
        trace("query z_v", query_emb, "L2-normalized")
        if query_text_emb is not None and "text_embeddings" in bank:
            trace("query z_t", query_text_emb, "MedGemma pseudo-report")
            trace("fused similarity S", sims,
                  f"alpha*cos(z_v) + (1-alpha)*cos(z_t), alpha={alpha}")
        else:
            trace("similarity S", sims, "cos(z_v) only")
        print()
    if query_text_emb is not None and "text_embeddings" in bank:
        sims = alpha * sims + (1.0 - alpha) * (bank["text_embeddings"] @ query_text_emb)
    out = []
    for i in np.argsort(-sims):
        uid = str(bank["uid"][i])
        if exclude_uid is not None and uid == exclude_uid:
            continue
        out.append({
            "uid": uid,
            "image": str(bank["image"][i]),
            "birads": str(bank["birads"][i]),
            "label": str(bank["label"][i]),
            "similarity": float(sims[i]),
        })
        if len(out) >= topk:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bank", required=True, help="reference_bank.npz from build_bank.py")
    ap.add_argument("--image", required=True)
    ap.add_argument("--text", default=None, help="Query report text; or use --text-csv")
    ap.add_argument("--text-csv", default=None, help="CSV with image_name,medgemma_txt")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Weight on image similarity; 1.0 = image only")
    ap.add_argument("--uid", default=None, help="Query uid to exclude from results")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default=None, help="Optional JSON output path")
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
    use_text = text is not None and args.alpha < 1.0

    model = BRADClip(args.ckpt, device=args.device, load_text=use_text)
    query = model.embed_image(model.load_image(args.image)).cpu().numpy()[0]
    query_text = model.embed_text(text).cpu().numpy()[0] if use_text else None

    bank = np.load(args.bank, allow_pickle=True)
    results = retrieve(query, bank, args.topk, exclude_uid=args.uid,
                       query_text_emb=query_text, alpha=args.alpha, verbose=args.verbose)

    mode = f"image+text (alpha={args.alpha})" if use_text else "image only"
    print(f"Top-{args.topk} references for {Path(args.image).name}  [{mode}]:")
    for rank, r in enumerate(results, 1):
        print(f"  {rank}. {r['uid']:>12}  BI-RADS {r['birads']:<3} {r['label']:<10} sim={r['similarity']:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"Saved -> {args.out}")

if __name__ == "__main__":
    main()
