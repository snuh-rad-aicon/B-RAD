"""Embed a corpus of images with B-RAD-CLIP into a retrieval reference bank.

Input CSV columns: uid, image (path relative to --data-root), birads, label.
With --text-csv (uid, medgemma_txt) the report embeddings are stored too, so
retrieval can fuse the image and text similarities as in the paper.

Output .npz: embeddings [N,512] and text_embeddings [N,512] (both L2-normalized),
uid, image, birads, label.

With --thumbs, also writes reference_thumbs.npz next to it: small JPEG previews
so the retrieval gallery renders without the full image corpus on disk.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from brad_clip import BRADClip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Fine-tuned B-RAD-CLIP checkpoint (.pt)")
    ap.add_argument("--corpus-csv", required=True, help="CSV with uid,image,birads,label")
    ap.add_argument("--text-csv", default=None, help="CSV with uid,medgemma_txt (enables text embeddings)")
    ap.add_argument("--data-root", required=True, help="Root that image paths are relative to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--thumbs", type=int, default=0,
                    help="Also write reference_thumbs.npz with NxN JPEG previews (e.g. 128)")
    args = ap.parse_args()

    model = BRADClip(args.ckpt, device=args.device, load_text=bool(args.text_csv))
    df = pd.read_csv(args.corpus_csv)
    root = Path(args.data_root)
    if args.text_csv:
        txt = pd.read_csv(args.text_csv)[["uid", "medgemma_txt"]]
        df = df.merge(txt, on="uid", how="left")

    embs, temb = [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="embedding corpus"):
        img = model.load_image(root / r["image"])
        embs.append(model.embed_image(img).cpu().numpy()[0])
        if args.text_csv:
            temb.append(model.embed_text(str(r["medgemma_txt"])).cpu().numpy()[0])

    extra = {"text_embeddings": np.stack(temb).astype(np.float32)} if temb else {}
    np.savez(
        args.out,
        embeddings=np.stack(embs).astype(np.float32),
        **extra,
        uid=df["uid"].to_numpy(),
        image=df["image"].to_numpy(),
        birads=df["birads"].astype(str).to_numpy(),
        label=df["label"].to_numpy(),
    )
    print(f"Saved bank with {len(df)} references -> {args.out}")

    if args.thumbs:
        blobs = []
        for _, r in tqdm(df.iterrows(), total=len(df), desc="thumbnails"):
            im = cv2.imread(str(root / r["image"]), cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, (args.thumbs, args.thumbs), interpolation=cv2.INTER_AREA)
            blobs.append(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes())
        out = Path(args.out).with_name("reference_thumbs.npz")
        np.savez_compressed(
            out,
            thumb=np.frombuffer(b"".join(blobs), dtype=np.uint8),
            offset=np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64),
            uid=df["uid"].astype(str).to_numpy(dtype="U"),   # loads without allow_pickle
        )
        print(f"Saved {len(blobs)} thumbnails -> {out}")


if __name__ == "__main__":
    main()
