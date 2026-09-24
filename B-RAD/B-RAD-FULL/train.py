"""Train the B-RAD-FULL BI-RADS head on top of a frozen B-RAD-CLIP encoder.

The encoder stays frozen: every case is embedded once (image under the ROI foveal
bias, concatenated with its report embedding), then a small MLP is trained per
fold with focal loss and class weights. The five folds are written back into one
checkpoint as `mlp_heads`, which is exactly what `infer_birads.py` consumes.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from brad_clip import BRADClip
from infer_birads import BIRADS, MODE_CFG, FusionMLP, embed


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()


def build_embeddings(model, df, root, mode):
    """One forward pass per case; the encoder is frozen so this is done once."""
    out = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="embedding"):
        out.append(embed(model, str(root / r["image"]), mode,
                         mask_path=str(root / r["mask"]), text=str(r["medgemma_txt"]))[0])
    return np.stack(out)


def train_fold(x_tr, y_tr, x_va, y_va, cfg, args, device):
    scaler = StandardScaler().fit(x_tr)
    xt = torch.tensor(scaler.transform(x_tr), dtype=torch.float32, device=device)
    xv = torch.tensor(scaler.transform(x_va), dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.long, device=device)
    yv = torch.tensor(y_va, dtype=torch.long, device=device)

    counts = np.bincount(y_tr, minlength=len(BIRADS)).astype(np.float32)
    weights = torch.tensor(len(y_tr) / (len(BIRADS) * np.maximum(counts, 1)), device=device)

    net = FusionMLP(cfg["input_dim"], cfg["hidden"], cfg["dropout"]).to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)
    criterion = FocalLoss(alpha=weights, gamma=2.0)

    best_f1, best_state, stale = -1.0, None, 0
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            if len(idx) < 2:            # BatchNorm needs >1 sample
                continue
            optim.zero_grad(set_to_none=True)
            criterion(net(xt[idx]), yt[idx]).backward()
            optim.step()
        sched.step()

        net.eval()
        with torch.no_grad():
            pred = net(xv).argmax(1).cpu().numpy()
        f1 = f1_score(yv.cpu().numpy(), pred, average="weighted")
        if f1 > best_f1:
            best_f1, best_state, stale = f1, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(xv).argmax(1).cpu().numpy()
    y_true = yv.cpu().numpy()
    return {
        "mlp": {k: v.cpu() for k, v in best_state.items()},
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "label_to_idx": {b: i for i, b in enumerate(BIRADS)},
        "metrics": {"acc": accuracy_score(y_true, pred),
                    "f1": f1_score(y_true, pred, average="weighted")},
        "confusion_matrix": confusion_matrix(y_true, pred, labels=range(len(BIRADS))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, help="Frozen B-RAD-CLIP encoder checkpoint")
    ap.add_argument("--data-csv", required=True, help="CSV with image,mask,birads (paths relative to --data-root)")
    ap.add_argument("--text-csv", required=True, help="CSV with image_name,medgemma_txt")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="../ckpts/b_rad_full_retrained.pt")
    ap.add_argument("--mode", choices=list(MODE_CFG), default="concat")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = MODE_CFG[args.mode]

    df = pd.read_csv(args.data_csv)
    txt = pd.read_csv(args.text_csv)
    df["image_name"] = df["image"].map(lambda p: Path(p).name)
    df = df.merge(txt[["image_name", "medgemma_txt"]], on="image_name", how="left")
    df["y"] = df["birads"].astype(str).str[0].astype(int).map({b: i for i, b in enumerate(BIRADS)})
    print(f"cases={len(df)}  class counts={np.bincount(df.y, minlength=len(BIRADS)).tolist()}")

    model = BRADClip(args.encoder, device=args.device, load_text=(args.mode == "concat"))
    x = build_embeddings(model, df, Path(args.data_root), args.mode)
    y = df["y"].to_numpy()

    heads = []
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(skf.split(x, y)):
        head = train_fold(x[tr], y[tr], x[va], y[va], cfg, args, args.device)
        heads.append(head)
        print(f"fold {fold}: acc={head['metrics']['acc']:.4f}  f1={head['metrics']['f1']:.4f}")

    ckpt = torch.load(args.encoder, map_location="cpu", weights_only=False)
    ckpt["mlp_heads"] = heads
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"saved encoder + {len(heads)} heads -> {args.out}")


if __name__ == "__main__":
    main()
