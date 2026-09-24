"""Train B-RAD-CLIP: unpaired, BI-RADS-supervised contrastive alignment.

Images and reports are never paired one-to-one. Positives are defined by the
BI-RADS label, both within each modality and across them, so the encoder learns a
single space where an image and a report of the same category attract:

    L_con = SupCon(image) + SupCon(report) + SupCon(image ++ report)
            + CrossModal(image, report)               # label-masked, symmetric
    L_ord = Ordinal(psi_image) + Ordinal(psi_report)  # BI-RADS boundary logits
    L     = L_con + lambda_ord * L_ord

Both projection heads g_v, g_t are 2-layer MLPs and the temperature tau is
learned, shared by the two contrastive terms.

A VPT prompt is prepended to a frozen BiomedCLIP ViT (only the last
`--img-ft-last` blocks are unfrozen); the BioLinkBERT text tower unfreezes its
last `--txt-ft-last` layers. One checkpoint is written per fold.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from brad_clip.backbone import TEXT_ENCODER, VPTVisionWrapper, projection_head
from brad_clip.dataset import MultiModal, collate
from brad_clip.losses import OrdinalHead, SupCon, birads_sim, corn_loss
from brad_clip.model import BiomedCLIPConfig, BiomedCLIPModel

MODEL_DIR = Path(__file__).resolve().parent / "brad_clip" / "model"
BIRADS = [2, 3, 4, 5]          # ordinal ranks 0..3
NUM_CLASSES = len(BIRADS)


def load_tables(csv_split, text_json):
    """Images (image, birads) and the BI-RADS-labelled report corpus."""
    img = pd.read_csv(csv_split)
    if "image_name" in img.columns:
        img = img.rename(columns={"image_name": "image"})
    img = img[["image", "birads"]]
    rows = [json.loads(line) for line in open(text_json, encoding="utf-8")]
    txt = pd.DataFrame(rows)[["birads", "natural_text"]].astype({"birads": int})
    return img, txt


def build_encoders(n_prompt, img_ft_last, txt_ft_last, device, base_model=None):
    """Fine-tuning starts from the pretrained BiomedCLIP checkpoint, not from scratch."""
    base = Path(base_model) if base_model else MODEL_DIR
    if not (list(base.glob("pytorch_model*.bin")) or list(base.glob("*.safetensors"))):
        raise SystemExit(
            f"No BiomedCLIP weights in {base}. Training fine-tunes the pretrained "
            "BiomedCLIP ViT-B/16, so pass --base-model <dir> pointing at a directory "
            "that holds its pytorch_model.bin. (Inference does not need it: the released "
            "checkpoints already carry the full encoder.)")
    clip = BiomedCLIPModel.from_pretrained(
        str(base), config=BiomedCLIPConfig.from_pretrained(str(MODEL_DIR / "config.json")))
    clip.vpt = VPTVisionWrapper(clip.vision_model, n_prompt)
    clip = clip.to(device)
    for layer in clip.vpt.vision.encoder.layers[-img_ft_last:]:
        for p in layer.parameters():
            p.requires_grad = True

    text = AutoModel.from_pretrained(TEXT_ENCODER).to(device)
    for p in text.parameters():
        p.requires_grad = False
    for layer in text.encoder.layer[-txt_ft_last:]:
        for p in layer.parameters():
            p.requires_grad = True

    img_proj = projection_head(clip.vision_model.config.hidden_size, 512).to(device)   # g_v
    txt_proj = projection_head(text.config.hidden_size, 512).to(device)                 # g_t
    img_ord = OrdinalHead(512, NUM_CLASSES).to(device)      # psi_v
    txt_ord = OrdinalHead(512, NUM_CLASSES).to(device)      # psi_t
    return clip, text, img_proj, txt_proj, img_ord, txt_ord


def class_balanced_sampler(img_tr, txt_tr, seed=0):
    """Resample images so each BI-RADS class matches the report-corpus mix.

    One epoch draws as many images as the report corpus holds, so the two
    modalities contribute the same number of samples per epoch.
    """
    img_counts = img_tr["birads"].value_counts()
    txt_counts = txt_tr["birads"].value_counts()
    w = {b: txt_counts.get(b, 0) / max(1, img_counts.get(b, 0)) for b in img_counts.index}
    weights = np.nan_to_num(img_tr["birads"].map(w).to_numpy(dtype="float64"), nan=0.0, posinf=0.0)
    n_samples = int(sum(int(txt_counts.get(b, 0)) for b in img_counts.index))
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, n_samples, replacement=True, generator=gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-split", required=True, help="CSV with image_name,birads")
    ap.add_argument("--text-json", required=True, help="JSONL corpus with birads,natural_text")
    ap.add_argument("--image-dir", required=True, help="Root holding the images")
    ap.add_argument("--base-model", default=None,
                    help="Directory with the pretrained BiomedCLIP weights (pytorch_model.bin)")
    ap.add_argument("--out-dir", default="../ckpts")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=3e-5)
    ap.add_argument("--lr-ft", type=float, default=5e-6)
    ap.add_argument("--temp", type=float, default=0.10, help="Initial (learnable) temperature")
    ap.add_argument("--lambda-ord", type=float, default=0.2,
                    help="Weight on the CORN ordinal term")
    ap.add_argument("--n-prompt", type=int, default=8)
    ap.add_argument("--grad-acc", type=int, default=2)
    ap.add_argument("--img-ft-last", type=int, default=4)
    ap.add_argument("--txt-ft-last", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_img, df_txt = load_tables(args.csv_split, args.text_json)
    print(f"images={len(df_img)}  reports={len(df_txt)}")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
    supcon = SupCon(t=args.temp)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    img_splits = list(skf.split(df_img, df_img.birads))
    txt_splits = list(skf.split(df_txt, df_txt.birads))

    for fold, (tr, va) in enumerate(img_splits):
        ckpt = out_dir / f"b_rad_clip_fold{fold}.pt"
        if ckpt.exists():
            print(f"fold {fold}: exists -> skip")
            continue

        clip, text, img_proj, txt_proj, img_ord, txt_ord = build_encoders(
            args.n_prompt, args.img_ft_last, args.txt_ft_last, args.device, args.base_model)
        # Learnable temperature shared by both contrastive objectives.
        log_tau = torch.nn.Parameter(torch.tensor(float(np.log(args.temp)), device=args.device))

        img_tr, img_va = df_img.iloc[tr].reset_index(drop=True), df_img.iloc[va].reset_index(drop=True)
        t_tr, t_va = txt_splits[fold]
        txt_tr = df_txt.iloc[t_tr].reset_index(drop=True)
        txt_va = df_txt.iloc[t_va].reset_index(drop=True)

        ld_tr = DataLoader(
            MultiModal(img_tr, txt_tr, Path(args.image_dir), tokenizer, is_train=True),
            batch_size=args.batch, sampler=class_balanced_sampler(img_tr, txt_tr, args.seed + fold),
            collate_fn=collate, num_workers=4, drop_last=True)
        ld_va = DataLoader(
            MultiModal(img_va, txt_va, Path(args.image_dir), tokenizer, is_train=False),
            batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=4)

        # Only the VPT vision path is trained on the image side: BiomedCLIP's own
        # text tower and projections take no part in the forward pass.
        params = [p for m in (clip.vpt, text, img_proj, txt_proj, img_ord, txt_ord)
                  for p in m.parameters() if p.requires_grad]
        head = (list(img_proj.parameters()) + list(txt_proj.parameters())
                + list(img_ord.parameters()) + list(txt_ord.parameters())
                + [clip.vpt.prompt, log_tau])
        head_ids = {id(p) for p in head}

        def groups(ps, lr):        # biases and norms are excluded from weight decay
            return [{"params": [p for p in ps if p.ndim > 1], "lr": lr, "weight_decay": 0.01},
                    {"params": [p for p in ps if p.ndim <= 1], "lr": lr, "weight_decay": 0.0}]

        optim = torch.optim.AdamW(
            groups(head, args.lr_head)
            + groups([p for p in params if id(p) not in head_ids], args.lr_ft))
        scaler = GradScaler(args.device)

        def step_loss(imgs, tok, li, lt):
            f_img = img_proj(clip.vpt(imgs))
            f_txt = txt_proj(text(**tok).last_hidden_state[:, 0, :])

            tau = log_tau.exp().clamp(0.01, 1.0)
            loss_con = (supcon(f_img, li, t=tau) + supcon(f_txt, lt, t=tau)
                        + supcon(torch.cat([f_img, f_txt], 0), torch.cat([li, lt], 0), t=tau)
                        + birads_sim(f_img, f_txt, li, lt, t=tau))

            li_ord, lt_ord = li - BIRADS[0], lt - BIRADS[0]          # {2..5} -> {0..3}
            loss_ord = (corn_loss(img_ord(f_img), li_ord, NUM_CLASSES)
                        + corn_loss(txt_ord(f_txt), lt_ord, NUM_CLASSES))
            return loss_con + args.lambda_ord * loss_ord

        def step(optim, scaler):
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(clip.parameters(), 1.0)
            scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)

        for ep in range(1, args.epochs + 1):
            clip.train(); text.train(); img_proj.train(); txt_proj.train()
            img_ord.train(); txt_ord.train()
            tr_loss, accum = 0.0, 0
            optim.zero_grad(set_to_none=True)
            for imgs, tok, li, lt in tqdm(ld_tr, desc=f"fold{fold} ep{ep}", leave=False):
                imgs, li, lt = imgs.to(args.device), li.to(args.device), lt.to(args.device)
                tok = {k: v.to(args.device) for k, v in tok.items()}
                with autocast(args.device):
                    loss = step_loss(imgs, tok, li, lt) / args.grad_acc
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                accum += 1
                if accum % args.grad_acc == 0:
                    step(optim, scaler)
                tr_loss += float(loss) * args.grad_acc
            if accum % args.grad_acc:            # flush a trailing partial group
                step(optim, scaler)

            clip.eval(); text.eval(); img_proj.eval(); txt_proj.eval()
            img_ord.eval(); txt_ord.eval()
            va_loss = 0.0
            with torch.no_grad():
                for imgs, tok, li, lt in ld_va:
                    imgs, li, lt = imgs.to(args.device), li.to(args.device), lt.to(args.device)
                    tok = {k: v.to(args.device) for k, v in tok.items()}
                    with autocast(args.device):
                        va_loss += float(step_loss(imgs, tok, li, lt))
            print(f"fold {fold} | ep {ep:02d} | train {tr_loss/max(1,len(ld_tr)):.4f} "
                  f"| val {va_loss/max(1,len(ld_va)):.4f} | tau {float(log_tau.detach().exp()):.4f}")

        torch.save({
            "img_enc": clip.state_dict(),
            "txt_back": text.state_dict(),
            "proj_img": img_proj.state_dict(),
            "proj_txt": txt_proj.state_dict(),
            "img_ord": img_ord.state_dict(),
            "txt_ord": txt_ord.state_dict(),
            "prompt": clip.vpt.prompt.detach().cpu(),
            "temperature": float(log_tau.detach().exp().clamp(0.01, 1.0)),
        }, ckpt)
        print(f"saved -> {ckpt}")

        # Release the fold's models before the next one (each is ~2 GB on GPU).
        del clip, text, img_proj, txt_proj, img_ord, txt_ord, log_tau, optim, scaler, ld_tr, ld_va
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
