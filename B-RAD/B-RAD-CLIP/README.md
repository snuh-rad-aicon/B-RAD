# B-RAD-CLIP

BI-RADS-aware image-text retrieval. A VPT-adapted BiomedCLIP encoder maps the
image and its MedGemma pseudo-report into one 512-d space, and retrieval ranks
the reference bank by `alpha * cos(image) + (1 - alpha) * cos(report)` with
`alpha = 0.5`. Pass `--alpha 1.0` for image-only retrieval.

The pseudo-reports ship as pre-generated text; MedGemma inference is not
included here. See the paper's supplementary material for the prompts.

## Setup

```bash
pip install -r requirements.txt
```

The text tower (`michiyasunaga/BioLinkBERT-large`) is fetched from the Hugging
Face Hub on first use. Place the encoder at `../ckpts/b_rad_clip.pt`
(download with [../ckpts/download_weights.sh](../ckpts/download_weights.sh)).

## Retrieval

```bash
bash scripts/inference.sh                              # bundled sample
IMAGE=<image.png> ALPHA=1.0 bash scripts/inference.sh  # any image, image only
```

Both wrappers forward extra flags to `infer_retrieval.py`; add `--verbose` to
print the shape of each intermediate quantity.

## Training

Positives are the samples sharing a BI-RADS category, never patient-matched
pairs. The objective sums within-modality SupCon, a joint SupCon over both
modalities, a label-masked symmetric cross-modal term, and an ordinal term on
the BI-RADS boundaries:

```text
L = SupCon(image) + SupCon(report) + SupCon(image ++ report)
    + CrossModal(image, report)
    + lambda_ord * [ Ordinal(psi_image) + Ordinal(psi_report) ]
```

Both projection heads are 2-layer MLPs and the temperature is learned, shared by
the contrastive terms. Training fine-tunes the pretrained BiomedCLIP ViT-B/16,
so point `--base-model` at a directory holding its `pytorch_model.bin`;
`brad_clip/model/` carries only the config and the model code. Inference needs
none of this.

```bash
CSV_SPLIT=<image_name,birads.csv> \
TEXT_JSON=<JSONL with birads,natural_text> \
IMAGE_DIR=<image root> \
bash scripts/train.sh --base-model <dir with pytorch_model.bin>
```

Writes one checkpoint per fold. Defaults match the release: 8 VPT tokens, last 4
ViT blocks and last 2 text layers unfrozen, initial temperature 0.10,
`lambda_ord` 0.2, 50 epochs. One epoch draws as many images as the report corpus
holds, so both modalities contribute equally.

## Reference bank

`assets/reference_bank.npz` is a **demonstration** bank over 252 public BrEaST
cases, not the paper's retrieval corpus. It holds embeddings and metadata only.
`assets/reference_thumbs.npz` holds 128x128 JPEG previews so the gallery renders
without the image corpus on disk; those previews are downscaled from
Breast-Lesions-USG and redistributed under CC BY 4.0. If you use them, cite the
dataset:

> Pawłowska, A., Ćwierz-Pieńkowska, A., Domalik, A., Jaguś, D., Kasprzak, P., Matkowski, R., Fura, Ł., Nowicki, A., & Zolek, N. A Curated benchmark dataset for ultrasound based breast lesion analysis. Sci Data 11, 148 (2024). https://doi.org/10.1038/s41597-024-02984-z.

Build a bank over your own corpus with:

```bash
python build_bank.py --ckpt ../ckpts/b_rad_clip.pt \
  --corpus-csv <uid,image,birads,label.csv> --text-csv <uid,medgemma_txt.csv> \
  --data-root <root> --out assets/reference_bank.npz --thumbs 128
```

## Files

```text
brad_clip/            encoder package (VPT-BiomedCLIP), losses, dataset
train.py              supervised contrastive training (5 folds)
infer_retrieval.py    embed a query image and return top-k references
build_bank.py         embed a corpus into a reference bank
scripts/              train.sh, inference.sh
assets/               reference_bank.npz, reference_thumbs.npz
```
