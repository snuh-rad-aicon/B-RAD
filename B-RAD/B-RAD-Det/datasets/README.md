# Dataset preparation

This directory contains the public metadata and command-line entry points needed to reconstruct the B-RAD-Det data tree. Original ultrasound images and masks are not redistributed. Review each provider's terms before downloading.

All paths stored in the released CSV files are relative paths. Commands accept an explicit data root, so no workstation-specific absolute path is embedded in the release.

## Files

```text
datasets/
├── README.md
├── BUSCOT_ACCESS.md
├── download_metadata.sh      fetch the released metadata (run this first)
├── download.py
├── prepare.py
├── build_splits.py
├── split_references.py
└── merge_references.py
```

`download_metadata.sh` writes two trees, neither of which is checked into git:

```text
datasets/metadata/                     reference tables
├── brad_det_annotations.csv
├── retrieval_top100.csv
├── fewshot_splits.csv
├── fewshot_references.csv
├── paper_crawl.csv
├── medgemma_text.csv
├── buscot_top100_tight_subset.csv
└── buscot_retrieval_top100.csv

data/metadata/splits/                  runtime manifests read by the configs
├── base/metadata_{train,valid}_fold{0..4}.csv
├── fewshot/metadata_{train,valid}_fold{0..4}.csv
├── evaluation/metadata_valid_fold{0..4}.csv
└── metadata_test.csv
```

- `brad_det_annotations.csv` contains 4,325 query/annotation rows and excludes retrieval lists.
- `retrieval_top100.csv` contains 100 ranked reference paths per lesion-level query. `(image_path, mask_path)` is the join key because BUSI can contain multiple masks for one image.
- `fewshot_splits.csv` preserves the exact stage-2 fold memberships, which differ from base folds 1-4.
- `fewshot_references.csv` stores the fold-specific non-normal top-1 reference actually consumed during one-shot training.
- `paper_crawl.csv` contains the relative image, selected mask, label, and bbox paths for the released paper-crawl reference bundle.
- `medgemma_text.csv` contains the generated pseudo-report per lesion-level sample. The generation
  code is not included; the prompts are in the paper's supplementary material.
- `buscot_top100_tight_subset.csv` contains a ten-query BUS-CoT schema-validation subset. It is not the full evaluation cohort.
- `buscot_retrieval_top100.csv` contains the 100 ranked references and similarities for each query in that schema-validation subset.
- The `data/metadata/splits/` tree is the materialized output of `build_splits.py`. It is released directly so the merge and split-construction steps can be skipped.

## Environment

Create and activate the release environment from the repository root.

```bash
conda env create -f environment.yml
conda activate b-rad
cd B-RAD/B-RAD-Det
```

## Download the released metadata

All metadata lives in the public
[B-RAD-Det dataset repository](https://huggingface.co/datasets/Subin94/B-RAD-Det-Dataset)
(public and ungated, so no Hugging Face token is needed):

```bash
bash datasets/download_metadata.sh
```

Remote-to-local mapping:

| Dataset repository path | Local path |
|---|---|
| `metadata/annotations/brad_det_annotations.csv` | `datasets/metadata/brad_det_annotations.csv` |
| `metadata/retrieval/retrieval_top100.csv` | `datasets/metadata/retrieval_top100.csv` |
| `metadata/retrieval/fewshot_references.csv` | `datasets/metadata/fewshot_references.csv` |
| `metadata/folds/fewshot_splits.csv` | `datasets/metadata/fewshot_splits.csv` |
| `metadata/folds/paper_crawl.csv` | `datasets/metadata/paper_crawl.csv` |
| `metadata/generated_text/medgemma_text.csv` | `datasets/metadata/medgemma_text.csv` |
| `metadata/folds/buscot_*.csv`, `metadata/retrieval/buscot_*.csv` | `datasets/metadata/buscot_*.csv` |
| `metadata/runtime/**` | `data/metadata/splits/**` |

The runtime tree resolves the paths the configs already reference, so after this
command and the image conversion below, `scripts/evaluate.py` runs without any
further metadata step. The reference tables under `datasets/metadata/` are only
needed to rebuild that tree or to inspect the released retrieval lists.

`paper_crawl.csv` is copied to `data/metadata/paper_crawl.csv` as well, because
`train_fewshot.yaml` reads it from there while `evaluate.yaml` reads it from
`datasets/metadata/`.

## Download the source datasets

The download command requires `curl`, `unzip`, and `git`. It displays the source URL, target path, progress bar, byte count, SHA-256 digest, retries, resume status, and ZIP integrity result.

## Download

Download all public datasets:

```bash
python datasets/download.py all --output-dir data/downloads
```

Or select individual datasets:

```bash
python datasets/download.py bus_bra busi bus_uclm busi_whu \
  breast_usg udiat bus_stu \
  --output-dir data/downloads
```

Supported names are `bus_bra`, `busi`, `bus_uclm`, `busi_whu`, `breast_usg`, `udiat`, and `bus_stu`.

`busi`, `bus_uclm`, `busi_whu`, and `udiat` resolve to the Kaggle dataset API.
Some of those datasets reject unauthenticated requests, `udiat` among them at the
time of writing. Credentials are attached automatically when they are available,
so export them before running those names:

```bash
export KAGGLE_USERNAME=<your kaggle username>
export KAGGLE_KEY=<your kaggle api key>
```

`~/.kaggle/kaggle.json` is read as a fallback. If a Kaggle download still fails,
fetch that archive manually from the provider page listed in the repository
README and place it in `data/downloads/` under the filename shown in the
extraction commands below.

BUS-CoT is not downloaded by this command. The processed held-out evaluation package used by B-RAD was provided directly by the BUS-CoT collaborators and is not redistributed here. To reproduce that evaluation, request the package from the corresponding authors of the BUS-CoT dataset paper. See [BUSCOT_ACCESS.md](BUSCOT_ACCESS.md).

### Optional OOD evaluation bundle

The prepared 457-case OOD archive is not redistributed in the current release.
Reconstruct these inputs from the original providers with the download and
conversion commands above. If a redistribution-cleared archive is published in
a later release, `datasets/download.py ood_bundle` can verify its checksum,
validate archive paths, and extract it into the relative project path
`data/processed`.

The archive must use the same relative paths as the released OOD metadata. Its top-level layout is:

```text
merged/
├── Breast_Lesions_USG/
├── UDIAT_DatasetB/
└── Breast_STU/
```

### Optional paper-crawl reference bundle

The paper-crawl image and pseudo-mask archive is not redistributed in the
current release. `metadata/paper_crawl.csv` documents the 2,436 selected
references and their expected relative paths. If a redistribution-cleared
archive is published in a later release, `datasets/download.py
paper_crawl_bundle` can validate and extract it into the processed data root.

BUS-UCLM requires the dataset version used by the paper. The currently available Kaggle mirror differs from the experiment source in 13 `HESN` images, including four lesion images. Use the official BUS-UCLM Version 1 archive for exact reproduction; the converter rejects ambiguous source trees but cannot infer the provider version from filenames alone.

## Extract

Extract archives without renaming their contents:

```bash
mkdir -p data/raw/bus_bra data/raw/busi data/raw/bus_uclm \
  data/raw/busi_whu data/raw/breast_usg data/raw/udiat

unzip data/downloads/BUSBRA.zip -d data/raw/bus_bra
unzip data/downloads/Dataset_BUSI.zip -d data/raw/busi
unzip data/downloads/BUS-UCLM.zip -d data/raw/bus_uclm
unzip data/downloads/BUS-WHU.zip -d data/raw/busi_whu
unzip data/downloads/BrEaST-Lesions_USG.zip -d data/raw/breast_usg
unzip data/downloads/udiat-data.zip -d data/raw/udiat
```

`bus_stu` is cloned directly to `data/downloads/STU-Hospital`.

## Convert

The converter reads the released manifest, writes the paths expected by the original code, and independently calculates a bbox audit column. It never edits files under `data/raw`.

```bash
python datasets/prepare.py \
  --dataset bus_bra \
  --raw-dir data/raw/bus_bra \
  --output-dir data/processed \
  --manifest datasets/metadata/brad_det_annotations.csv \
  --output-metadata data/audit/bus_bra.csv
```

Repeat with the other dataset names. BUS-UCLM should point directly to the directory containing `images/` and `masks/`; UDIAT should point to `DatasetB2/`.

The command prints the number of prepared rows and the number of bbox mismatches. A reproducible conversion must report zero bbox mismatches.

## Restore retrieval columns

The release keeps retrieval results separate from query metadata. Restore the `seg_ref_imgs` column expected by the original training and evaluation code with:

```bash
python datasets/merge_references.py \
  --format standard \
  --metadata datasets/metadata/brad_det_annotations.csv \
  --references datasets/metadata/retrieval_top100.csv \
  --output data/metadata/metadata_with_references.csv
```

Restore the BUS-CoT `top1_path` through `top100_path` columns with:

```bash
python datasets/merge_references.py \
  --format buscot \
  --metadata datasets/metadata/buscot_top100_tight_subset.csv \
  --references datasets/metadata/buscot_retrieval_top100.csv \
  --output data/metadata/buscot_top100_tight_subset_with_references.csv
```

This command requires the separately requested BUS-CoT evaluation package. The included tables describe the expected relative-path schema; they do not grant redistribution rights for the BUS-CoT images or the collaborator-prepared evaluation package.

The merge validates complete ranks 1-100, duplicate keys, and missing values. The released split files were round-trip tested against the original metadata.

Generate the original five-fold train/validation files and the 457-row OOD test file:

```bash
python datasets/build_splits.py \
  --metadata data/metadata/metadata_with_references.csv \
  --fewshot-splits datasets/metadata/fewshot_splits.csv \
  --fewshot-references datasets/metadata/fewshot_references.csv \
  --output-dir data/metadata/splits
```

Running this is optional: `download_metadata.sh` already provides the resulting
tree under `data/metadata/splits/`. It also requires a `split_multi` column,
which the released `brad_det_annotations.csv` does not carry for base folds, so
use the downloaded runtime tree unless you are regenerating from private tables.

This command reads the released split memberships. It does not generate a new random split. Base and few-shot outputs are written separately because their historical folds 1-4 are not identical.

The command also writes `data/metadata/splits/evaluation/metadata_valid_fold{fold}.csv`. These evaluation manifests preserve the same few-shot validation membership while retaining the ranked top-100 retrieval list required by the paper's top-4 evaluation. Do not use the top-1 training manifests for Table 5 evaluation.

The base split preserves the original detector preprocessing. `normal` rows are removed. Folds 1-4 remove every row whose `uid` is duplicated (multi-mask BUSI cases); fold 0 train retains those rows because the original fold-0 manifest predates that filtering step. The few-shot split uses `fewshot_splits.csv`, reconstructed from the released fold manifests. `fewshot_references.csv` stores only the single non-normal reference consumed by the one-shot training path; it is distinct from the top-100 evaluation retrieval table. The OOD test split retains repeated UDIAT UIDs and is keyed by image/mask paths.
