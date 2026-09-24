# B-RAD-Det checkpoints

Model checkpoints are hosted in the [B-RAD-Det model repository](https://huggingface.co/Subin94/B-RAD-Det).

## Required top-4 inference checkpoints

The paper evaluation uses the five `best` few-shot stage-2 checkpoints.

| Fold | Filename | SHA-256 | Download URL |
|---:|---|---|---|
| 0 | `brad_det_fewshot_1shot_best_fold0.pth` | `1a6a86433d7fc381c20a380a42217d1bf1ba4e9558a3becabe28b4ee9656e4de` | [download](https://huggingface.co/Subin94/B-RAD-Det/resolve/main/brad_det_fewshot_1shot_best_fold0.pth) |
| 1 | `brad_det_fewshot_1shot_best_fold1.pth` | `ad35dfe41661df34db643182b61cdb610e99663180adadf10d363f3949cb17fa` | [download](https://huggingface.co/Subin94/B-RAD-Det/resolve/main/brad_det_fewshot_1shot_best_fold1.pth) |
| 2 | `brad_det_fewshot_1shot_best_fold2.pth` | `a3c1fe71078e453f6adb63fc2c78ecf6c7f22fba65c33aca46c93228cc8b9d0c` | [download](https://huggingface.co/Subin94/B-RAD-Det/resolve/main/brad_det_fewshot_1shot_best_fold2.pth) |
| 3 | `brad_det_fewshot_1shot_best_fold3.pth` | `c331fa3fa64cb8bc280c25203d7863a2b04165ffe6be628e152ab8e8211b4476` | [download](https://huggingface.co/Subin94/B-RAD-Det/resolve/main/brad_det_fewshot_1shot_best_fold3.pth) |
| 4 | `brad_det_fewshot_1shot_best_fold4.pth` | `0183051d29d33197c34230d7164977bc6bdcfc1ae0a40c68db597455184c5db7` | [download](https://huggingface.co/Subin94/B-RAD-Det/resolve/main/brad_det_fewshot_1shot_best_fold4.pth) |

Place them under:

```text
models/checkpoints/fewshot/
```

## Required zero-shot inference checkpoints

Zero-shot release evaluation uses the five fold-matched base `best` checkpoints.

| Fold | Filename | SHA-256 | Download URL |
|---:|---|---|---|
| 0 | `brad_det_base_best_fold0.pth` | `caa972c42c03f75177ba5690c780800cffbab7de94f5ed12fc4515c9141112a5` | [download](https://huggingface.co/Subin94/B-RAD-Det/resolve/main/brad_det_base_best_fold0.pth) |
| 1 | `brad_det_base_best_fold1.pth` | `5e6950a26a03e4eb09090b0669250c8a68f06de31c42fec03721fd03fbd1d8e0` | [download](https://huggingface.co/Subin94/B-RAD-Det/resolve/main/brad_det_base_best_fold1.pth) |
| 2 | `brad_det_base_best_fold2.pth` | `100ef26d81652bb4a8a4585e80ee1fea971826d9e3a9469ba0dec86cea4fd1d0` | [download](https://huggingface.co/Subin94/B-RAD-Det/resolve/main/brad_det_base_best_fold2.pth) |
| 3 | `brad_det_base_best_fold3.pth` | `26b94c1a602b114d7d96c5bc7e37f380cc10b6f5b8dd5ca12d333ec015dbb506` | [download](https://huggingface.co/Subin94/B-RAD-Det/resolve/main/brad_det_base_best_fold3.pth) |
| 4 | `brad_det_base_best_fold4.pth` | `252a4c40c96bf7fbbdd8ccfdb0f1c3eab9d7cfab601edcc438bb42a339d60975` | [download](https://huggingface.co/Subin94/B-RAD-Det/resolve/main/brad_det_base_best_fold4.pth) |

Place them under:

```text
models/checkpoints/base/
```

## Optional base checkpoints for training reproduction

Few-shot training starts from the fold-matched base `final` checkpoint.

| Fold | Filename | SHA-256 | Download URL |
|---:|---|---|---|
| 0 | `brad_det_base_final_fold0.pth` | `2f73aa10e80ab76cb0c5f5397bcd80faf5b86139b38c8f2d6c978ea815a31638` | [download](https://huggingface.co/Subin94/B-RAD-Det/resolve/main/brad_det_base_final_fold0.pth) |
| 1 | `brad_det_base_final_fold1.pth` | `81a965c71dfe42be80ff8294fb8f4f0f9c042fd3525228b902641086ac5cde21` | [download](https://huggingface.co/Subin94/B-RAD-Det/resolve/main/brad_det_base_final_fold1.pth) |
| 2 | `brad_det_base_final_fold2.pth` | `ca16456519704cde327a045e9b5c5c7892be4ebb90a238fee4a07c9a48c72d89` | [download](https://huggingface.co/Subin94/B-RAD-Det/resolve/main/brad_det_base_final_fold2.pth) |
| 3 | `brad_det_base_final_fold3.pth` | `f1e6abd24b455d747d57b83b4a809e0153738e6e0d91bacf8932de2b476a9e0c` | [download](https://huggingface.co/Subin94/B-RAD-Det/resolve/main/brad_det_base_final_fold3.pth) |
| 4 | `brad_det_base_final_fold4.pth` | `029e746a317f0679117ad9b971fa09d68f47dffcc78951271c45d0d994d2ef87` | [download](https://huggingface.co/Subin94/B-RAD-Det/resolve/main/brad_det_base_final_fold4.pth) |

Place them under:

```text
models/checkpoints/base/
```

## Download example

```bash
mkdir -p models/checkpoints/fewshot models/checkpoints/base

curl -L --fail --continue-at - --progress-bar \
  -o models/checkpoints/fewshot/brad_det_fewshot_1shot_best_fold0.pth \
  https://huggingface.co/Subin94/B-RAD-Det/resolve/main/brad_det_fewshot_1shot_best_fold0.pth

sha256sum models/checkpoints/fewshot/brad_det_fewshot_1shot_best_fold0.pth
```

Repeat for the remaining folds and compare every digest with the tables above.

## Backbone weights

The selected training scripts initialize ViT-B/16 through the official `timm` pretrained-weight mechanism. The unused local `vit_model.pt` artifact is not released.

## Files not released

Fold-level prediction CSV files and intermediate experimental checkpoints are private verification artifacts and are not included.
