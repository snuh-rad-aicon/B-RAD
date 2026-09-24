import math
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from typing import Dict, Tuple, List

def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns image transforms.

    Output shapes:
      - train:  [3, 224, 224]  (C, H, W)
      - val:    [3, 224, 224]
    """
    tfm_train = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, fill=0),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    return tfm_train, tfm_val


# ----------------------------- Dataset -----------------------------

class MultiModal(Dataset):
    """
    For each image example, pick ONE text example with the SAME birads label on-the-fly.

    Returns per-sample (shapes per item, not batched yet):
      - image_tensor:      torch.float32, [3, 224, 224]
      - token_dict:
          * input_ids:     torch.long,    [L_text] (fixed 128 with padding/truncation)
          * attention_mask:torch.long,    [L_text] (0 for pad, 1 for real token)
      - image_label:       int            (e.g., 2/3/4/5)
      - text_label:        int            (same set; typically equals image_label)

    Note:
      * At training time, the text is chosen randomly from the same-class bucket (diversity).
      * At validation time, the text is chosen deterministically for reproducibility.
    """

    def __init__(
        self,
        image_df: pd.DataFrame,
        text_df: pd.DataFrame,
        images_root: Path,
        tokenizer,
        is_train: bool = True,
        seed: int = 0,
        max_text_len: int = 128,
    ):
        self.image_df = image_df.reset_index(drop=True)
        self.text_df = text_df.reset_index(drop=True)
        self.images_root = images_root
        self.tokenizer = tokenizer
        self.transform_train, self.transform_val = build_transforms()
        self.is_train = is_train
        self.max_text_len = max_text_len

        # Labels as numpy arrays for fast indexing
        self.image_labels: np.ndarray = self.image_df["birads"].astype(int).to_numpy()
        text_labels: np.ndarray = self.text_df["birads"].astype(int).to_numpy()

        # Build class -> indices bucket for texts
        self.text_indices_by_class: Dict[int, np.ndarray] = {
            int(lbl): np.where(text_labels == lbl)[0]
            for lbl in np.unique(text_labels)
        }

        # RNG for randomized text picking (training)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        # Length is based on the number of images.
        # The sampler upstream will handle rebalancing at epoch level.
        return len(self.image_df)

    def _pick_text_index(self, birads_label: int, image_index: int) -> int:
        """
        Pick one text index that shares 'birads_label'.
        - Train: random pick from bucket (diversity).
        - Val:   deterministic pick using image_index modulo bucket size.
        Fallback (rare): if bucket is empty, pick a random text globally.
        """
        pool: np.ndarray = self.text_indices_by_class.get(int(birads_label), None)

        if (pool is None) or (len(pool) == 0):
            return int(self.rng.integers(0, len(self.text_df)))  # fallback

        if self.is_train:
            j = int(self.rng.integers(0, len(pool)))
            return int(pool[j])
        else:
            return int(pool[image_index % len(pool)])

    def __getitem__(self, index: int):
        # -------- Image side --------
        image_row = self.image_df.iloc[index]
        image_label = int(self.image_labels[index])

        image_path = self.images_root / image_row.image
        image = Image.open(image_path).convert("RGB")

        image_tensor = (self.transform_train if self.is_train else self.transform_val)(image)
        # image_tensor: torch.float32, [3, 224, 224]

        # -------- Text side (same class as image_label) --------
        text_index = self._pick_text_index(image_label, index)
        text_row = self.text_df.iloc[text_index]

        tokenized = self.tokenizer(
            text_row.natural_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,  # L_text
            return_tensors="pt",
        )
        # tokenized["input_ids"]:     [1, L_text]
        # tokenized["attention_mask"]: [1, L_text]
        token_dict = {k: v.squeeze(0) for k, v in tokenized.items()}
        # After squeeze: [L_text]

        text_label = int(text_row.birads)

        return image_tensor, token_dict, image_label, text_label


# ----------------------------- Collate -----------------------------

def collate(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor], int, int]]):
    """
    Convert a list of N=BatchSize samples into a batched tensor.

    Input (list length = B):
      Each item = (image_tensor, token_dict, image_label, text_label)
        - image_tensor:  [3, 224, 224]
        - token_dict:
            input_ids:      [L_text]
            attention_mask: [L_text]
        - image_label:    []
        - text_label:     []

    Output (batched):
      - images:        torch.float32, [B, 3, 224, 224]
      - tokens:        dict with
          * input_ids:      torch.long, [B, L_text]
          * attention_mask: torch.long, [B, L_text]  (can be cast to bool in the model if desired)
      - image_labels:  torch.long,    [B]
      - text_labels:   torch.long,    [B]
    """
    images, tokens_list, image_labels, text_labels = zip(*batch)  # length B tuples

    images = torch.stack(images, dim=0)  # [B, 3, 224, 224]

    input_ids = torch.stack([t["input_ids"] for t in tokens_list], dim=0).long() # [B, L_text]
    attention_mask = torch.stack([t["attention_mask"] for t in tokens_list], dim=0).long() # [B, L_text]

    image_labels = torch.tensor(image_labels, dtype=torch.long)  # [B]
    text_labels = torch.tensor(text_labels, dtype=torch.long)    # [B]

    tokens = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    return images, tokens, image_labels, text_labels
