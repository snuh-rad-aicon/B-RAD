"""Detection heads used by B-RAD-Det."""

import torch
import torch.nn as nn


class PredictorMLP(nn.Module):
    def __init__(self, input_dim=768, out_dim=1, dropout=0.0):
        super().__init__()
        self.dense0 = nn.Linear(input_dim, input_dim)
        self.norm0 = nn.LayerNorm(input_dim)
        self.dense1 = nn.Linear(input_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(input_dim, out_dim)
        nn.init.zeros_(self.dense2.weight)

    def forward(self, x):
        x = self.gelu(self.norm0(self.dense0(x)))
        x = self.dropout(x)
        x = self.gelu(self.norm1(self.dense1(x)))
        x = self.dropout(x)
        return self.dense2(x)


class ClassPredictor(nn.Module):
    def __init__(self, image_dim=768, out_dim=768, n_cls=2, dropout=0.0):
        super().__init__()
        self.dense0 = nn.Linear(image_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.logit_shift = nn.Linear(image_dim, n_cls)
        self.logit_scale = nn.Linear(image_dim, n_cls)
        self.elu = nn.ELU()
        self.n_cls = n_cls

    def forward(self, image_embeds, query_embeds, query_mask=None):
        image_proj = self.dense0(image_embeds)
        image_proj = self.norm(image_proj)
        image_proj = self.dropout(image_proj)
        image_proj = image_proj / (image_proj.norm(dim=-1, keepdim=True) + 1e-6)

        query_embeds = query_embeds / (
            query_embeds.norm(dim=-1, keepdim=True) + 1e-6
        )
        logits = torch.einsum("bnd,bqd->bnq", image_proj, query_embeds)

        shift = self.logit_shift(image_embeds)
        scale = self.elu(self.logit_scale(image_embeds)) + 1.0
        logits = (logits + shift) * scale

        return {
            "pred_logits": logits,
            "class_embeddings": image_proj,
        }


class OwlVitDetectionHead(nn.Module):
    def __init__(self, num_classes=3, num_anchors=None, dropout=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.objectness = PredictorMLP(out_dim=1, dropout=dropout)
        self.box = PredictorMLP(out_dim=4, dropout=dropout)
        self.classifier = ClassPredictor(
            out_dim=768,
            n_cls=self.num_classes,
            dropout=dropout,
        )

    def forward(self, x, queries):
        batch_size, num_regions, _ = x.shape
        num_classes = self.num_classes

        obj = self.objectness(x).reshape(batch_size, num_regions, 1)
        box = self.box(x).reshape(batch_size, num_regions, 4)
        cls = self.classifier(x, queries)
        pred_logit = cls["pred_logits"].reshape(
            batch_size,
            num_regions,
            num_classes,
        )

        return {
            "objectness_logits": obj,
            "pred_boxes": box,
            "pred_logits": pred_logit,
            "class_embeddings": cls["class_embeddings"],
        }

