"""B-RAD-Det model composition."""

import torch
import torch.nn as nn

from brad_det.ops.anchors import pool_anchor_region_features_roi


class OwlVitModel(nn.Module):
    """Compose a ViT backbone, learned class queries, and detection head."""

    def __init__(self, backbone, head, num_classes=4, input_size=224, patch_size=16):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.head = head
        self.query_embeddings = nn.Embedding(num_classes, 768)
        self.input_size = input_size
        self.patch_size = patch_size

    def forward(self, x, anchors, labels=None):
        if labels is not None:
            query_labels = torch.arange(
                self.num_classes,
                device=x.device,
            ).unsqueeze(0).repeat(x.shape[0], 1)
            queries = self.query_embeddings(query_labels)
        else:
            queries = self.query_embeddings.weight.unsqueeze(0).expand(
                x.size(0),
                -1,
                -1,
            )

        _, tokens = self.backbone(x)
        grid_size = self.input_size // self.patch_size
        features = tokens.reshape(tokens.shape[0], grid_size, grid_size, -1)
        pooled = pool_anchor_region_features_roi(
            features,
            anchors,
            img_size=self.input_size,
            patch_size=self.patch_size,
        )
        return self.head(pooled, queries)

