"""Vision Transformer backbone used by B-RAD-Det."""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTBackbone(nn.Module):
    """Expose the class token and patch tokens from a timm Vision Transformer."""

    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        vit = timm.create_model(model_name, pretrained=pretrained)

        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.out_dim = vit.embed_dim

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:, :]


def interpolate_pos_encoding(model, new_h, new_w):
    """Interpolate patch position embeddings to a new input resolution."""
    patch_size_h, patch_size_w = model.backbone.patch_embed.patch_size
    num_patches = (new_h // patch_size_h) * (new_w // patch_size_w)

    pos_embed = model.backbone.pos_embed
    cls_pos = pos_embed[:, 0:1, :]
    patch_pos = pos_embed[:, 1:, :]
    dim = patch_pos.shape[-1]
    num_pos = patch_pos.shape[1]
    old_grid_size = int(num_pos**0.5)

    if num_patches == num_pos:
        return

    patch_pos = patch_pos.reshape(1, old_grid_size, old_grid_size, dim)
    patch_pos = patch_pos.permute(0, 3, 1, 2)
    new_grid_size = (new_h // patch_size_h, new_w // patch_size_w)
    patch_pos = F.interpolate(
        patch_pos,
        size=new_grid_size,
        mode="bicubic",
        align_corners=False,
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)

    new_pos_embed = torch.cat((cls_pos, patch_pos), dim=1)
    model.backbone.pos_embed = nn.Parameter(new_pos_embed)
    model.backbone.patch_embed.img_size = (new_h, new_w)

