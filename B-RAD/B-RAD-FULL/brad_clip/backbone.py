"""VPT-BiomedCLIP backbone shared by B-RAD-CLIP (retrieval) and B-RAD-FULL.

The fine-tuned checkpoint (`img_enc`) is a complete state dict for the whole
model, so the encoder is built from `config.json` and the checkpoint alone; the
original BiomedCLIP `pytorch_model.bin` is not required at inference time.
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .model import BiomedCLIPConfig, BiomedCLIPModel

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
MODEL_DIR = Path(__file__).resolve().parent / "model"
TEXT_ENCODER = "michiyasunaga/BioLinkBERT-large"


def image_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize(size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def projection_head(in_dim: int, out_dim: int = 512, hidden: int = None) -> nn.Module:
    """2-layer MLP projection head g(.) used for contrastive alignment."""
    hidden = hidden or out_dim
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


def _load_projection(state: dict, in_dim: int, out_dim: int = 512) -> nn.Module:
    """Rebuild the 2-layer MLP head g(.) with the checkpoint's hidden width."""
    head = projection_head(in_dim, out_dim, hidden=state["0.weight"].shape[0])
    head.load_state_dict(state)
    return head


class VPTVisionWrapper(nn.Module):
    """Prepend learnable visual-prompt tokens to a frozen BiomedCLIP ViT."""

    def __init__(self, vision_model, n_prompt: int = 8):
        super().__init__()
        self.vision = vision_model
        self.prompt = nn.Parameter(torch.randn(n_prompt, vision_model.config.hidden_size) * 0.02)
        for p in self.vision.parameters():
            p.requires_grad = False

    def forward(self, x, attention_bias=None):
        b = x.size(0)
        x = self.vision.embeddings(x)
        cls, patch = x[:, :1], x[:, 1:]
        p = self.prompt.unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([cls, p, patch], 1)
        if attention_bias is not None:
            for layer in self.vision.encoder.layers:
                layer.self_attn._foveal_bias = attention_bias
        x = self.vision.encoder(x)[0]
        if attention_bias is not None:
            for layer in self.vision.encoder.layers:
                if hasattr(layer.self_attn, "_foveal_bias"):
                    delattr(layer.self_attn, "_foveal_bias")
        return self.vision.post_layernorm(x)[:, 0]


class BRADClip(nn.Module):
    """Load a fine-tuned VPT-BiomedCLIP checkpoint and expose 512-d embeddings."""

    def __init__(self, ckpt_path, n_prompt: int = 8, device: str = "cuda", load_text: bool = False):
        super().__init__()
        self.device = device
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)

        clip = BiomedCLIPModel(BiomedCLIPConfig.from_pretrained(str(MODEL_DIR / "config.json")))
        clip.vpt = VPTVisionWrapper(clip.vision_model, n_prompt)
        clip.load_state_dict(ck["img_enc"])
        self.clip = clip.to(device).eval()

        self.img_proj = _load_projection(
            ck["proj_img"], clip.vision_model.config.hidden_size).to(device).eval()

        self.mlp_heads = ck.get("mlp_heads")  # 5-fold classification ensemble, when bundled
        self.grid_side = int(round(math.sqrt(clip.vision_model.embeddings.num_patches)))
        self.n_prompt = n_prompt
        self.tokenizer = self.text_back = self.txt_proj = None
        if load_text:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
            text_back = AutoModel.from_pretrained(TEXT_ENCODER)
            text_back.load_state_dict(ck["txt_back"])
            self.text_back = text_back.to(device).eval()
            self.txt_proj = _load_projection(
                ck["proj_txt"], text_back.config.hidden_size).to(device).eval()

        for p in self.parameters():
            p.requires_grad = False

    def load_image(self, path):
        return image_transform()(Image.open(path).convert("RGB")).unsqueeze(0)

    @torch.no_grad()
    def embed_image(self, img_tensor, attention_bias=None, normalize=True):
        with torch.autocast(self.device.split(":")[0], enabled=self.device.startswith("cuda")):
            f = self.img_proj(self.clip.vpt(img_tensor.to(self.device), attention_bias=attention_bias))
        f = f.float()
        return F.normalize(f, dim=1) if normalize else f

    @torch.no_grad()
    def embed_text(self, text, normalize=True):
        tok = self.tokenizer(text, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        tok = {k: v.to(self.device) for k, v in tok.items()}
        with torch.autocast(self.device.split(":")[0], enabled=self.device.startswith("cuda")):
            f = self.txt_proj(self.text_back(**tok).last_hidden_state[:, 0, :])
        f = f.float()
        return F.normalize(f, dim=1) if normalize else f

    @torch.no_grad()
    def attention_map(self, img_tensor, attention_bias=None, rollout=False, start_layer=0):
        """CLS->patch attention over the patch grid, as a [B, grid, grid] map.

        Runs the same VPT (and optional ROI foveal-bias) path used for the
        embedding, so the map shows where the encoder actually looked.
        Defaults to last-layer attention; `rollout=True` accumulates all layers.
        """
        vision = self.clip.vision_model
        x = vision.embeddings(img_tensor.to(self.device))
        cls, patch = x[:, :1], x[:, 1:]
        prompt = self.clip.vpt.prompt.unsqueeze(0).expand(x.size(0), -1, -1)
        x = torch.cat([cls, prompt, patch], 1)
        if attention_bias is not None:
            for layer in vision.encoder.layers:
                layer.self_attn._foveal_bias = attention_bias
        attentions = vision.encoder(x, output_attentions=True, return_dict=True).attentions
        if attention_bias is not None:
            for layer in vision.encoder.layers:
                if hasattr(layer.self_attn, "_foveal_bias"):
                    delattr(layer.self_attn, "_foveal_bias")

        if rollout:
            n = attentions[0].size(-1)
            eye = torch.eye(n, device=attentions[0].device).unsqueeze(0)
            acc = eye.repeat(attentions[0].size(0), 1, 1)
            for attn in attentions[start_layer:]:
                a = attn.mean(1).float() + eye
                a = a / a.sum(-1, keepdim=True)
                acc = torch.bmm(a, acc)
            cls_row = acc[:, 0]
        else:
            cls_row = attentions[-1].mean(1).float()[:, 0]   # last-layer CLS attention

        cls_to_patch = cls_row[:, 1 + self.n_prompt:]      # drop CLS and VPT prompt tokens
        g = self.grid_side
        return cls_to_patch.reshape(-1, g, g).cpu().numpy()
