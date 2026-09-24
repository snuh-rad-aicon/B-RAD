# Supervised contrastive + cross-modal InfoNCE objectives for B-RAD-CLIP.
import torch
import torch.nn as nn
import torch.nn.functional as F

def _big_neg_like(x: torch.Tensor) -> torch.Tensor:
    """Return a big negative scalar tensor with the same dtype/device as x, representable in that dtype."""
    if x.dtype in (torch.float16, torch.bfloat16):
        val = -1e4  # representable in fp16/bf16; exp(val) ~ 0
    else:
        val = -1e9  # safe for fp32/fp64
    return torch.tensor(val, dtype=x.dtype, device=x.device)

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)

class SupCon(nn.Module):
    """
    Stable Supervised Contrastive Loss
      - compute in fp32
      - no +/-inf (mask with big finite negative)
      - ignore anchors with zero positives
    """
    def __init__(self, t: float = 0.07):
        super().__init__()
        self.t = float(t)

    def forward(self, z: torch.Tensor, y: torch.Tensor, t=None) -> torch.Tensor:
        # fp32 for stability; `t` may be a learnable tensor
        tau = self.t if t is None else t
        z = l2_normalize(z.float(), dim=-1)                     # [B, d]
        logits = (z @ z.t()) / tau                              # [B, B], fp32
        B = logits.size(0)

        # self mask with finite big negative
        eye = torch.eye(B, dtype=torch.bool, device=logits.device)
        big_neg = _big_neg_like(logits)
        logits = logits.masked_fill(eye, big_neg)

        # row-wise log-softmax (stable)
        lse = torch.logsumexp(logits, dim=1, keepdim=True)      # [B, 1]
        log_prob = logits - lse                                  # [B, B]

        # positives: same label, excluding self
        pos = y.view(-1, 1).eq(y.view(1, -1)) & (~eye)          # [B, B]
        pos_cnt = pos.sum(1)                                     # [B]
        valid = pos_cnt > 0

        pos_log = (log_prob * pos.float()).sum(1)               # [B]
        if valid.any():
            loss = -(pos_log[valid] / pos_cnt[valid].float()).mean()
        else:
            loss = logits.new_tensor(0.0)
        return loss

def _xmod_infonce(fi: torch.Tensor, ft: torch.Tensor, li: torch.Tensor, lt: torch.Tensor, tau) -> torch.Tensor:
    """Label-aware cross-modal loss (paper Eq. 2), symmetric and label-masked.

    For each anchor, sigma is the softmax over the whole opposite modality in the
    batch, and the loss averages log(sigma) over that anchor's positives P(i) --
    the same aggregation SupCon uses, applied across modalities.
    """
    fi = l2_normalize(fi.float(), dim=-1)                       # [B, d]
    ft = l2_normalize(ft.float(), dim=-1)                       # [B, d]
    # clamp tiny numerical overshoot of cosine
    sim = (fi @ ft.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6)          # [B, B]
    logits = sim / tau                                          # [B, B]
    pos = li.view(-1, 1).eq(lt.view(1, -1))                     # [B, B]

    def _direction(lg: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        log_prob = lg - torch.logsumexp(lg, dim=1, keepdim=True)
        cnt = p.sum(1)
        valid = cnt > 0                                         # anchors with no positive
        if not valid.any():
            return lg.new_tensor(0.0)
        pos_log = (log_prob * p.float()).sum(1)
        return -(pos_log[valid] / cnt[valid].float()).mean()

    return 0.5 * (_direction(logits, pos) + _direction(logits.t(), pos.t()))


def birads_sim(
    fi: torch.Tensor,
    ft: torch.Tensor,
    li: torch.Tensor,
    lt: torch.Tensor,
    temp: float = 0.07,
    t=None,            # float or learnable tensor
) -> torch.Tensor:
    tau = temp if t is None else t
    return _xmod_infonce(fi, ft, li, lt, tau)


class OrdinalHead(nn.Module):
    """Projects a pooled embedding to the K-1 CORN boundary logits (psi)."""

    def __init__(self, in_dim: int, num_classes: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The CORN boundaries stay in fp32: their BCE is unstable under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.fc(x.float())                               # [B, K-1]


@torch.no_grad()
def corn_decode(logits: torch.Tensor) -> torch.Tensor:
    """Rank prediction: how many boundaries the sample is predicted to exceed."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def corn_loss(logits: torch.Tensor, y_ord: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """Ordinal loss over the K-1 BI-RADS boundaries (paper Eq. 3).

    The summation is split at the ground-truth rank: boundaries below it are
    pushed toward one and boundaries at or above it toward zero. `y_ord` is the
    zero-based rank, i.e. BI-RADS {2,3,4,5} -> {0,1,2,3}.
    """
    assert logits.dim() == 2 and logits.size(1) == num_classes - 1
    logits = logits.float()
    r = torch.arange(num_classes - 1, device=logits.device)          # [K-1]
    target = (r[None, :] < y_ord.long()[:, None]).float()            # [B, K-1]
    return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
