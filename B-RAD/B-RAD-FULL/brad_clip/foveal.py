"""ROI foveal attention bias used by B-RAD-FULL.

Turns the lesion mask into the additive attention bias of paper Eq. 5, applied
at inference time to the frozen vision encoder.
"""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def foveal_bias_from_mask(mask, grid_side, *, gamma=1.18, n_prefix_tokens=9, device=None):
    """Foveal attention bias from the lesion mask (paper Eq. 5).

    The mask is down-sampled to the ViT token grid, giving each token its lesion
    coverage M_j. The bias is `gamma * M_j`, which the attention softmax sees as
    the paper's `0` on lesion tokens and `-gamma` on background tokens (a softmax
    row is invariant to a constant shift). Tokens that straddle the lesion border
    take the intermediate value their coverage implies.

    Returns a `[1, 1, N, N]` bias on the CLS query row, with
    N = n_prefix_tokens + grid_side**2 and n_prefix_tokens = 1 CLS + VPT prompts.
    """
    m = torch.as_tensor(np.asarray(mask), dtype=torch.float32, device=device)
    m = (m > 0).float()[None, None]
    
    cov = F.adaptive_avg_pool2d(m, grid_side)[0, 0]            # M_j in [0, 1]
    cov = (cov - cov.min()) / (cov.max() - cov.min() + 1e-8)
    n = n_prefix_tokens + grid_side * grid_side
    
    row = torch.zeros(n, device=cov.device)
    row[n_prefix_tokens:] = gamma * cov.reshape(-1)

    bias = torch.zeros(1, 1, n, n, device=cov.device)
    bias[:, :, 0, :] = row
    
    return bias
