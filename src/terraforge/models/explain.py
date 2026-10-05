"""Attention rollout (Abnar & Zuidema 2020) for the from-scratch ViT.

Raw attention from one layer is misleading because residual connections let
information bypass attention. Rollout models this: per layer, A' = 0.5*A + 0.5*I
(head-averaged attention plus the identity for the residual), re-normalise rows,
then multiply layers together. The CLS row gives how much each input patch
contributed to the prediction.

Caveat: this is a visualisation of information flow, not a
causal attribution. It can look plausible without being faithful; validate with an
occlusion test (see `occlusion_drop`).
"""
from __future__ import annotations

import torch

from terraforge.models.vit import ViT


@torch.no_grad()
def attention_rollout(model: ViT, x: torch.Tensor) -> torch.Tensor:
    """Returns (B, grid, grid) importance maps summing to 1 per image."""
    model.eval()
    model(x)
    n = model.embed.n_patches + 1
    eye = torch.eye(n, device=x.device)
    roll = eye.expand(x.shape[0], n, n).clone()
    for blk in model.blocks:
        a = blk.attn.last_attention.mean(1)  # average heads -> (B,N,N)
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(-1, keepdim=True)
        roll = a @ roll
    cls_to_patches = roll[:, 0, 1:]
    cls_to_patches = cls_to_patches / cls_to_patches.sum(-1, keepdim=True)
    g = int(cls_to_patches.shape[1] ** 0.5)
    return cls_to_patches.reshape(-1, g, g)


@torch.no_grad()
def occlusion_drop(model: torch.nn.Module, x: torch.Tensor, patch: int, grid_scores: torch.Tensor):
    """Faithfulness check: zero the top-attended patch and measure the drop in the
    predicted-class probability. A faithful map should cause a larger drop than
    occluding a random patch. Returns (drop_top, drop_random) averaged over the batch."""
    model.eval()
    p0 = model(x).softmax(-1)
    cls = p0.argmax(-1)
    base = p0.gather(1, cls[:, None]).squeeze(1)
    b, g, _ = grid_scores.shape
    top = grid_scores.flatten(1).argmax(1)
    rnd = torch.randint(0, g * g, (b,))

    def occlude(idx):
        xo = x.clone()
        for i in range(b):
            r, c = divmod(int(idx[i]), g)
            xo[i, :, r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] = 0
        return model(xo).softmax(-1).gather(1, cls[:, None]).squeeze(1)

    return float((base - occlude(top)).mean()), float((base - occlude(rnd)).mean())
