"""Masked Autoencoder (MAE) pretraining for multispectral ViTs.

Idea (He et al. 2021): hide a large random fraction of patches, encode only the
visible ones, and reconstruct the hidden pixels. To do that well the encoder must
learn spectral and spatial structure - without a single label. That matters in
Earth observation, where unlabeled imagery is abundant and labels are scarce.

Design choices and their reasons:
- Mask ratio 0.75: neighbouring pixels are highly redundant, so a low ratio makes
  the task a trivial interpolation. High ratios force semantic features.
- The encoder sees ONLY visible tokens (~4x cheaper), and a light decoder handles
  the mask tokens. The decoder is discarded after pretraining.
- Targets are per-patch normalised: stops the loss being dominated by bright bands.
- The encoder IS a `ViT`, so pretrained weights drop straight into a classifier.
"""
from __future__ import annotations

import torch
from torch import nn

from terraforge.models.vit import TransformerBlock, ViT


def patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    """(B,C,H,W) -> (B,N,patch*patch*C)."""
    b, c, h, w = x.shape
    gh, gw = h // patch, w // patch
    x = x.reshape(b, c, gh, patch, gw, patch)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(b, gh * gw, patch * patch * c)


class MaskedAutoencoder(nn.Module):
    def __init__(self, encoder: ViT, decoder_dim: int = 64, decoder_depth: int = 2,
                 decoder_heads: int = 4, norm_target: bool = True):
        super().__init__()
        self.encoder = encoder
        self.patch = encoder.embed.proj.kernel_size[0]
        in_ch = encoder.embed.proj.in_channels
        dim = encoder.pos.shape[-1]
        n = encoder.embed.n_patches
        self.norm_target = norm_target
        self.dec_embed = nn.Linear(dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, n + 1, decoder_dim))
        nn.init.trunc_normal_(self.dec_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.dec_blocks = nn.ModuleList(
            TransformerBlock(decoder_dim, decoder_heads) for _ in range(decoder_depth)
        )
        self.dec_norm = nn.LayerNorm(decoder_dim)
        self.dec_pred = nn.Linear(decoder_dim, self.patch * self.patch * in_ch)

    def encode(self, x: torch.Tensor, mask_ratio: float):
        enc = self.encoder
        tokens = enc.embed(x) + enc.pos[:, 1:]
        b, n, d = tokens.shape
        keep = max(1, int(n * (1 - mask_ratio)))
        noise = torch.rand(b, n, device=x.device)
        ids_shuffle = noise.argsort(dim=1)  # random permutation per sample
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :keep]
        visible = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
        mask = torch.ones(b, n, device=x.device)  # 1 = hidden
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        cls = (enc.cls + enc.pos[:, :1]).expand(b, -1, -1)
        t = torch.cat([cls, visible], dim=1)
        for blk in enc.blocks:
            t = blk(t)
        return enc.norm(t), mask, ids_restore

    def decode(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        t = self.dec_embed(latent)
        b, n_vis_plus_cls, d = t.shape
        n = ids_restore.shape[1]
        masks = self.mask_token.expand(b, n + 1 - n_vis_plus_cls, -1)
        body = torch.cat([t[:, 1:], masks], dim=1)
        body = torch.gather(body, 1, ids_restore.unsqueeze(-1).expand(-1, -1, d))
        t = torch.cat([t[:, :1], body], dim=1) + self.dec_pos
        for blk in self.dec_blocks:
            t = blk(t)
        return self.dec_pred(self.dec_norm(t))[:, 1:]  # drop CLS

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.75):
        latent, mask, ids_restore = self.encode(x, mask_ratio)
        pred = self.decode(latent, ids_restore)
        target = patchify(x, self.patch)
        if self.norm_target:
            mu, var = target.mean(-1, keepdim=True), target.var(-1, keepdim=True)
            target = (target - mu) / (var + 1e-6).sqrt()
        per_patch = ((pred - target) ** 2).mean(-1)
        loss = (per_patch * mask).sum() / mask.sum()  # loss on hidden patches only
        return loss, pred, mask
