"""Vision Transformer written from scratch (no timm/torchvision).

Pipeline: image -> patchify -> linear patch embedding -> prepend [CLS] token
-> add learned positional embeddings -> N x (self-attention + MLP) -> head on [CLS].

Every piece is explicit so each design choice can be explained:
- Patch embedding is a strided conv: a conv with kernel=stride=patch is exactly
  "cut into patches and apply one shared linear layer to each".
- Attention has no notion of position, so positional embeddings are required.
- Pre-norm blocks (LayerNorm before attention/MLP) train more stably than
  post-norm at small scale.
"""
import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch: int, in_ch: int, dim: int):
        super().__init__()
        if img_size % patch:
            raise ValueError("img_size must be divisible by patch")
        self.n_patches = (img_size // patch) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):  # (B,C,H,W) -> (B,N,D)
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.last_attention = None  # kept for interpretability / explanation

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B,H,N,Dh)
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        self.last_attention = attn.detach()
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(b, n, d)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class ViT(nn.Module):
    def __init__(self, img_size: int = 64, patch: int = 8, in_channels: int = 13,
                 n_classes: int = 10, dim: int = 128, depth: int = 6,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed = PatchEmbed(img_size, patch, in_channels, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.embed.n_patches + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, heads, dropout=dropout) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        t = self.embed(x)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1) + self.pos
        for blk in self.blocks:
            t = blk(t)
        return self.head(self.norm(t)[:, 0])
