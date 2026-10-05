"""CNN baseline.

A deliberately small 4-stage conv net. It is the control in the CNN-vs-ViT
experiment: convolutions bake in locality and translation equivariance, so on
small patches with little data a CNN is a strong baseline that a transformer
must justify beating.
"""
import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 13, n_classes: int = 10, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_channels, width),
            _block(width, width * 2),
            _block(width * 2, width * 4),
            _block(width * 4, width * 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(width * 4, n_classes),
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Latent embedding: globally pooled conv features, shape (B, width*4)."""
        return self.features(x).mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
