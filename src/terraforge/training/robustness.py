"""Robustness benchmark for multispectral models.

Accuracy on a clean test set says little about operational behaviour. Real EO inputs
arrive with failed/missing bands, haze, thin cloud and sensor noise. This module
corrupts a batch in physically motivated ways, at several severities, and reports how
accuracy degrades. A model that collapses when one band drops is a deployment risk
even with a high clean score.

Corruptions operate on *standardised* tensors (B,C,H,W), as the models see them.
"""
from __future__ import annotations

import torch


def band_dropout(x: torch.Tensor, severity: float, gen: torch.Generator) -> torch.Tensor:
    """Zero a random fraction of bands per sample (sensor/band failure)."""
    b, c = x.shape[:2]
    keep = (torch.rand(b, c, 1, 1, generator=gen) >= severity).to(x.dtype)
    return x * keep


def gaussian_noise(x: torch.Tensor, severity: float, gen: torch.Generator) -> torch.Tensor:
    """Additive noise with std = severity (in standardised units)."""
    return x + severity * torch.randn(x.shape, generator=gen)


def haze(x: torch.Tensor, severity: float, gen: torch.Generator) -> torch.Tensor:
    """Atmospheric haze: raise all bands toward a bright offset, stronger in the
    visible/blue end (bands 0-3), which scatter most. severity in [0,1]."""
    out = x.clone()
    weights = torch.linspace(1.0, 0.2, x.shape[1]).view(1, -1, 1, 1)
    return out + severity * 2.0 * weights


def cloud_patches(x: torch.Tensor, severity: float, gen: torch.Generator) -> torch.Tensor:
    """Cover a `severity` fraction of the image area with bright, spectrally flat blocks
    (opaque cloud). Blocks are 8x8 pixels, placed at random."""
    out = x.clone()
    b, _, h, w = x.shape
    gh, gw = h // 8, w // 8
    n_blocks = int(round(severity * gh * gw))
    for i in range(b):
        idx = torch.randperm(gh * gw, generator=gen)[:n_blocks]
        for j in idx.tolist():
            r, c = divmod(j, gw)
            out[i, :, r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = 3.0
    return out


CORRUPTIONS = {
    "band_dropout": band_dropout,
    "gaussian_noise": gaussian_noise,
    "haze": haze,
    "cloud_patches": cloud_patches,
}


@torch.no_grad()
def robustness_report(model, loader, severities=(0.0, 0.1, 0.25, 0.5), seed: int = 0,
                      device: str = "cpu") -> dict:
    """accuracy[corruption][severity] over the loader. Severity 0 = clean reference."""
    model.eval().to(device)
    report: dict[str, dict[float, float]] = {}
    for name, fn in CORRUPTIONS.items():
        report[name] = {}
        for s in severities:
            gen = torch.Generator().manual_seed(seed)  # same corruption draw per model
            correct = total = 0
            for x, y in loader:
                xc = fn(x, s, gen) if s > 0 else x
                pred = model(xc.to(device)).argmax(1).cpu()
                correct += int((pred == y).sum())
                total += len(y)
            report[name][s] = correct / total
    return report
