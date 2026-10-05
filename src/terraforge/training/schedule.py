"""Learning-rate schedule and weight EMA.

Warmup + cosine decay: transformers are unstable at full LR from step 0 (attention
logits are poorly scaled with random weights), so LR ramps up linearly, then decays
smoothly to a floor. Stepped per iteration, not per epoch, for smoothness.

EMA of weights: an exponential moving average of parameters is a cheap ensemble of
the optimisation trajectory and usually generalises a little better than the last
raw iterate; it is evaluated instead of the raw weights.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn


def warmup_cosine(step: int, total: int, warmup: int, floor: float = 0.01) -> float:
    """LR multiplier in [floor, 1]."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:  # e.g. BatchNorm num_batches_tracked
                s.copy_(m)
