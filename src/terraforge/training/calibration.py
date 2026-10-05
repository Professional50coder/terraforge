"""Calibration: do predicted confidences mean what they say?

A model that says "90% sure" should be right ~90% of the time. Deep nets are usually
over-confident. For a service that reports confidence to farmers or analysts this is a
product-correctness issue, not a nicety.

- ECE (expected calibration error): bin predictions by confidence, average
  |accuracy - confidence| weighted by bin size.
- Temperature scaling (Guo et al. 2017): divide logits by one scalar T fitted on the
  VALIDATION set. It cannot change the argmax, so accuracy is untouched; only the
  confidences are repaired.
"""
from __future__ import annotations

import numpy as np
import torch


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Minimise NLL over log T (keeps T positive). Fit on validation logits only."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())
