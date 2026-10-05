"""Input drift monitoring with the Population Stability Index (PSI).

In production the model never sees the training distribution again: seasons change,
sensors age, processing baselines shift (Sentinel-2 L2A processing baseline changes
altered reflectance offsets in 2022). Silent drift degrades accuracy without any error.

PSI compares a live sample against a reference histogram per band:
    PSI = sum_bins (p_live - p_ref) * ln(p_live / p_ref)
Common rule of thumb: <0.1 stable, 0.1-0.25 investigate, >0.25 significant shift.
Reference bin edges come from the TRAINING data so the comparison is anchored.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def reference_histograms(samples: np.ndarray, bins: int = 20):
    """samples: (N, C) pixel values per band. Returns edges (C, bins+1) and
    reference probabilities (C, bins), using quantile edges (equal-mass bins)."""
    edges, probs = [], []
    for c in range(samples.shape[1]):
        e = np.unique(np.quantile(samples[:, c], np.linspace(0, 1, bins + 1)))
        e[0], e[-1] = -np.inf, np.inf
        h = np.histogram(samples[:, c], e)[0].astype(float)
        edges.append(e)
        probs.append(h / h.sum())
    return edges, probs


def psi(live: np.ndarray, edges, ref_probs) -> np.ndarray:
    """Per-band PSI of live (M, C) against the reference. Returns (C,)."""
    out = np.zeros(live.shape[1])
    for c in range(live.shape[1]):
        h = np.histogram(live[:, c], edges[c])[0].astype(float)
        p = np.clip(h / max(h.sum(), 1), EPS, None)
        q = np.clip(ref_probs[c][: len(p)], EPS, None)
        out[c] = float(((p - q) * np.log(p / q)).sum())
    return out


def status(value: float) -> str:
    return "stable" if value < 0.1 else "investigate" if value < 0.25 else "drifted"
