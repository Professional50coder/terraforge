"""Split conformal prediction: sets with a finite-sample coverage guarantee.

Setting: a held-out calibration set (here: the VALIDATION split, disjoint from train and test).
For each calibration example compute a nonconformity score s(x, y) - larger means the label
fits the model's output worse. With n calibration scores and miscoverage level alpha, take
q = the ceil((n+1)(1-alpha))-th smallest score. For a new exchangeable example,

    P( y in { c : s(x, c) <= q } ) >= 1 - alpha                       (marginal coverage)

with no assumption on the model or the data distribution. Two scores are provided:

- LAC  : s = 1 - p_y.           Smallest average sets; guarantee is marginal only.
- APS  : s = total probability mass of all classes at least as likely as y (deterministic,
         i.e. non-randomised, so slightly conservative). Adapts set size to difficulty.

Class-conditional (Mondrian) calibration gives the guarantee per true class, which matters
when a rare or confusable class (e.g. Pasture vs HerbaceousVegetation) would otherwise be
under-covered while the average looks fine.

The guarantee requires exchangeability. Under distribution shift (haze, band loss, a different
processing level) it can fail - quantifying that failure is the point of the evaluation code,
not something to hide.
"""
from __future__ import annotations

import numpy as np


def lac_scores(probs: np.ndarray) -> np.ndarray:
    """(N, C) nonconformity of every candidate class: 1 - p_c."""
    return 1.0 - probs


def aps_scores(probs: np.ndarray) -> np.ndarray:
    """(N, C): mass of all classes with probability >= p_c (includes c itself)."""
    ge = probs[:, None, :] >= probs[:, :, None]          # [n, c, j]: p_j >= p_c
    return (ge * probs[:, None, :]).sum(-1)


SCORERS = {"lac": lac_scores, "aps": aps_scores}


def true_class_scores(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    return scores[np.arange(len(y)), y]


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest score; +inf if n is too small (set = everything)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    n = len(cal_scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(cal_scores)[k - 1]) if 1 <= k <= n else float("inf")


def class_conditional_quantiles(cal_scores: np.ndarray, y: np.ndarray, alpha: float,
                                n_classes: int) -> np.ndarray:
    """One quantile per true class; +inf where a class has too few calibration samples."""
    return np.array([conformal_quantile(cal_scores[y == c], alpha) if (y == c).any()
                     else float("inf") for c in range(n_classes)])


def prediction_sets(probs: np.ndarray, q, method: str = "lac") -> np.ndarray:
    """(N, C) boolean membership. `q` is a scalar or a per-class array (Mondrian).

    The arg-max class is always included, so sets are never empty; adding a class can only
    raise coverage, so the guarantee is preserved.
    """
    mask = SCORERS[method](probs) <= np.asarray(q)
    mask[np.arange(len(probs)), probs.argmax(1)] = True
    return mask


def evaluate_sets(mask: np.ndarray, y: np.ndarray, n_classes: int | None = None) -> dict:
    n_classes = n_classes or mask.shape[1]
    covered = mask[np.arange(len(y)), y]
    sizes = mask.sum(1)
    per_class = {int(c): float(covered[y == c].mean()) for c in range(n_classes) if (y == c).any()}
    return {"coverage": float(covered.mean()), "mean_size": float(sizes.mean()),
            "singleton_fraction": float((sizes == 1).mean()),
            "worst_class_coverage": min(per_class.values()), "per_class_coverage": per_class}


def named_sets(mask_row: np.ndarray, class_names, probs_row: np.ndarray) -> list[str]:
    """Human-readable set, most likely class first."""
    idx = np.flatnonzero(mask_row)
    return [class_names[i] for i in idx[np.argsort(-probs_row[idx])]]
