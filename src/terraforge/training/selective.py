"""Selective prediction: can the system tell when it is probably wrong?

A classifier that is 95% accurate overall can still be 60% accurate on the cases it is
unsure about. What matters operationally is whether its *confidence ranks its own errors
last*, so that abstaining on the least-confident fraction removes most of the mistakes.

- risk-coverage curve: sort by confidence (high first); at coverage c keep the top c of
  predictions and report the error rate among them.
- AURC: area under that curve; lower is better (0 = confidence perfectly orders errors).
- coverage@accuracy: the largest fraction of inputs answerable while meeting an accuracy bar.
- error AUROC: probability that a randomly chosen wrong prediction gets a lower confidence
  than a randomly chosen correct one (0.5 = confidence carries no information).
- kNN novelty: mean cosine distance from an embedding to its k nearest *training* embeddings.
  An input far from everything seen in training is a candidate for abstention even when the
  softmax is confident (softmax is known to be over-confident on out-of-distribution inputs).
"""
from __future__ import annotations

import numpy as np


def risk_coverage(confidence: np.ndarray, correct: np.ndarray):
    """Returns (coverage, risk) arrays, one entry per kept-prefix size, confidence descending."""
    order = np.argsort(-np.asarray(confidence), kind="stable")
    err = 1.0 - np.asarray(correct, dtype=float)[order]
    n = np.arange(1, len(err) + 1)
    return n / len(err), np.cumsum(err) / n


def aurc(confidence, correct) -> float:
    _, risk = risk_coverage(confidence, correct)
    return float(risk.mean())


def coverage_at_accuracy(confidence, correct, target: float = 0.95) -> float:
    """Largest coverage whose kept predictions reach `target` accuracy (0 if never)."""
    cov, risk = risk_coverage(confidence, correct)
    ok = cov[(1 - risk) >= target]
    return float(ok.max()) if len(ok) else 0.0


def error_auroc(score_higher_is_error: np.ndarray, correct: np.ndarray) -> float:
    """AUROC of a score at ranking wrong predictions above correct ones (ties count half)."""
    s = np.asarray(score_higher_is_error, dtype=float)
    wrong = ~np.asarray(correct, dtype=bool)
    if wrong.all() or not wrong.any():
        return float("nan")  # undefined without both outcomes
    ranks = _average_ranks(s)
    n_pos, n_neg = wrong.sum(), (~wrong).sum()
    return float((ranks[wrong].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):  # average over ties
        m = x == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def knn_novelty(train_emb: np.ndarray, query_emb: np.ndarray, k: int = 5,
                chunk: int = 512) -> np.ndarray:
    """Mean cosine distance to the k nearest training embeddings (higher = more novel)."""
    def unit(a):
        return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    t, q = unit(train_emb.astype("float32")), unit(query_emb.astype("float32"))
    k = min(k, len(t))
    out = np.empty(len(q), dtype="float32")
    for i in range(0, len(q), chunk):  # chunked: never materialise the full N x M matrix
        sims = q[i:i + chunk] @ t.T
        top = -np.partition(-sims, k - 1, axis=1)[:, :k]
        out[i:i + chunk] = 1.0 - top.mean(axis=1)
    return out


def nearest_train_similarity(train_emb: np.ndarray, query_emb: np.ndarray) -> np.ndarray:
    """Cosine similarity to the single nearest training embedding (near-duplicate probe)."""
    return 1.0 - knn_novelty(train_emb, query_emb, k=1)
