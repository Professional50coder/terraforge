"""Classification metrics in plain numpy (no sklearn dependency).

Why macro-F1 next to accuracy: accuracy hides failure on rare or confusable
classes (e.g. Pasture vs HerbaceousVegetation). Macro-F1 weights every class
equally, so one badly handled class visibly drags the score down.
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def precision_recall_f1(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    pred = cm.sum(axis=0).astype(float)
    true = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def summarize(y_true, y_pred, n_classes: int) -> dict:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    p, r, f1 = precision_recall_f1(cm)
    return {
        "accuracy": float(np.trace(cm) / cm.sum()),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": cm.tolist(),
    }
