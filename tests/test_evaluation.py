import numpy as np

from terraforge.training.evaluation import confusion_matrix, summarize


def test_confusion_matrix_orientation():
    cm = confusion_matrix([0, 0, 1], [0, 1, 1], 2)
    assert cm.tolist() == [[1, 1], [0, 1]]  # rows true, cols predicted


def test_perfect_prediction():
    s = summarize([0, 1, 2, 2], [0, 1, 2, 2], 3)
    assert s["accuracy"] == 1.0 and s["macro_f1"] == 1.0


def test_unpredicted_class_scores_zero_not_nan():
    s = summarize([0, 1], [0, 0], 2)
    assert np.isfinite(s["macro_f1"]) and s["per_class_f1"][1] == 0.0
    assert abs(s["macro_f1"] - (2 / 3) / 2) < 1e-9
