import numpy as np

from terraforge.training import selective as s


def test_perfect_confidence_ordering_gives_zero_aurc_and_full_selective_accuracy():
    conf = np.array([0.99, 0.95, 0.9, 0.6, 0.55])
    correct = np.array([1, 1, 1, 0, 0])
    assert s.aurc(conf, correct) < 0.2                    # errors all at the tail
    assert s.coverage_at_accuracy(conf, correct, 1.0) == 0.6   # answer 3 of 5, all correct
    assert s.error_auroc(1 - conf, correct) == 1.0


def test_uninformative_confidence_gives_chance_auroc_and_base_error_aurc():
    rng = np.random.default_rng(0)
    conf = rng.random(20000)
    correct = rng.random(20000) < 0.8                      # independent of confidence
    assert abs(s.error_auroc(1 - conf, correct) - 0.5) < 0.02
    assert abs(s.aurc(conf, correct) - 0.2) < 0.02


def test_inverted_confidence_is_worse_than_chance():
    conf = np.array([0.5, 0.6, 0.9, 0.95])
    correct = np.array([1, 1, 0, 0])                      # confident ones are the wrong ones
    assert s.error_auroc(1 - conf, correct) == 0.0
    assert s.coverage_at_accuracy(conf, correct, 0.9) == 0.0


def test_auroc_ties_count_half_and_undefined_without_both_outcomes():
    assert s.error_auroc(np.array([1.0, 1.0]), np.array([0, 1])) == 0.5
    assert np.isnan(s.error_auroc(np.array([1.0, 2.0]), np.array([1, 1])))


def test_risk_coverage_endpoints():
    cov, risk = s.risk_coverage(np.array([0.9, 0.8, 0.7]), np.array([1, 0, 1]))
    assert cov[-1] == 1.0 and abs(risk[-1] - 1 / 3) < 1e-9 and risk[0] == 0.0


def test_knn_novelty_is_higher_for_shifted_inputs_and_chunking_is_exact():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(500, 16)) + 5.0               # one cluster
    same = rng.normal(size=(100, 16)) + 5.0
    other = rng.normal(size=(100, 16)) - 5.0               # opposite direction
    n_same, n_other = s.knn_novelty(train, same), s.knn_novelty(train, other)
    assert n_other.mean() > n_same.mean() + 0.5
    assert np.allclose(s.knn_novelty(train, same, chunk=7), n_same, atol=1e-6)


def test_exact_duplicate_has_similarity_one():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(50, 8))
    assert abs(s.nearest_train_similarity(train, train[3:4])[0] - 1.0) < 1e-5
