import numpy as np
import pytest

from terraforge.training import conformal as cf

C = 6


def calibrated_world(rng, n):
    """Labels drawn from the model's own probabilities: a perfectly calibrated model."""
    logits = rng.normal(size=(n, C)) * 2
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(C, p=row) for row in p])
    return p, y


@pytest.mark.parametrize("method", ["lac", "aps"])
def test_marginal_coverage_guarantee_holds_in_expectation(method):
    alpha, rng, covs = 0.1, np.random.default_rng(0), []
    for _ in range(150):
        pc, yc = calibrated_world(rng, 200)
        q = cf.conformal_quantile(cf.true_class_scores(cf.SCORERS[method](pc), yc), alpha)
        pt, yt = calibrated_world(rng, 400)
        covs.append(cf.evaluate_sets(cf.prediction_sets(pt, q, method), yt)["coverage"])
    # finite-sample theory: E[coverage] in [1-alpha, 1-alpha + 1/(n+1)] (LAC); APS is conservative
    assert np.mean(covs) >= 0.9 - 0.005 and np.mean(covs) <= 0.97


def test_guarantee_breaks_under_label_shift_and_we_can_measure_it():
    rng = np.random.default_rng(1)
    pc, yc = calibrated_world(rng, 2000)
    q = cf.conformal_quantile(cf.true_class_scores(cf.lac_scores(pc), yc), 0.1)
    pt, _ = calibrated_world(rng, 4000)
    y_shift = rng.integers(0, C, len(pt))          # labels no longer follow the model: shift
    cov = cf.evaluate_sets(cf.prediction_sets(pt, q), y_shift)["coverage"]
    assert cov < 0.8                                 # far below the 0.9 target: guarantee void


def test_sets_adapt_to_uncertainty():
    sure = np.array([[0.98, 0.005, 0.004, 0.005, 0.0025, 0.0035]])
    unsure = np.array([[0.25, 0.22, 0.20, 0.15, 0.10, 0.08]])
    for method in ("lac", "aps"):
        assert cf.prediction_sets(sure, 0.85, method).sum() == 1
        assert cf.prediction_sets(unsure, 0.85, method).sum() == 4   # the four most likely


def test_aps_with_exact_ties_is_conservative_not_wrong():
    # uniform output: every class "ties" with all others, so each scores 1.0. A threshold below
    # 1.0 excludes everything but the forced arg-max; q = 1.0 (what calibration would pick for
    # genuinely uninformative outputs) admits all classes.
    p = np.full((1, C), 1 / C)
    assert cf.prediction_sets(p, 0.99, "aps").sum() == 1
    assert cf.prediction_sets(p, 1.0, "aps").sum() == C


def test_argmax_always_in_set_so_sets_never_empty():
    p = np.array([[0.4, 0.3, 0.3, 0.0, 0.0, 0.0]])
    assert cf.prediction_sets(p, 0.0, "lac").sum() >= 1
    assert cf.prediction_sets(p, 0.0, "lac")[0, 0]


def test_quantile_definition_and_tiny_calibration_set():
    s = np.arange(1.0, 11.0)                        # n=10, alpha=0.2 -> k=ceil(11*0.8)=9
    assert cf.conformal_quantile(s, 0.2) == 9.0
    assert cf.conformal_quantile(np.array([0.1, 0.2]), 0.01) == float("inf")  # cannot certify
    with pytest.raises(ValueError):
        cf.conformal_quantile(s, 1.5)


def test_aps_score_is_cumulative_mass_up_to_and_including_class():
    p = np.array([[0.5, 0.3, 0.2]])
    assert np.allclose(cf.aps_scores(p), [[0.5, 0.8, 1.0]])
    tie = np.array([[0.4, 0.4, 0.2]])
    assert np.allclose(cf.aps_scores(tie)[0, :2], 0.8)   # ties share the same score


def test_class_conditional_covers_each_class_not_just_on_average():
    rng = np.random.default_rng(2)
    # class 0 is easy and common; class 1 is hard and rare: marginal calibration under-covers it
    def world(n):
        y = (rng.random(n) < 0.1).astype(int)
        p1 = np.where(y == 1, rng.beta(2, 3, n), rng.beta(1, 8, n))
        p = np.stack([1 - p1, p1], axis=1)
        return p, y
    pc, yc = world(20000)
    pt, yt = world(20000)
    sc = cf.lac_scores(pc)
    marg = cf.evaluate_sets(cf.prediction_sets(pt, cf.conformal_quantile(cf.true_class_scores(sc, yc), 0.1)), yt, 2)
    mond_q = cf.class_conditional_quantiles(cf.true_class_scores(sc, yc), yc, 0.1, 2)
    mond = cf.evaluate_sets(cf.prediction_sets(pt, mond_q), yt, 2)
    assert mond["worst_class_coverage"] >= 0.88
    assert mond["worst_class_coverage"] > marg["worst_class_coverage"]


def test_named_sets_ordering():
    names = ["a", "b", "c"]
    assert cf.named_sets(np.array([True, True, False]), names, np.array([0.3, 0.6, 0.1])) == ["b", "a"]
