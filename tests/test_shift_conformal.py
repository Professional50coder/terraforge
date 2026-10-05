import numpy as np
import pytest

from terraforge.training import conformal as cf
from terraforge.training import selective
from terraforge.training import shift_conformal as sc

K, D, ALPHA = 6, 12, 0.1


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


class World:
    """Controlled shift simulator with KNOWN severity s in [0, 1].

    Higher s weakens the class signal in the logits (more errors) and inflates embedding noise
    (so kNN novelty rises). It is a mechanism check for the code, NOT evidence about satellite data.
    """

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.centers = self.rng.normal(size=(K, D)) * 2.0
        self.train_emb = self.sample(3000, 0.0)[1]

    def sample(self, n, s):
        r = self.rng
        y = r.integers(0, K, n)
        emb = self.centers[y] + r.normal(size=(n, D)) * (0.7 + 2.0 * s)
        signal = 3.0 * (1 - 0.85 * s)
        logits = r.normal(size=(n, K)) * 1.0
        logits[np.arange(n), y] += signal
        return y, emb, softmax(logits)

    def batch(self, n, s, labelled=True):
        y, emb, p = self.sample(n, s)
        nov = selective.knn_novelty(self.train_emb, emb, k=5)
        return sc.Batch(p, emb, nov, y if labelled else None, np.full(n, s))

    def augmented(self, n_per, levels=(0.15, 0.3, 0.45, 0.6, 0.75)):
        return sc.Batch.concat([self.batch(n_per, s) for s in levels])


def test_weighted_quantile_reduces_to_standard_split_conformal_with_equal_weights():
    rng = np.random.default_rng(0)
    for n in (40, 100, 333):
        s = rng.random(n)
        q = sc.weighted_quantiles(s, np.ones(n), np.array([1.0]), 0.1)[0]
        assert q == cf.conformal_quantile(s, 0.1)


def test_weighted_quantile_downweights_irrelevant_scores_and_gives_inf_when_uncertifiable():
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    heavy_low = sc.weighted_quantiles(s, np.array([10, 10, 10, .01, .01]), np.array([1.0]), 0.1)[0]
    assert heavy_low <= 0.9                       # the huge-score tail barely matters
    assert sc.weighted_quantiles(s, np.ones(5), np.array([1000.0]), 0.1)[0] == np.inf


def test_severity_estimator_recovers_a_linear_signal_and_clips():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 5))
    sev = np.clip(0.5 + 0.2 * x[:, 0] - 0.1 * x[:, 1], 0, 1)
    est = sc.SeverityEstimator().fit(x, sev, lam=0.1)
    pred = est.predict(x)
    assert np.corrcoef(pred, sev)[0, 1] > 0.98 and pred.min() >= 0 and pred.max() <= 1


def test_domain_ratio_is_near_one_without_shift_and_large_for_shifted_inputs():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(1500, 3)), rng.normal(size=(1500, 3))
    r = sc.fit_domain_ratio(a, b)
    assert 0.7 < np.median(r(a)) < 1.4
    shifted = rng.normal(size=(1500, 3)) + np.array([2.0, 0, 0])
    r2 = sc.fit_domain_ratio(a, shifted)
    assert np.median(r2(shifted)) > 3 * np.median(r2(a))


def test_clean_condition_every_method_keeps_coverage_and_set_sizes_stay_sane():
    w = World(0)
    clean, aug = w.batch(3000, 0.0), w.augmented(800)
    res = sc.compare(clean, aug, {"clean": w.batch(4000, 0.0)}, ALPHA)["clean"]
    for m, r in res.items():
        assert r["coverage"] >= 0.86, (m, r)
    assert res["sacp"]["mean_size"] <= res["scp"]["mean_size"] * 1.6


def test_marginal_conformal_fails_under_shift_but_conditional_methods_recover_coverage():
    w = World(1)
    clean, aug = w.batch(3000, 0.0), w.augmented(800)
    res = sc.compare(clean, aug, {"s=0.6": w.batch(4000, 0.6, labelled=True)}, ALPHA)["s=0.6"]
    assert res["scp"]["coverage"] < 0.83, res["scp"]            # the problem: guarantee voided
    assert res["sacp"]["coverage"] > res["scp"]["coverage"] + 0.05, (res["sacp"], res["scp"])
    assert res["sacp"]["undercoverage"] < res["scp"]["undercoverage"]


def test_sets_grow_with_severity_for_sacp_but_not_for_marginal_scp():
    w = World(2)
    clean, aug = w.batch(3000, 0.0), w.augmented(800)
    conds = {f"s={s}": w.batch(3000, s) for s in (0.0, 0.3, 0.6)}
    res = sc.compare(clean, aug, conds, ALPHA)
    assert res["s=0.6"]["sacp"]["mean_size"] > res["s=0.0"]["sacp"]["mean_size"]   # adapts
    assert abs(res["s=0.6"]["scp"]["mean_size"] - res["s=0.0"]["scp"]["mean_size"]) < \
        res["s=0.6"]["sacp"]["mean_size"] - res["s=0.0"]["sacp"]["mean_size"] + 1.0


def test_coverage_deficit_bound_holds_total_variation_proposition():
    # Q(S<=q) >= P(S<=q) - TV(P,Q): verify numerically on two discrete score distributions.
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(20)); q = rng.dirichlet(np.ones(20))
    tv = 0.5 * np.abs(p - q).sum()
    for k in range(20):
        assert q[: k + 1].sum() >= p[: k + 1].sum() - tv - 1e-12


# ---------------------------------------------------------------- new standards & hybrids ----------

def test_envelope_pvalues_are_uniform_in_distribution_and_flag_far_shift():
    rng = np.random.default_rng(0)
    cal = rng.normal(size=5000)
    p, out = sc.envelope_flags(cal, rng.normal(size=20000), gamma=0.05)
    assert abs(out.mean() - 0.05) < 0.01                        # chance flags ~ gamma
    assert abs(np.mean(p <= 0.5) - 0.5) < 0.02                  # p-values ~ uniform
    _, far = sc.envelope_flags(cal, rng.normal(loc=6.0, size=1000), gamma=0.05)
    assert far.mean() > 0.99


def test_envelope_flags_inputs_beyond_the_simulated_severity_range():
    w = World(3)
    clean, aug = w.batch(3000, 0.0), w.augmented(800)           # calibrated up to s=0.75
    res = sc.compare(clean, aug, {"in": w.batch(3000, 0.3), "beyond": w.batch(3000, 1.0)}, ALPHA)
    assert res["in"]["sacp"]["outside_envelope_fraction"] < 0.08
    assert res["beyond"]["sacp"]["outside_envelope_fraction"] > res["in"]["sacp"]["outside_envelope_fraction"] + 0.15


def test_paired_bootstrap_ci_contains_truth_and_excludes_zero_for_a_real_difference():
    rng = np.random.default_rng(0)
    n, C = 4000, 5
    y = rng.integers(0, C, n)
    a = np.zeros((n, C), bool); b = np.zeros((n, C), bool)
    a[np.arange(n), y] = rng.random(n) < 0.95      # A covers 95%
    b[np.arange(n), y] = rng.random(n) < 0.80      # B covers 80%
    r = sc.paired_bootstrap(a, b, y, n_boot=500)
    lo, hi = r["coverage_ci"]
    assert lo > 0.0 and lo < r["coverage_diff"] < hi and abs(r["coverage_diff"] - 0.15) < 0.03
    same = sc.paired_bootstrap(a, a, y, n_boot=200)
    assert same["coverage_ci"] == [0.0, 0.0]


def test_size_at_coverage_interpolates_and_handles_unreachable_targets():
    pts = [(0.5, 1.0), (0.8, 2.0), (0.95, 4.0)]
    assert sc.size_at_coverage(pts, 0.9) == pytest.approx(2.0 + (0.1 / 0.15) * 2.0)
    assert sc.size_at_coverage(pts, 0.99) == float("inf")
    assert sc.size_at_coverage(pts, 0.3) == 1.0


def test_effective_severity_puts_different_nominal_scales_on_one_axis():
    w = World(4)
    clean = w.batch(2000, 0.0)
    # same physical inflation produced by two "families" with wildly different nominal scales
    a = w.batch(1500, 0.3); a.severity = np.full(1500, 0.05)    # nominal scale 1
    b = w.batch(1500, 0.3); b.severity = np.full(1500, 40.0)    # nominal scale 2
    c = w.batch(1500, 0.7); c.severity = np.full(1500, 80.0)
    aug = sc.Batch.concat([a, b, c])
    eff = sc.effective_severity(clean, aug)
    ea, eb, ec = eff[:1500].mean(), eff[1500:3000].mean(), eff[3000:].mean()
    assert abs(ea - eb) < 0.1 and ec > ea + 0.2 and ec == pytest.approx(1.0, abs=1e-6)


def test_min_bin_size_keeps_quantiles_finite_with_small_calibration():
    w = World(5)
    clean, aug = w.batch(150, 0.0), w.augmented(40)
    res = sc.compare(clean, aug, {"t": w.batch(1500, 0.3)}, ALPHA)["t"]
    assert res["sacp"]["mean_size"] < 5.0                       # not collapsed to the full class set


def test_hybrid_union_is_a_superset_of_its_components_so_covers_at_least_as_often():
    w = World(6)
    clean, aug, test = w.batch(2500, 0.0), w.augmented(600), w.batch(3000, 0.5)
    m = sc.predict_sets_all(clean, aug, test, ALPHA)
    assert (m["hybrid_union"] | m["sacp"] == m["hybrid_union"]).all()
    assert (m["hybrid_union"] | m["cond_novelty"] == m["hybrid_union"]).all()
    r = {k: cf.evaluate_sets(v, test.y)["coverage"] for k, v in m.items()}
    assert r["hybrid_union"] >= max(r["sacp"], r["cond_novelty"], r["cond_entropy"]) - 1e-12


def test_raps_score_penalises_depth_and_yields_smaller_sets_than_aps():
    p = np.array([[0.4, 0.3, 0.15, 0.1, 0.05]])
    aps, raps = cf.aps_scores(p), cf.raps_scores(p, lam=0.1, k_reg=1)
    assert np.allclose(raps - aps, 0.1 * np.maximum(0, np.arange(1, 6) - 1))   # rank penalty only
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(8) * 0.4, size=3000)
    y = np.array([rng.choice(8, p=r) for r in probs])
    sizes = {}
    for method in ("aps", "raps"):
        q = cf.conformal_quantile(cf.true_class_scores(cf.SCORERS[method](probs[:1500]), y[:1500]), 0.1)
        sizes[method] = cf.prediction_sets(probs[1500:], q, method).sum(1).mean()
    assert sizes["raps"] < sizes["aps"]


def test_linear_quantile_regression_recovers_conditional_quantile():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 20000)
    y = 1.0 + 2.0 * x + (0.5 + 1.0 * (x + 1)) * rng.normal(size=20000)   # noise grows with x
    w = sc.fit_linear_quantile(x[:, None], y, 0.9)
    pred = np.column_stack([x, np.ones_like(x)]) @ w
    for lo, hi in ((-1, -0.3), (-0.3, 0.3), (0.3, 1)):
        m = (x >= lo) & (x < hi)
        assert abs((y[m] <= pred[m]).mean() - 0.9) < 0.06            # near tau within each x-slab
