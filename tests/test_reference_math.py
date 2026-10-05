"""Cross-checks of hand-written numerics against independent reference implementations.

Hand-rolled maths is where silent errors hide. Each test here compares our implementation to
an established library (scikit-learn, SciPy, NumPy, PyTorch) on random inputs, so a formula or
indexing mistake cannot pass just because it agrees with itself.
"""
import numpy as np
import pytest

sk = pytest.importorskip("sklearn.metrics")
scipy_stats = pytest.importorskip("scipy.stats")
torch = pytest.importorskip("torch")

from terraforge.training import conformal as cf  # noqa: E402
from terraforge.training import drift, selective  # noqa: E402
from terraforge.training.evaluation import confusion_matrix, precision_recall_f1, summarize  # noqa: E402

RNG = np.random.default_rng(1234)


def test_confusion_matrix_and_prf_match_sklearn():
    for _ in range(20):
        n_cls = int(RNG.integers(2, 10))
        y = RNG.integers(0, n_cls, 500)
        p = np.where(RNG.random(500) < 0.7, y, RNG.integers(0, n_cls, 500))
        np.testing.assert_array_equal(confusion_matrix(y, p, n_cls),
                                      sk.confusion_matrix(y, p, labels=range(n_cls)))
        prec, rec, f1 = precision_recall_f1(confusion_matrix(y, p, n_cls))
        sp, sr, sf, _ = sk.precision_recall_fscore_support(
            y, p, labels=range(n_cls), zero_division=0)
        np.testing.assert_allclose(prec, sp, atol=1e-12)
        np.testing.assert_allclose(rec, sr, atol=1e-12)
        np.testing.assert_allclose(f1, sf, atol=1e-12)
        s = summarize(y, p, n_cls)
        assert abs(s["macro_f1"] - sk.f1_score(y, p, labels=range(n_cls), average="macro",
                                                zero_division=0)) < 1e-12
        assert abs(s["accuracy"] - sk.accuracy_score(y, p)) < 1e-12


def test_error_auroc_matches_sklearn_including_ties():
    for _ in range(30):
        n = 400
        correct = RNG.random(n) < 0.8
        score = np.round(RNG.random(n), 1)            # coarse values force many ties
        ours = selective.error_auroc(score, correct)
        ref = sk.roc_auc_score(~correct, score)       # positive class = "wrong"
        assert abs(ours - ref) < 1e-12


def test_average_ranks_match_scipy_rankdata():
    x = np.round(RNG.normal(size=300), 1)
    np.testing.assert_allclose(selective._average_ranks(x), scipy_stats.rankdata(x, method="average"))


def test_conformal_quantile_is_kth_smallest_checked_two_independent_ways():
    # Definition (Vovk; Angelopoulos & Bates): q = the k-th smallest calibration score with
    # k = ceil((n+1)(1-alpha)); +inf when k > n.
    # NumPy's "higher" method indexes at p*(n-1) and is NOT this quantity; the matching NumPy
    # definition is "inverted_cdf" (smallest x with empirical CDF(x) >= p), here p = k/n.
    for n in (20, 99, 100, 500):
        for alpha in (0.05, 0.1, 0.2):
            s = RNG.random(n)
            k = int(np.ceil((n + 1) * (1 - alpha)))
            ours = cf.conformal_quantile(s, alpha)
            if k > n:
                assert ours == float("inf")
                continue
            assert ours == np.quantile(s, k / n, method="inverted_cdf")          # reference 1
            assert (s <= ours).sum() >= k and (s < ours).sum() < k              # counting: exactly the k-th
            assert ours == np.sort(s)[k - 1]                                     # reference 2


def test_conformal_coverage_is_exact_in_the_continuous_case():
    # For exchangeable continuous scores, P(score_new <= q) = k/(n+1) exactly; check by averaging
    # over many independent draws (the theoretical value, not just ">= target").
    n, alpha, trials = 50, 0.1, 40000
    k = int(np.ceil((n + 1) * (1 - alpha)))
    cal = RNG.random((trials, n))
    q = np.sort(cal, axis=1)[:, k - 1]
    new = RNG.random(trials)
    assert abs((new <= q).mean() - k / (n + 1)) < 0.006


def test_aps_scores_match_sorted_cumulative_mass():
    p = RNG.dirichlet(np.ones(7), size=200)           # continuous -> no ties
    ours = cf.aps_scores(p)
    order = np.argsort(-p, axis=1)
    cum = np.cumsum(np.take_along_axis(p, order, axis=1), axis=1)
    ref = np.empty_like(p)
    np.put_along_axis(ref, order, cum, axis=1)
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_knn_novelty_matches_bruteforce_cosine():
    t, q = RNG.normal(size=(300, 12)), RNG.normal(size=(40, 12))
    tn = t / np.linalg.norm(t, axis=1, keepdims=True)
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    d = 1 - qn @ tn.T
    ref = np.sort(d, axis=1)[:, :5].mean(1)
    np.testing.assert_allclose(selective.knn_novelty(t, q, k=5, chunk=11), ref, atol=1e-5)


def test_risk_coverage_matches_manual_prefix_error():
    conf, correct = RNG.random(200), RNG.random(200) < 0.7
    cov, risk = selective.risk_coverage(conf, correct)
    order = np.argsort(-conf, kind="stable")
    for k in (1, 17, 100, 200):
        assert abs(risk[k - 1] - (1 - correct[order][:k].mean())) < 1e-12
        assert abs(cov[k - 1] - k / 200) < 1e-12


def test_psi_matches_direct_formula():
    ref = RNG.normal(size=(5000, 1))
    live = RNG.normal(loc=0.3, size=(3000, 1))
    edges, probs = drift.reference_histograms(ref, bins=10)
    ours = drift.psi(live, edges, probs)[0]
    p = np.clip(np.histogram(live[:, 0], edges[0])[0] / 3000, 1e-6, None)
    q = np.clip(probs[0], 1e-6, None)
    assert abs(ours - float(np.sum((p - q) * np.log(p / q)))) < 1e-9
    assert ours > 0.02                                # a 0.3-sigma shift is detectable


def test_ece_matches_independent_binning():
    from terraforge.training.calibration import expected_calibration_error as ece
    probs = RNG.dirichlet(np.ones(5) * 0.3, size=2000)
    y = RNG.integers(0, 5, 2000)
    conf, hit = probs.max(1), probs.argmax(1) == y
    edges, total = np.linspace(0, 1, 16), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.sum() / len(y) * abs(hit[m].mean() - conf[m].mean())
    assert abs(ece(probs, y) - total) < 1e-12


def test_temperature_scaling_matches_scipy_optimiser():
    from scipy.optimize import minimize_scalar

    from terraforge.training.calibration import fit_temperature
    logits = torch.randn(800, 4) * 3
    y = torch.randint(0, 4, (800,))
    logits[torch.arange(800), y] += 2.0
    ours = fit_temperature(logits, y)

    def nll(logT):
        return float(torch.nn.functional.cross_entropy(logits / np.exp(logT), y))
    ref = np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x)
    assert abs(ours - ref) / ref < 0.02


def test_multihead_attention_matches_torch_sdpa_and_manual_softmax():
    from terraforge.models.vit import MultiHeadSelfAttention
    torch.manual_seed(0)
    m = MultiHeadSelfAttention(32, 4).eval()
    x = torch.randn(3, 9, 32)
    with torch.no_grad():
        ours = m(x)
        b, n, d = x.shape
        qkv = m.qkv(x).reshape(b, n, 3, 4, d // 4).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)       # PyTorch's own kernel
        ref = m.out(ref.transpose(1, 2).reshape(b, n, d))
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)


def test_vit_patch_embedding_equals_explicit_patch_linear():
    from terraforge.models.vit import PatchEmbed
    torch.manual_seed(0)
    pe = PatchEmbed(16, 4, 3, 8)
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        ours = pe(x)
        w = pe.proj.weight.reshape(8, -1)                                      # (D, C*p*p)
        patches = x.unfold(2, 4, 4).unfold(3, 4, 4).permute(0, 2, 3, 1, 4, 5).reshape(2, 16, -1)
        # conv weight is laid out (C, ph, pw); reorder the unfolded patch the same way
        patches = x.unfold(2, 4, 4).unfold(3, 4, 4).permute(0, 2, 3, 1, 4, 5)  # (B,gh,gw,C,ph,pw)
        patches = patches.reshape(2, 16, 3 * 4 * 4)
        ref = patches @ w.T + pe.proj.bias
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)


def test_mae_patchify_is_invertible():
    from terraforge.models.mae import patchify
    x = torch.randn(2, 5, 16, 16)
    p = patchify(x, 4)                                   # (B, 16, 4*4*5)
    back = p.reshape(2, 4, 4, 4, 4, 5).permute(0, 5, 1, 3, 2, 4).reshape(2, 5, 16, 16)
    torch.testing.assert_close(back, x)


def test_warmup_cosine_endpoints():
    from terraforge.training.schedule import warmup_cosine
    assert warmup_cosine(0, 100, 10) == pytest.approx(0.1)           # (0+1)/10
    assert warmup_cosine(9, 100, 10) == pytest.approx(1.0)
    assert warmup_cosine(10, 100, 10) == pytest.approx(1.0)          # cosine starts at 1
    assert warmup_cosine(100, 100, 10) == pytest.approx(0.01)        # floor
    assert warmup_cosine(55, 100, 10) == pytest.approx(0.01 + 0.99 * 0.5)   # midpoint of cosine
