"""Conformal calibration under structured shift: SACP and the baselines it is compared with.

Motivation. Split conformal prediction guarantees 1 - alpha coverage only if test data is
exchangeable with calibration data. Earth-observation shift is not arbitrary: haze, band loss and
noise are *physical corruptions with a severity*. If severity were known per input, calibrating the
conformal quantile as a function of severity would restore coverage (conditional calibration;
Gibbs, Cherian & Candes 2023, arXiv:2305.12616, give the general framework).

SACP (severity-aware conformal prediction) - the candidate method evaluated here:
  1. Corrupt the validation set with the physical corruption family at known severities.
  2. Fit a severity estimator s_hat(x) (ridge regression) from label-free input features
     (embedding, kNN novelty, softmax entropy, max-probability) to the applied severity.
  3. Bin the pooled (clean + corrupted) calibration set by s_hat and compute one conformal
     quantile per bin (Mondrian calibration with s_hat as the taxonomy).
  4. At test time, route each input to its s_hat bin and use that bin's quantile.

Proposition (coverage deficit). Let a test point fall in bin b, with calibration scores S ~ P_b and
test scores S' ~ Q_b. Calibration gives P_b(S <= q_b) >= 1 - alpha. Then
        Q_b(S' <= q_b) >= 1 - alpha - TV(P_b, Q_b),
because |P(A) - Q(A)| <= TV(P, Q) for every event A. Coverage is therefore restored exactly to the
extent that s_hat is *sufficient for the shift's effect on scores* ("severity sufficiency"): within a
bin, real-shift scores must look like simulated-shift scores. The guarantee fails when real shift
differs from the simulated family in a way s_hat cannot see; TV is the measurable quantity.

Baselines (all use the LAC score unless noted):
  scp          split conformal, clean calibration only (marginal guarantee)
  scp_aps      same with the APS score
  mondrian_cls class-conditional, clean calibration
  pooled_aug   one marginal quantile over clean + corrupted calibration
  cond_novelty Mondrian over bins of kNN embedding novelty (pooled calibration)
  cond_entropy Mondrian over bins of softmax entropy (pooled calibration)
  weighted     weighted conformal under covariate shift (Tibshirani et al. 2019) with a
               logistic domain-classifier density ratio; uses the unlabeled test batch
  sacp         the method above
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from terraforge.training import conformal as cf


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(1)


@dataclass
class Batch:
    """Everything the calibrators may use. `y` is None for unlabeled test batches."""
    probs: np.ndarray
    emb: np.ndarray
    novelty: np.ndarray
    y: np.ndarray | None = None
    severity: np.ndarray | None = None  # applied severity (calibration only; unknown at test)
    group: np.ndarray | None = None     # which corruption condition each row came from

    def features(self) -> np.ndarray:
        return np.column_stack([self.novelty, entropy(self.probs), self.probs.max(1), self.emb])

    @staticmethod
    def concat(batches: list["Batch"]) -> "Batch":
        sev = None if any(b.severity is None for b in batches) else np.concatenate([b.severity for b in batches])
        groups, offset = [], 0
        for b in batches:  # each source batch is one condition unless it already carries groups
            g = b.group if b.group is not None else np.zeros(len(b.probs), dtype=int)
            groups.append(g + offset)
            offset += int(g.max()) + 1 if len(g) else 0
        return Batch(np.concatenate([b.probs for b in batches]), np.concatenate([b.emb for b in batches]),
                     np.concatenate([b.novelty for b in batches]),
                     np.concatenate([b.y for b in batches]), sev, np.concatenate(groups))


class SeverityEstimator:
    """Ridge regression from label-free features to applied corruption severity in [0, 1]."""

    def fit(self, feats: np.ndarray, severity: np.ndarray, lam: float = 1.0) -> "SeverityEstimator":
        self.mu, self.sd = feats.mean(0), feats.std(0) + 1e-8
        z = np.column_stack([(feats - self.mu) / self.sd, np.ones(len(feats))])
        reg = lam * np.eye(z.shape[1]); reg[-1, -1] = 0.0  # do not penalise the intercept
        self.w = np.linalg.solve(z.T @ z + reg, z.T @ severity)
        return self

    def predict(self, feats: np.ndarray) -> np.ndarray:
        z = np.column_stack([(feats - self.mu) / self.sd, np.ones(len(feats))])
        return np.clip(z @ self.w, 0.0, 1.0)


def _bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    return np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])  # interior edges


def _binned_quantiles(cov: np.ndarray, scores: np.ndarray, edges: np.ndarray, alpha: float) -> np.ndarray:
    b = np.searchsorted(edges, cov, side="right")
    return np.array([cf.conformal_quantile(scores[b == i], alpha) if (b == i).any() else np.inf
                     for i in range(len(edges) + 1)])


def _lac_true(batch: Batch) -> np.ndarray:
    return cf.true_class_scores(cf.lac_scores(batch.probs), batch.y)


def _true(batch: Batch, method: str) -> np.ndarray:
    return cf.true_class_scores(cf.SCORERS[method](batch.probs), batch.y)


def fit_linear_quantile(z: np.ndarray, y: np.ndarray, tau: float, iters: int = 600, lr: float = 0.03):
    """Linear quantile regression by Adam on the pinball loss (features few, so this is ample).

    Returns weights w with prediction [z, 1] @ w. Initialised at the marginal tau-quantile so the
    optimiser only has to learn the *deviation* driven by the covariates.
    """
    x = np.column_stack([z, np.ones(len(z))])
    w = np.zeros(x.shape[1]); w[-1] = np.quantile(y, tau)
    m, v = np.zeros_like(w), np.zeros_like(w)
    for t in range(1, iters + 1):
        r = y - x @ w
        g = -(x * np.where(r > 0, tau, tau - 1)[:, None]).mean(0)   # subgradient of pinball loss
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return w


def _cqr_features(batch: Batch, s_hat: np.ndarray) -> np.ndarray:
    h, mx = entropy(batch.probs), batch.probs.max(1)
    return np.column_stack([s_hat, batch.novelty, h, mx, s_hat ** 2, s_hat * h])


def effective_severity(clean: Batch, aug: Batch) -> np.ndarray:
    """Per-row severity defined as measured score inflation, comparable across corruption families.

    Nominal severities are family-specific numbers (fraction of bands dropped vs noise std vs haze
    offset) and cannot be placed on one axis. What the conformal quantile actually needs is how much
    a condition *inflates the nonconformity score*, so: for each condition (group) take the mean
    true-class LAC score, subtract the clean mean, and normalise by the largest inflation seen.
    """
    base = _lac_true(clean).mean()
    sc_ = _lac_true(aug)
    infl = {g: max(0.0, sc_[aug.group == g].mean() - base) for g in np.unique(aug.group)}
    top = max(max(infl.values()), 1e-9)
    return np.array([infl[g] / top for g in aug.group])


def envelope_flags(cal_novelty: np.ndarray, test_novelty: np.ndarray, gamma: float = 0.02):
    """Conformal p-values for 'this input is inside the calibrated region' (novelty as the score).

    p = (1 + #{calibration novelty >= test novelty}) / (n + 1). Under exchangeability p is
    (super-)uniform, so about a gamma fraction of in-region inputs are flagged by chance. Inputs with
    p < gamma are *outside the calibrated envelope*: the coverage statement is not made for them.
    """
    cal = np.sort(cal_novelty)
    ge = len(cal) - np.searchsorted(cal, test_novelty, side="left")
    p = (1 + ge) / (len(cal) + 1)
    return p, p < gamma


def paired_bootstrap(mask_a: np.ndarray, mask_b: np.ndarray, y: np.ndarray, n_boot: int = 1000,
                     seed: int = 0) -> dict:
    """95% CI for (A - B) in coverage and in mean set size, resampling test points in pairs."""
    rng = np.random.default_rng(seed)
    n = len(y)
    cov_a = mask_a[np.arange(n), y].astype(float)   # bool - bool is undefined in numpy
    cov_b = mask_b[np.arange(n), y].astype(float)
    size_a, size_b = mask_a.sum(1), mask_b.sum(1)
    idx = rng.integers(0, n, (n_boot, n))
    d_cov = (cov_a[idx] - cov_b[idx]).mean(1)
    d_size = (size_a[idx] - size_b[idx]).mean(1)
    lo, hi = np.percentile(d_cov, [2.5, 97.5]), np.percentile(d_size, [2.5, 97.5])
    return {"coverage_diff": float((cov_a - cov_b).mean()), "coverage_ci": [float(lo[0]), float(lo[1])],
            "size_diff": float((size_a - size_b).mean()), "size_ci": [float(hi[0]), float(hi[1])]}


def size_at_coverage(points: list[tuple[float, float]], target: float) -> float:
    """Mean set size needed to reach `target` coverage, by linear interpolation on (coverage, size)
    points; +inf if the method never reaches it, its smallest size if it always exceeds it."""
    pts = sorted(points)
    cov = np.array([c for c, _ in pts]); size = np.array([z for _, z in pts])
    if cov.max() < target:
        return float("inf")
    if cov.min() >= target:
        return float(size[0])
    return float(np.interp(target, cov, size))


def efficiency_at_coverage(clean_cal: Batch, aug_cal: Batch, test: Batch, target: float = 0.9,
                           alphas=(0.02, 0.05, 0.1, 0.2, 0.3, 0.4), n_bins: int = 6) -> dict[str, float]:
    """Set size each method needs to reach `target` coverage on `test` (matched-coverage standard).

    Coverage at a *nominal* alpha rewards conservatism; a method that over-covers looks safe and
    large. Sweeping alpha traces each method's coverage-size frontier, and reading it at one common
    coverage compares the quality of the underlying ranking, not how cautious its calibration is.
    (It uses test labels to pick the operating point, so it is an analysis standard, never a deployed rule.)
    """
    curves: dict[str, list] = {}
    for a in alphas:
        for m, mask in predict_sets_all(clean_cal, aug_cal, test, a, n_bins).items():
            r = cf.evaluate_sets(mask, test.y)
            curves.setdefault(m, []).append((r["coverage"], r["mean_size"]))
    return {m: size_at_coverage(pts, target) for m, pts in curves.items()}


def weighted_quantiles(scores: np.ndarray, w_cal: np.ndarray, w_test: np.ndarray, alpha: float):
    """Per-test-point quantile of weighted conformal prediction (Tibshirani et al. 2019).

    q_t = smallest score s_(k) with sum_{i<=k} w_i >= (1 - alpha) * (sum_i w_i + w_t); the test
    point's own weight sits at +inf, so if the target exceeds the calibration mass q_t = +inf.
    """
    order = np.argsort(scores)
    s, cw = scores[order], np.cumsum(w_cal[order])
    target = (1 - alpha) * (cw[-1] + w_test)
    idx = np.searchsorted(cw, target, side="left")
    return np.where(idx < len(s), s[np.minimum(idx, len(s) - 1)], np.inf)


def fit_domain_ratio(cal_feats: np.ndarray, test_feats: np.ndarray, l2: float = 1.0, iters: int = 300):
    """Logistic regression 'is this input from the test batch?'; returns ratio fn w(x)=p/(1-p)*n_c/n_t."""
    x = np.vstack([cal_feats, test_feats])
    mu, sd = x.mean(0), x.std(0) + 1e-8
    z = np.column_stack([(x - mu) / sd, np.ones(len(x))])
    y = np.concatenate([np.zeros(len(cal_feats)), np.ones(len(test_feats))])
    w = np.zeros(z.shape[1])
    for _ in range(iters):  # full-batch gradient descent with L2 (convex, converges reliably)
        p = 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))
        g = z.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0] / len(y)
        w -= 1.0 * g
    prior = len(cal_feats) / len(test_feats)

    def ratio(feats: np.ndarray) -> np.ndarray:
        zz = np.column_stack([(feats - mu) / sd, np.ones(len(feats))])
        p = 1 / (1 + np.exp(-np.clip(zz @ w, -30, 30)))
        return np.clip(prior * p / np.clip(1 - p, 1e-6, None), 1e-3, 1e3)
    return ratio


def predict_sets_all(clean_cal: Batch, aug_cal: Batch, test: Batch, alpha: float = 0.1,
                     n_bins: int = 6) -> dict[str, np.ndarray]:
    """Boolean (N, C) prediction-set masks from every method, for one test batch."""
    pooled = Batch.concat([clean_cal, aug_cal])
    out: dict[str, np.ndarray] = {}
    lac_c, aps_c = cf.lac_scores(clean_cal.probs), cf.aps_scores(clean_cal.probs)
    out["scp"] = cf.prediction_sets(test.probs, cf.conformal_quantile(
        cf.true_class_scores(lac_c, clean_cal.y), alpha), "lac")
    out["scp_aps"] = cf.prediction_sets(test.probs, cf.conformal_quantile(
        cf.true_class_scores(aps_c, clean_cal.y), alpha), "aps")
    out["mondrian_cls"] = cf.prediction_sets(test.probs, cf.class_conditional_quantiles(
        cf.true_class_scores(lac_c, clean_cal.y), clean_cal.y, alpha, test.probs.shape[1]), "lac")
    s_pool = _lac_true(pooled)
    out["pooled_aug"] = cf.prediction_sets(test.probs, cf.conformal_quantile(s_pool, alpha), "lac")

    def conditional(cal_cov: np.ndarray, test_cov: np.ndarray, method: str = "lac") -> np.ndarray:
        # a bin needs enough calibration points to certify 1-alpha, else its quantile is +inf
        nb = max(1, min(n_bins, len(cal_cov) // max(30, int(np.ceil(3 / alpha)))))
        edges = _bin_edges(cal_cov, nb)
        qs = _binned_quantiles(cal_cov, _true(pooled, method), edges, alpha)
        q_test = qs[np.searchsorted(edges, test_cov, side="right")]
        return cf.SCORERS[method](test.probs) <= q_test[:, None]
    for name, fn in (("cond_novelty", lambda b: b.novelty), ("cond_entropy", lambda b: entropy(b.probs))):
        m = conditional(fn(pooled), fn(test)); m[np.arange(len(test.probs)), test.probs.argmax(1)] = True
        out[name] = m

    argmax = (np.arange(len(test.probs)), test.probs.argmax(1))
    est = SeverityEstimator().fit(pooled.features(), pooled.severity if pooled.severity is not None
                                  else np.zeros(len(pooled.probs)))
    m = conditional(est.predict(pooled.features()), est.predict(test.features()))
    m[np.arange(len(test.probs)), test.probs.argmax(1)] = True
    out["sacp"] = m

    s_test = est.predict(test.features())
    if aug_cal.group is not None:
        eff_target = np.concatenate([np.zeros(len(clean_cal.probs)), effective_severity(clean_cal, aug_cal)])
        est_eff = SeverityEstimator().fit(pooled.features(), eff_target)
        m = conditional(est_eff.predict(pooled.features()), est_eff.predict(test.features()))
        m[argmax] = True
        out["sacp_eff"] = m
    # --- hybrids ---
    out["raps"] = cf.prediction_sets(test.probs, cf.conformal_quantile(
        cf.true_class_scores(cf.raps_scores(clean_cal.probs), clean_cal.y), alpha), "raps")
    m = conditional(est.predict(pooled.features()), s_test, "raps"); m[argmax] = True
    out["sacp_raps"] = m
    # sacp_cqr: regress the conformity score on (severity, novelty, entropy, ...) with quantile
    # regression on one half of the pooled calibration set, then conformalise the residual on the
    # other half. Continuous adaptation (no bins) with a marginal guarantee from the residual step.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(pooled.probs)); half = len(perm) // 2
    fit_i, cal_i = perm[:half], perm[half:]
    s_pool_hat = est.predict(pooled.features())
    zf = _cqr_features(pooled, s_pool_hat)
    mu, sd = zf[fit_i].mean(0), zf[fit_i].std(0) + 1e-8
    w = fit_linear_quantile((zf[fit_i] - mu) / sd, s_pool[fit_i], 1 - alpha)
    pred = lambda z: np.column_stack([(z - mu) / sd, np.ones(len(z))]) @ w
    delta = cf.conformal_quantile(s_pool[cal_i] - pred(zf[cal_i]), alpha)
    thr = pred(_cqr_features(test, s_test)) + delta
    m = cf.lac_scores(test.probs) <= thr[:, None]; m[argmax] = True
    out["sacp_cqr"] = m
    # hybrid_union: a superset of valid sets covers at least as often as each of them.
    out["hybrid_union"] = out["cond_novelty"] | out["cond_entropy"] | out["sacp"]

    ratio = fit_domain_ratio(clean_cal.features(), test.features())
    qw = weighted_quantiles(_lac_true(clean_cal), ratio(clean_cal.features()), ratio(test.features()), alpha)
    m = cf.lac_scores(test.probs) <= qw[:, None]
    m[np.arange(len(test.probs)), test.probs.argmax(1)] = True
    out["weighted"] = m
    return out


def compare(clean_cal: Batch, aug_cal: Batch, conditions: dict[str, Batch], alpha: float = 0.1,
            n_bins: int = 6) -> dict:
    """{condition: {method: {coverage, mean_size, undercoverage}}} - undercoverage = max(0, 1-a-cov)."""
    result = {}
    pooled_nov = np.concatenate([clean_cal.novelty, aug_cal.novelty])
    for name, test in conditions.items():
        masks = predict_sets_all(clean_cal, aug_cal, test, alpha, n_bins)
        _, outside = envelope_flags(pooled_nov, test.novelty)
        inside = ~outside
        result[name] = {}
        for method, mask in masks.items():
            r = cf.evaluate_sets(mask, test.y)
            covered = mask[np.arange(len(test.y)), test.y]
            result[name][method] = {
                "coverage": r["coverage"], "mean_size": r["mean_size"],
                "undercoverage": max(0.0, 1 - alpha - r["coverage"]),
                "outside_envelope_fraction": float(outside.mean()),
                # the guarantee is only claimed inside the calibrated envelope
                "coverage_inside_envelope": float(covered[inside].mean()) if inside.any() else None,
                "coverage_outside_envelope": float(covered[outside].mean()) if outside.any() else None}
    return result
