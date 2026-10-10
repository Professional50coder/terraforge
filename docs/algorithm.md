# Conformal calibration under physical shift: SACP, hybrids and evaluation standards

Status: **hypothesis under test.** Implemented and verified on a controlled simulation. The
real-data comparison is pre-registered below and has not been run.

## The gap

Split conformal prediction guarantees `1 - alpha` coverage only when test data is exchangeable
with calibration data. It is known to fail under distribution shift, where re-calibration is
normally required. Earth-observation shift is structured: haze, band loss, cloud and noise are
physical corruptions with a severity. If an input's severity were known, the quantile could be
calibrated as a function of it.

## Prior art (what this is *not*)

- Conditional and group-conditional conformal calibration, including guarantees over a
  finite-dimensional class of covariate shifts: [Gibbs, Cherian & Candes](https://arxiv.org/pdf/2305.12616).
- Mondrian (group-conditional) conformal prediction, e.g. as documented in
  [MAPIE](https://mapie.readthedocs.io/en/v0.9.0/theoretical_description_mondrian.html).
- Weighted conformal prediction under covariate shift (Tibshirani, Barber, Candes & Ramdas, 2019).
- Regularised APS sets (RAPS; Angelopoulos et al., 2021).
- Conformal prediction on satellite and aerial imagery, e.g.
  [Assessing Predictive Uncertainties in Remote Sensing Image Classification via Conformal Prediction](https://elib.dlr.de/208182/).

Binning by an uncertainty or novelty score is therefore **not new**, and novelty of any combination
below is not established; that would need a systematic literature review.

## Candidate method: SACP

1. Corrupt the validation set with the physical corruption family at known severities.
2. Fit a severity estimator `s_hat(x)` (ridge regression) from label-free features (embedding, kNN
   novelty, softmax entropy, max-probability) to the applied severity.
3. Bin the pooled clean + corrupted calibration set by `s_hat`; one conformal quantile per bin.
4. At test time route each input by `s_hat` and use its bin's quantile.

What is tested is the *choice of covariate*: one learned from simulated physical severity versus raw
softmax entropy or raw embedding novelty.

### Proposition (coverage deficit)

For a test point in bin `b`, with calibration scores `~ P_b` and test scores `~ Q_b`:

    Q_b(S' <= q_b)  >=  1 - alpha - TV(P_b, Q_b).

*Proof.* `|P(A) - Q(A)| <= TV(P, Q)` for every event `A`; take `A = {S <= q_b}` and use
`P_b(A) >= 1 - alpha` from calibration. QED.

Coverage is restored exactly to the extent that the covariate is *sufficient for the shift's effect on
scores* ("severity sufficiency"). It degrades by up to `TV` when real shift contains something the
simulated family does not. Verified numerically in `tests/test_shift_conformal.py`.

## Hybrids (exploratory)

| Method | Construction | Reason to try it |
|---|---|---|
| `raps` | Regularised APS, clean calibration | Strong existing baseline; fixes APS over-conservatism |
| `sacp_raps` | SACP bins with the RAPS score | Adaptive and size-regularised |
| `sacp_cqr` | Quantile-regress the score on (severity, novelty, entropy, ...) on half of the pooled calibration set, conformalise the residual on the other half | Continuous adaptation with no hard bins; marginal guarantee from the residual step |
| `hybrid_union` | Union of the novelty-, entropy- and severity-conditioned sets | A superset covers at least as often as each member (proved, tested), at a cost in size |
| `sacp_eff` | SACP with *effective* severity (see below) | Removes the incomparable-scales weakness |

Hybrids are exploratory: with many variants, picking a winner from one benchmark is how false
discoveries happen. The **primary** comparison below is fixed in advance and is not changed by the
hybrids; a hybrid counts only if it also wins on the held-out-family protocol.

## Evaluation standards

1. **Validity** - coverage at the nominal `alpha`, and worst under-coverage across conditions.
2. **Matched-coverage efficiency** - the mean set size each method needs to reach 90% coverage, found
   by sweeping `alpha` and reading each method's coverage-size frontier at one common coverage.
   Coverage at a nominal `alpha` rewards conservatism; a method that over-covers looks safe but large.
   This compares the quality of the ranking itself. It uses test labels to place the operating point,
   so it is an analysis standard, never a deployed rule.
3. **Calibrated envelope** - a conformal p-value on novelty,
   `p = (1 + #{calibration novelty >= test novelty}) / (n + 1)`, flags inputs outside the calibrated
   region (`p < gamma`). Under exchangeability about a `gamma` fraction of in-region inputs are flagged by
   chance; far-shifted inputs are flagged almost always (tested). Coverage is reported separately inside
   and outside the envelope, so the guarantee is only ever stated where it can hold.
4. **Effective severity** - nominal severities are family-specific numbers (fraction of bands dropped,
   noise std, haze offset) and cannot share one axis. Severity is instead defined as the measured
   inflation of the mean true-class score over clean data, normalised by the largest inflation seen.
5. **Paired bootstrap confidence intervals** - 95% CIs on the coverage and size differences between methods,
   resampling test points in pairs. Point estimates are never used to claim a win.
6. **Minimum bin size** - bins are merged so each has enough calibration points to certify `1 - alpha`;
   otherwise its quantile is `+inf` (a silent full-class set).

## Evidence so far: controlled simulation only

A simulator with known severity (class signal weakens and embedding noise grows with severity). It
checks the mechanism and the code, **not** satellite data. Regenerate both tables with
`python scripts/simulate_shift.py` (about 20 s on CPU; seeds 0-4 for validity, 0-3 for efficiency).

Validity - coverage / mean set size at target 0.90 (mean of 5 seeds):

| method | s=0.0 | s=0.2 | s=0.4 | s=0.6 | s=0.8 |
|---|---|---|---|---|---|
| scp | 0.941 / 1.01 | 0.868 / 1.01 | 0.758 / 1.01 | 0.604 / 1.01 | 0.441 / 1.00 |
| raps | 0.993 / 1.95 | 0.982 / 2.27 | 0.954 / 2.57 | 0.901 / 2.82 | 0.807 / 2.99 |
| pooled_aug | 0.984 / 1.25 | 0.951 / 1.37 | 0.884 / 1.52 | 0.773 / 1.64 | 0.623 / 1.71 |
| cond_novelty | 0.954 / 1.07 | 0.930 / 1.34 | 0.903 / 1.77 | 0.852 / 2.24 | 0.762 / 2.58 |
| cond_entropy | 0.985 / 1.31 | 0.964 / 1.56 | 0.925 / 1.88 | 0.854 / 2.21 | 0.744 / 2.47 |
| sacp | 0.966 / 1.11 | 0.952 / 1.45 | 0.930 / 1.96 | 0.882 / 2.45 | 0.796 / 2.81 |
| sacp_raps | 0.983 / 1.72 | 0.968 / 1.90 | 0.943 / 2.36 | 0.898 / 2.79 | 0.813 / 3.06 |
| sacp_cqr | 0.970 / 1.12 | 0.944 / 1.41 | 0.911 / 1.97 | 0.870 / 2.72 | 0.807 / 3.33 |
| hybrid_union | 0.989 / 1.34 | 0.975 / 1.68 | 0.950 / 2.14 | 0.898 / 2.58 | 0.809 / 2.90 |
| weighted | 0.941 / 1.01 | 0.899 / 1.35 | 0.881 / 2.81 | 0.872 / 3.89 | 0.871 / 4.61 |

Matched-coverage efficiency - set size needed to reach 90% coverage (mean of 4 seeds; `inf` = never):

| method | s=0.0 | s=0.3 | s=0.6 |
|---|---|---|---|
| scp | 1.00 | 1.31 | inf |
| scp_aps | 1.36 | 1.71 | 2.81 |
| raps | 1.29 | 1.58 | 2.81 |
| pooled_aug | 1.00 | 1.33 | 2.64 |
| cond_novelty | 1.00 | 1.43 | 2.78 |
| cond_entropy | 1.03 | 1.34 | 2.65 |
| sacp | 1.00 | 1.34 | 2.68 |
| sacp_raps | 1.13 | 1.45 | 2.83 |
| sacp_cqr | 1.01 | 1.40 | 3.12 |
| hybrid_union | 1.03 | 1.31 | 2.61 |
| weighted | 1.00 | 2.03 | 4.14 |

**Honest reading.**
- The problem is real: marginal conformal coverage falls from 0.94 to 0.44 as severity rises, and at
  severity 0.6 it cannot reach 90% at any size.
- What fixes it is *having a covariate that indexes the shift at all*. Once one is used, the methods
  that do (`cond_entropy`, `sacp`, `cond_novelty`, `hybrid_union`) sit within a few percent of each other
  at matched coverage (2.61 to 2.78 at s=0.6), which is within noise for 4 seeds.
- **A plain `pooled_aug` baseline (one quantile over the pooled clean + corrupted calibration set, no
  covariate at all) needs 2.64 at s=0.6, as small as SACP (2.68).** It is weaker on validity at nominal
  alpha (0.773 coverage at s=0.6 vs 0.882) but equal on the matched-coverage ranking. The simulation
  therefore does not separate SACP from the simplest augmentation baseline on efficiency, which is why
  `pooled_aug` is one of the pre-registered spoilers below.
- **SACP's advantage over simple entropy- or novelty-conditioning is not demonstrated here.** In this
  simulation novelty and entropy already track severity almost perfectly.
- `sacp_cqr`, `sacp_raps` and weighted conformal are *worse* at matched coverage. Weighted conformal
  also needs the unlabeled test batch. They are kept in the benchmark as negative results.
- Severity 0.8 lies beyond the simulated calibration range (maximum 0.75); coverage there is
  extrapolation, which is exactly what the envelope is meant to flag.

## Pre-registered real-data test

Written before any real-data run; amended once, also before any real-data run, to add the efficiency
and CI criteria (the amendment follows the simulation above).

Protocol (`scripts/analyze.py`, section `shift_conformal`): calibrate on clean validation data plus all
corruption families **except one**; test on the held-out family at two severities; repeat for every family.

SACP is **supported** only if, against the best of `cond_novelty` and `cond_entropy`, on trained models:
1. its worst under-coverage is no larger, **and**
2. its mean set size is no more than 10% larger, **and**
3. its size at 90% matched coverage is smaller, with a paired-bootstrap 95% CI that excludes zero.

It is **not supported** if any condition fails, or if `raps`, `weighted` or `pooled_aug` match its coverage
at equal size. Given the simulation, a null result against `cond_entropy` is the expected outcome and
will be reported as such.

The verdict is computed, not argued: `python scripts/sacp_verdict.py runs/analysis_<model>.json` applies
these rules to the `analyze.py` output (`training/preregistered.py`, unit-tested). Details the
pre-registration left open are resolved in the strict direction and printed with the verdict.

## Known limitations

- Simulated corruptions are not real atmospheric-correction differences. The Level-1C to Level-2A study
  (`data/chip_fetcher.py`) is the check against real shift.
- The envelope uses novelty alone; a shift that leaves embeddings in place but changes labels is invisible to it.
- Marginal coverage within a bin is not coverage for every individual input or region.
