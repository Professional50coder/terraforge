# The problem: knowing when an Earth-observation model is wrong

## Why classification is not the hard part

Land-cover classification on benchmark patches is close to solved: well-tuned CNNs and
transformers exceed 95% accuracy on clean, held-out EuroSAT data. What is not solved is the
operational question that follows every prediction: **should this particular answer be believed?**

That question is hard for each of the obvious actors:

| Actor | Why it cannot answer reliably |
|---|---|
| A human analyst | Cannot inspect millions of scenes, and cannot judge 13-band reflectance statistics by eye. |
| A general-purpose LLM | Can describe an image fluently but cannot measure pixels, check cloud cover, or know the model's error rate on this input. Fluency makes a wrong answer *more* persuasive. |
| A classifier's softmax | Trained to be confident. It is routinely over-confident, and stays confident on inputs unlike anything seen in training. |

In Earth observation the gap between benchmark and deployment is unusually wide, because the
inputs shift in physically structured ways:

- **Processing level.** Training data is top-of-atmosphere (L1C); live imagery is surface
  reflectance (L2A), with a reflectance offset introduced at processing baseline 04.00 and no B10 band.
- **Atmosphere.** Haze and thin cirrus change band ratios without changing the land.
- **Sensor and band faults.** A missing or corrupted band is a realistic failure.
- **Geography and season.** A model trained on one region meets another.
- **Spatial autocorrelation.** Patches cut from the same scene are near-duplicates, so a random
  split can overstate accuracy.

## What TerraForge contributes

Not a new architecture. A rigorous, reproducible treatment of *trust* for EO classifiers, and a
product that exposes it. Established tools from selective prediction, uncertainty quantification
and robustness evaluation are assembled and, importantly, **tested against the shifts that matter
in EO**, with the failures reported.

### Research questions

Each has a metric, a protocol, and a place where the result will be recorded. None is claimed
before it is measured.

**Q1. How far do conformal guarantees survive real EO shift?**
Split-conformal prediction sets (LAC and APS scores, with class-conditional/Mondrian variants) give
a finite-sample marginal coverage guarantee of `1 - alpha`, with no assumption on the model, but
*only under exchangeability*. We calibrate on the validation split, confirm coverage on clean test
data, then measure empirical coverage and set size as haze, band dropout, cloud and noise increase.
The deliverable is the curve showing where the guarantee fails, not just that it holds on clean data.
*Code:* `training/conformal.py`, `scripts/analyze.py`.
*Verified:* coverage equals `k/(n+1)` exactly for continuous scores in simulation; the guarantee is
shown to fail under label shift; quantile definition cross-checked two independent ways.

**Q2. Does the model's confidence, or embedding novelty, rank its own errors last?**
Measured with the risk-coverage curve, AURC, coverage at a fixed accuracy bar (95% / 99%), and the
AUROC of confidence and of k-NN embedding distance at separating wrong from correct predictions.
Softmax confidence and an embedding-space novelty score are compared head to head, because they fail
differently: softmax can be confidently wrong off-distribution, while novelty can flag it.
*Code:* `training/selective.py`. *Verified:* error-AUROC and rank computation against scikit-learn
and SciPy, including ties.

**Q3. How much of the headline accuracy is spatial leakage?**
For every test patch, find its nearest training patch in embedding space; compare accuracy on
test patches that have a near-duplicate neighbour against those that do not. A large gap means
neighbours are flattering the score. The reference set is a subsample of train, so the measured
near-duplicate fraction is a lower bound.
*Code:* `scripts/analyze.py` (`leakage_audit`).

**Q4. What does the Level-1C to Level-2A gap cost, and can it be recovered?**
Measured on real chips fetched through STAC at known locations, comparing predictions on L2A chips
with and without offset harmonisation and under calibration. *Status: pipeline built
(`data/chip_fetcher.py`); the quantitative study needs trained weights and is not yet run.*

### Supporting method

- **Self-supervised pretraining (masked autoencoder)** on unlabeled multispectral patches: with
  75% of patches hidden the encoder must learn spectral and spatial structure. Evaluated against
  training from scratch under the same protocol.
- **Calibration** (temperature scaling, ECE) so reported confidence is meaningful.
- **Drift monitoring** (population stability index per band) for the deployed model.
- **Retrieval-grounded explanation:** nearest labelled patches in embedding space ground the
  language explanation; the language model only phrases computed facts.

## Correctness of the mathematics

Every hand-written numerical routine is cross-checked against an independent reference in
`tests/test_reference_math.py`: confusion matrix and macro-F1 vs scikit-learn; error AUROC vs
`roc_auc_score` on tie-heavy data; average ranks vs SciPy; the conformal quantile against NumPy's
`inverted_cdf` and by direct counting; k-NN cosine novelty vs brute force; PSI and ECE vs direct
formulas; temperature scaling vs a SciPy scalar optimiser; the transformer's multi-head attention vs
PyTorch's own `scaled_dot_product_attention`; patch embedding vs an explicit patch-wise linear map;
patchify invertibility; and the warmup-cosine schedule at its endpoints and midpoint.

## What this does not claim

- Conformal prediction, selective prediction, calibration and MAE pretraining are established
  methods. The contribution here is a careful, reproducible application to EO with the failure modes measured,
  not a new algorithm.
- Marginal coverage is not coverage for every individual input or every region.
- Weak or proxy signals (for example NDVI-based stress) are never presented as ground truth.
