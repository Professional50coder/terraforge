# Handoff: where the project stands and how to continue

Written at the end of a working session so anyone (including the author, later) can resume
without the original context. Nothing here contains credentials.

## 1. Snapshot

| Item | State |
|---|---|
| Repository | `Professional50coder/terraforge`, public, branch `main` |
| Last verified full test run | 138 tests passing (venv, Python 3.11; re-verified on CPU torch) |
| Real training results | **pending**, running on Kaggle (section 3) |
| Results table in README | still says "pending GPU run" |
| UI redesign | not started (ideas in section 6) |
| Docker image | written, **never built** (no Docker on the dev machine) |

## 2. What exists and works

Verified by tests unless marked otherwise.

- **Data:** STAC client, windowed COG chip fetcher (offset detected from pixels, parallel reads, timeouts), EuroSAT-MS manifest, stratified splits, uint16 memmap cache, train-only statistics.
- **Models:** CNN baseline, ViT written from scratch, masked-autoencoder pretraining, attention rollout.
- **Trust layer:** temperature scaling and ECE, split-conformal sets (LAC, APS, RAPS, class-conditional), selective prediction (risk-coverage, AURC, error AUROC, kNN novelty), robustness benchmark, PSI drift monitor, embedding index.
- **Candidate method:** SACP plus baselines and hybrids in `training/shift_conformal.py`; protocol and pre-registered criteria in `docs/algorithm.md`.
- **Service:** FastAPI (`/predict`, `/similar`, `/explain`, `/analyze`, `/agent`, `/sky`, `/health`), WebSocket stream, live dashboard (`/`), globe page (`/globe`), tool-calling agent with geocoding (checked live against the hosted LLM and the geocoder).
- **Cross-checks:** `tests/test_reference_math.py` compares hand-written maths against scikit-learn, SciPy, NumPy and PyTorch.

Not verified: how the globe and dashboard *render* in a real browser (the browser-automation extension was not connected; one headless screenshot showed the panels with a blank globe, cause undetermined, possibly just no WebGL in headless mode).

## 3. The Kaggle training process

### What it is
A private Kaggle script kernel, `<kaggle-username>/terraforge-train`, with GPU and internet enabled. It
runs the whole experiment suite on Kaggle's hardware so the dev machine does no training.

### What the kernel does (`kaggle/entry.py`)
1. `git clone --depth 1` this repository (so it runs the code **as of the moment it started**; push before launching).
2. `pip install -e . rasterio geopandas pystac-client fastapi httpx` (PyTorch is preinstalled on Kaggle).
3. `scripts/build_dataset.py`: download EuroSAT-MS (Hugging Face mirror first), build manifest, uint16 cache and train-only statistics. Large intermediates go under `/kaggle/temp` so they are not saved as output.
4. `scripts/pretrain_mae.py`: 80 epochs of masked-autoencoder pretraining (no labels).
5. For seeds 0, 1, 2: train `cnn` (30 epochs), `vit` from scratch (40 epochs, lr 5e-4), and `vit` initialised from the MAE encoder (30 epochs, lr 3e-4).
6. A 2-process DDP smoke check (`training/distributed.py`).
7. Copy small result files to `/kaggle/working/results`.

A failed step is recorded and the next steps still run, except cloning, installing and dataset building, which stop the run.

### Expected outputs
`results/` should contain, per run tag (`cnn_s0`, `vit_scratch_s0`, `vit_mae_s0`, and seeds 1 and 2):
`<tag>.json` (test accuracy, macro-F1, per-class F1, confusion matrix, calibration, robustness report, training history), `<tag>.pt` (best-epoch weights), plus `mae_encoder.pt`, `ddp_check.log` and `env.json` (GPU name, torch version, elapsed time, list of failed commands).

### How it was launched and how to operate it
Credentials come from a `.env` kept **outside the repository**; point `TERRAFORGE_ENV_FILE` at it. The launcher
imports only `KAGGLE_USERNAME` and `KAGGLE_KEY` (an allowlist) into the child process and never prints them.

```
$env:TERRAFORGE_ENV_FILE = '<path to your .env>'
python kaggle\launch.py push      # (re)start the kernel from current main
python kaggle\launch.py status    # QUEUED / RUNNING / COMPLETE / ERROR
python kaggle\launch.py pull      # download output into runs_kaggle\ (git-ignored)
```

State when this was written: the kernel had been `RUNNING` since late evening; there was no way to know
how far it had got without opening it on kaggle.com. Kaggle enforces its own session limit; if it expires
before finishing, `pull` still returns whatever was written, and `env.json` lists the failures. Re-run
`push` after fixing anything (it clones the latest `main`).

### If it failed, check in this order
1. **Internet disabled / phone verification missing** on the Kaggle account (needed for internet access): the `git clone` or the download fails first.
2. **Dataset download**: the original host times out; the Hugging Face mirror is tried first. Both failing stops the run.
3. **Out of time**: Kaggle session limits. Reduce seeds or epochs in `kaggle/entry.py` (for example only seed 0).
4. **A code error**: the kernel log shows the failing command; reproduce locally with `--subset 128` (smoke runs).

## 4. After the results arrive: the resume checklist

1. `kaggle\launch.py pull`, then confirm the expected files exist.
2. For each model of interest run the trust analysis locally (CPU is enough):
   ```
   python scripts/analyze.py --arch cnn --weights runs_kaggle/cnn_s0.pt --out runs/analysis_cnn_s0.json
   python scripts/analyze.py --arch vit --weights runs_kaggle/vit_scratch_s0.pt --out runs/analysis_vit_scratch_s0.json
   python scripts/analyze.py --arch vit --weights runs_kaggle/vit_mae_s0.pt --out runs/analysis_vit_mae_s0.json
   ```
   Each also writes `<name>_conformal.json` (temperature + conformal quantile) for serving. `analyze.py` needs
   the local data cache: run `python scripts/build_dataset.py` first if `data/processed/` is missing.
3. Fill the README results table from the `<tag>.json` files (mean and spread over the 3 seeds, not a single run).
4. Evaluate SACP against the criteria in `docs/algorithm.md` *exactly as pre-registered*, and report the outcome whichever way it goes. The simulation predicts SACP will tie with entropy-conditioning.
5. Serve the real model: `python scripts/serve.py --arch vit --weights <weights>.pt --conformal runs/analysis_<name>_conformal.json --root <EuroSAT root>`; the dashboard then replays real test patches and drops the untrained banner.
6. Run the L1C to L2A study (section 6) with the trained weights.
7. Update `docs/decisions.md` and the README; keep every number traceable to a run.

## 5. Gotchas learned the hard way

| Gotcha | Detail |
|---|---|
| Ignore rules | An unanchored `data/` in `.gitignore` also ignored `src/terraforge/data/`, so a module was never committed. Rules are now `/data/`, `/runs/`, `/runs_kaggle/`. After adding files, run `git status --ignored` on `src/` and verify with a fresh clone. |
| Python version | The project venv must be Python 3.11; 3.14 has no geospatial wheels and pip tries to compile from source. |
| Reflectance offset | Never trust the catalog's declared offset or the baseline number: decide from pixel statistics (`offset_present`). The real Paris scene declared an offset its data did not have. |
| Imagery latency | Remote chip reads took 11 s to 244 s for the same request. Timeouts, retries, a cache and a 504 path exist; latency is not solved. |
| Hosted LLM | Default Python user-agent got HTTP 403; retired model names got 404. The agent sends an honest user-agent and walks a model list. |
| Secrets | Load only needed keys from an external `.env` via `load_env(path, allow=...)`. A shared `.env` holds unrelated credentials. Rotate any credential that appeared in logs or chats. |
| CI | Runs on pull requests or manually only, to save free-tier minutes. Local tests: `pytest` (about 100 s). |
| Smoke runs | `--subset N` takes evenly spaced samples; the manifest is class-ordered, so taking the first N gives one class. |
| Background processes | Stop servers and pollers you start; a wait loop on a file that never appears runs forever. |

## 6. Next work, in priority order

1. **Results and trust analysis** (section 4). Highest value; everything else waits on it.
2. **L1C to L2A domain-gap study.** Fetch real chips at known locations with `data/chip_fetcher.py`, compare predictions with and without offset harmonisation, and measure calibration and conformal coverage on the real shift. This is the only place the project meets real, not simulated, shift.
3. **Look at the UI in a real browser** and fix what is broken. Then improve it. Ideas drawn from common 3D-globe viewers: place search with fly-to, one-tap place chips, a time scrubber for day/night, a pulsing selection marker, shareable links (coordinates in the URL hash), keyboard shortcuts, a clearer result card with a confidence bar, and a mobile bottom-sheet layout.
4. **Spatial-leakage audit** results from `analyze.py` into the docs.
5. **Smaller:** build and verify the Docker image; log runs to MLflow; vegetation-stress layer using clearly labelled proxy signals; a short technical report once results exist.

## 7. Where things are

See the layout section of the [README](../README.md), the plan with acceptance tests in
[ROADMAP.md](ROADMAP.md), design reasoning in [decisions.md](decisions.md), the research questions in
[problem.md](problem.md), and the method and evaluation standards in [algorithm.md](algorithm.md).
