# TerraForge roadmap

One document for everything planned. Each item has an acceptance test: it is only
"done" when the test passes, and anything not yet built says so.

Current status and how to resume: see [HANDOFF.md](HANDOFF.md).

Legend: **done** (code + tests), **running** (in progress), **planned**.

## Pipeline at a glance

```
 STAC / EuroSAT -> preprocessing -> cache -> train (CNN | ViT | MAE-pretrained | EO-FM)
      -> evaluate (F1, calibration, robustness) -> registry -> serve (REST + WebSocket)
      -> monitor (drift, confidence) -> UI (globe, voice/text agent)
```

## Phase 1 - Data foundation

| Item | State | Acceptance |
|---|---|---|
| Raster I/O, reflectance scaling, NDVI, SCL cloud mask (rasterio + xarray) | done | CRS and grid preserved; zero denominator -> NaN |
| STAC client with validated queries | done | invalid bbox/dates/cloud rejected; missing asset raises |
| EuroSAT-MS manifest, stratified split, train-only stats | done | no path in two splits; stats ignore val/test |
| uint16 memmap cache | running | cache equals file reads; stats match |
| GeoPandas AOI layer (parcels, footprints, area in hectares) | planned | footprint area matches reference within 1% |

## Phase 2 - Models and training

| Item | State | Acceptance |
|---|---|---|
| CNN baseline | done | output shape; trains |
| ViT from scratch (own attention, patch embed, CLS, pos-emb) | done | overfits a tiny batch; attention rows sum to 1 |
| Trainer, numpy metrics, warmup+cosine LR, EMA, label smoothing | done | metrics hand-checked; schedule shape tested |
| Masked-autoencoder pretraining -> ViT transfer | done (code) | loss falls; weights load into classifier. Pretraining run: planned |
| Real CNN vs ViT vs MAE-ViT results, multiple seeds | running | mean +- std over >= 3 seeds, same protocol |
| TerraTorch / pretrained EO foundation model fine-tune | planned | beats from-scratch ViT, or the report says why not |
| Mixed precision + GPU run (Kaggle/Colab free GPU) | planned | same metrics reproduced on GPU |

## Phase 3 - Trust and evaluation

| Item | State | Acceptance |
|---|---|---|
| Calibration: ECE + temperature scaling | done | ECE falls, argmax unchanged |
| Robustness benchmark (band dropout, noise, haze, cloud) | done (code) | deterministic report. Real model report: planned |
| Attention rollout + occlusion faithfulness check | done (code) | rollout is a distribution; faithfulness measured on trained model |
| Spatial-leakage audit (near-duplicate detection via embeddings) | planned | report of test patches with a near twin in train |
| L1C->L2A domain-gap measurement | planned | accuracy gap on real STAC L2A chips quantified |
| Vegetation-stress layer (NDVI anomaly over time, proxy labels) | planned | clearly labelled proxy; sanity-checked against known drought periods |
| Known-limits page | planned | every limitation has a test or a measurement behind it |

## Phase 4 - Serving and MLOps

| Item | State | Acceptance |
|---|---|---|
| Inference engine: chunked parallel batches, calibration, embeddings | done | 70-patch batch returns valid distributions |
| FastAPI: /predict /similar /explain /health | done | bad shape, oversize, wrong bands -> 422 |
| WebSocket live stream | done | hello + batches with thumbnails and neighbours |
| Vector index + retrieval | done | exact match ranks first; save/load round-trips |
| RAG explainer (Groq / HF / Ollama / template) | done | LLM unavailable -> template, request never fails |
| Drift monitoring (PSI per band) | done (code) | shifted band flagged, others stable |
| MLflow tracking + model registry | planned | every run logged; best model registered |
| ONNX export with parity check | done (script) | max abs diff vs PyTorch < 1e-3 |
| Docker + docker-compose (api, mlflow) | planned | `docker compose up` serves the dashboard |
| DDP module + SLURM launch doc | done (code) / planned (doc) | replicas stay identical; doc has working sbatch |
| CI: pull-request/manual only, cancel-in-progress | done | no minutes used on plain pushes |

## Phase 5 - Product experience

Default view is simple; complexity is opt-in via an **Engineer view** toggle.

| Item | State | Acceptance |
|---|---|---|
| Live dashboard (stream, class mix, latency, drift) | done | served at `/`; labels untrained/synthetic honestly |
| 3D globe (globe.gl): real sun/moon day-night from astronomy formulas | planned | terminator matches a reference ephemeris |
| Live Sentinel-2 orbit tracks (public TLEs) + latest scene footprints (STAC) | planned | positions update; footprints come from real catalog items |
| Click a point -> fetch real chip -> classify -> plain-language answer | planned | works end to end; domain-gap caveat shown |
| Text agent (Groq LLM with tool calls into our API) | planned | answers cite tool output; refuses to invent |
| Voice mode (browser speech recognition + synthesis -> same agent) | planned | works in Chrome/Edge; text fallback elsewhere |
| Compare-dates slider, alerts, reduced-detail mode for weak devices | planned | - |

## Phase 6 - Documentation and proof

- Technical report with real numbers and failure analysis.
- Architecture/system-design doc: trade-offs, failure modes, scaling path.
- Per-stage design notes: why this approach, alternatives considered, what breaks first,
  how it is monitored.

## Engineering rules (apply to everything)

1. No number in the docs that did not come from a run in this repo.
2. Test set is touched once; selection happens on validation.
3. Demo/untrained/synthetic states are always labelled on screen and in responses.
4. Secrets never enter the repo; they load from an external `.env` at runtime.
5. Free-tier cost awareness: heavy jobs are manual; CI is PR/manual only.
6. No AI attribution in commits, docs or code.

## Compute plan (free tiers)

| Job | Where |
|---|---|
| Unit tests, small CPU runs | local machine |
| Reproducible checks | GitHub Actions (manual) |
| Interactive dev | GitHub Codespaces (devcontainer included) |
| GPU training, foundation-model fine-tune | Kaggle / Colab free GPU |
