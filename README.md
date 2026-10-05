# TerraForge

A modular Earth-Observation ML pipeline for Sentinel-2: STAC discovery, geospatial
preprocessing, leakage-aware dataset engineering, a CNN baseline versus a Vision
Transformer written from scratch, and (planned) EO foundation-model fine-tuning,
MLflow tracking, a FastAPI service and distributed training.

The goal is not a leaderboard number. It is a pipeline where every decision is
documented, tested and reproducible, and where the limitations are stated.

## Status

| Stage | State |
|---|---|
| Raster processing (rasterio + xarray): reflectance, NDVI, SCL cloud mask | done, tested |
| STAC acquisition (Earth Search, validated queries) | done, unit-tested (live search not yet exercised) |
| EuroSAT-MS dataset: stratified manifest, train-only statistics | done, tested |
| CNN baseline, from-scratch ViT, trainer, numpy metrics | done, tested |
| Real training runs and results | in progress, see `docs/algorithm_baseline.md` |
| TerraTorch foundation-model fine-tune | planned |
| Vegetation-stress layer (NDVI anomaly over time, proxy labels) | planned |
| MLflow registry, FastAPI, Docker, DDP | planned |

The full plan, with acceptance tests per item, is in [docs/ROADMAP.md](docs/ROADMAP.md).

## Design principles

- **No leakage.** Normalisation statistics come from the train split only; model
  selection uses validation; the test set is evaluated once.
- **Geography travels with the data.** CRS and affine transform stay attached to
  every array.
- **Fail loudly.** Invalid bounding boxes, missing STAC assets and bad patch sizes
  raise, rather than producing silent garbage.
- **Honest scope.** EuroSAT labels land cover, not stress. Stress work will use
  proxy labels and be described as such.
- **Cost-aware.** CI runs on pull request or manually, never on every push.

## Layout

```
src/terraforge/
  data/       stac_client, raster_processor, eurosat, torch_dataset
  models/     cnn, vit (from scratch), foundation (planned)
  training/   trainer, evaluation (numpy metrics), distributed (planned)
scripts/      build_dataset.py, train.py
docs/         architecture, dataset (decisions + weaknesses), algorithm_baseline
tests/        geospatial, dataset, metrics, model tests
```

## Run it

```bash
pip install -e ".[train]"
python scripts/build_dataset.py            # downloads EuroSAT-MS, writes manifest + stats
python scripts/train.py --model cnn --root data/raw/<extracted-root> --epochs 10
python scripts/train.py --model vit --root data/raw/<extracted-root> --epochs 10
pytest
```

A devcontainer is provided for GitHub Codespaces.

## Documentation

- [Architecture](docs/architecture.md)
- [Dataset: choices and known weaknesses](docs/dataset.md)
- [Algorithm baseline and protocol](docs/algorithm_baseline.md)
