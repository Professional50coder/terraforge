# Architecture

```
STAC catalog -> STACClient -> RasterProcessor (xarray) -> [future: stress layer]
EuroSAT zip  -> eurosat.py (manifest, stats) -> EuroSATDataset -> ModelTrainer -> metrics -> MLflow
```

## Modules

- `data/stac_client.py` - validated queries (bbox, dates, cloud) against Earth Search; returns
  items least-cloudy first; fails loudly on missing assets.
- `data/raster_processor.py` - GeoTIFF -> xarray with CRS and grid preserved; reflectance scaling;
  NDVI; SCL cloud mask. CRS and transform travel with the data so geography is never lost.
- `data/eurosat.py`, `data/torch_dataset.py` - indexing, stratified splits, train-only stats, loader.
- `models/` - CNN baseline, from-scratch ViT, foundation-model wrapper (planned).
- `training/` - model-agnostic trainer; numpy metrics.

## Compute policy

Free-tier limits matter. CI does not run on every push (pull request / manual only, with
cancel-in-progress). Heavy jobs are manual `workflow_dispatch` workflows. Free runners are CPU
only, so GPU-scale training needs a Codespace/Colab/Kaggle GPU and is documented, not assumed.
