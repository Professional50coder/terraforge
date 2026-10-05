# Dataset: EuroSAT (multispectral)

**What:** 27,000 Sentinel-2 L1C patches, 64x64 px at 10 m, 13 bands, 10 land-use classes
(AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop,
Residential, River, SeaLake), expert-labelled.

**Why this one:** real labels, small enough for a free CI runner, and multispectral
(the red-edge and SWIR bands are where vegetation signal lives, which RGB datasets discard).

**What it is not:** a vegetation *stress* dataset. It labels land cover. The stress/NDVI
anomaly layer is a separate later stage and any labels there must be described as proxies.

## Decisions and the reasoning behind them

| Decision | Reason |
|---|---|
| Manifest CSV (path, label, split) | Reproducible, diffable; separates split membership from tensor loading. |
| Stratified split, 70/15/15, seeded | Every class appears in each split in equal proportion; rerunning gives identical splits. |
| Normalisation stats from **train only** | Computing them on val/test leaks information about held-out data. |
| Model selection on **val**, report on **test** once | Choosing the best epoch on test would make the test number optimistic. |
| Flip / 90-degree rotation augmentation | Label-preserving for nadir imagery (no "up" in a satellite view). |
| Macro-F1 beside accuracy | Pasture vs HerbaceousVegetation confusion is hidden by accuracy. |

## Known weaknesses

- Patches are cropped from larger scenes, so neighbouring patches can share pixels/context.
  A random split can therefore leak spatial neighbours between train and test and inflate
  scores. A geographic split would be stricter; EuroSAT does not ship coordinates in this form.
- L1C is top-of-atmosphere, not atmospherically corrected like the L2A used in the STAC layer.
  A model trained here does not transfer to L2A without recalibration.
- Single time step: no phenology, which is what real stress detection needs.
