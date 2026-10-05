"""Patch sources feeding the live stream.

Every source reports `kind`, which the dashboard displays, so a viewer can always tell
real imagery from synthetic test signal.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rasterio

from terraforge.data.eurosat import BANDS, CLASSES

N_BANDS = len(BANDS)


class SyntheticSource:
    """Class-conditioned synthetic spectra. For plumbing/demo only - NOT real imagery.

    `drift` adds a reflectance offset, letting the drift monitor be exercised on demand.
    """
    kind = "synthetic"

    def __init__(self, seed: int = 0, drift: float = 0.0):
        self.rng = np.random.default_rng(seed)
        self.drift = drift
        self.base = self.rng.uniform(500, 3500, size=(len(CLASSES), N_BANDS))

    def next_batch(self, n: int):
        labels = self.rng.integers(0, len(CLASSES), n)
        x = self.base[labels][:, :, None, None] + self.rng.normal(0, 150, (n, N_BANDS, 64, 64))
        return np.clip(x + self.drift * 1000, 0, 10000).astype("float32"), labels


class EuroSATSource:
    """Replays the held-out TEST split of real EuroSAT patches with their true labels."""
    kind = "eurosat-test-replay"

    def __init__(self, root: str | Path, manifest_csv: str | Path, seed: int = 0):
        self.root = Path(root)
        with open(manifest_csv) as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
        self.rng = np.random.default_rng(seed)

    def next_batch(self, n: int):
        picks = self.rng.integers(0, len(self.rows), n)
        xs, ys = [], []
        for i in picks:
            with rasterio.open(self.root / self.rows[i]["path"]) as src:
                xs.append(src.read().astype("float32"))
            ys.append(int(self.rows[i]["label"]))
        return np.stack(xs), np.array(ys)


def rgb_thumbnail(x: np.ndarray) -> bytes:
    """(C,H,W) reflectance -> HxWx3 uint8 true-colour bytes (bands B04,B03,B02)."""
    rgb = np.stack([x[3], x[2], x[1]], axis=-1) / 3000.0
    return (np.clip(rgb, 0, 1) * 255).astype("uint8").tobytes()


def mean_ndvi(x: np.ndarray) -> float:
    nir, red = x[7].astype("float64"), x[3].astype("float64")
    # (chip arrays are (C,H,W) reflectance-scale; NaN-safe even for all-zero/no-data pixels)
    d = nir + red
    return float(np.nanmean(np.where(d > 0, (nir - red) / np.where(d > 0, d, 1), np.nan)))
