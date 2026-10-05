"""EuroSAT (multispectral) dataset indexing, splitting and statistics.

EuroSAT: 27,000 Sentinel-2 L1C patches, 64x64 px, 13 bands, 10 land-use
classes, labelled by experts. Layout on disk: <root>/<ClassName>/<ClassName>_<n>.tif

Why a manifest instead of loading everything: a CSV of (path, label, split)
is tiny, reproducible (seeded), diffable, and decouples *what is in each split*
from *how tensors are loaded*. Splits are stratified so every class appears in
train/val/test in the same proportion - otherwise rare-class metrics are noise.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio

CLASSES = (
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
)
# Sentinel-2 band order in the EuroSAT GeoTIFFs.
BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
         "B8A", "B09", "B10", "B11", "B12")
DOWNLOAD_URL = "https://madm.dfki.de/files/sentinel/EuroSATallBands.zip"


def index_files(root: str | Path) -> list[tuple[str, int]]:
    """Return sorted (relative_path, class_index) for every tif under root."""
    root = Path(root)
    rows = []
    for idx, cls in enumerate(CLASSES):
        files = sorted((root / cls).glob("*.tif"))
        if not files:
            raise FileNotFoundError(f"no tifs for class {cls} under {root}")
        rows += [(str(f.relative_to(root)).replace("\\", "/"), idx) for f in files]
    return rows


def stratified_split(rows, fractions=(0.7, 0.15, 0.15), seed=42):
    """Assign each row a split, preserving class proportions per split."""
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for i, (_, label) in enumerate(rows):
        by_class.setdefault(label, []).append(i)
    split = [None] * len(rows)
    names = ("train", "val", "test")
    for idxs in by_class.values():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_tr = int(round(fractions[0] * len(idxs)))
        n_va = int(round(fractions[1] * len(idxs)))
        for j, i in enumerate(idxs):
            split[i] = names[0 if j < n_tr else 1 if j < n_tr + n_va else 2]
    return split


def write_manifest(root, out_csv, seed=42):
    rows = index_files(root)
    split = stratified_split(rows, seed=seed)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "class", "split"])
        for (p, label), s in zip(rows, split):
            w.writerow([p, label, CLASSES[label], s])
    return len(rows)


def band_statistics(root, manifest_csv, max_files=None):
    """Per-band mean/std from the TRAIN split only (never val/test: leakage)."""
    with open(manifest_csv) as f:
        paths = [r["path"] for r in csv.DictReader(f) if r["split"] == "train"]
    if max_files:
        paths = paths[:max_files]
    n = 0
    s = np.zeros(len(BANDS))
    sq = np.zeros(len(BANDS))
    for p in paths:
        with rasterio.open(Path(root) / p) as src:
            a = src.read().astype("float64")
        s += a.sum(axis=(1, 2))
        sq += (a ** 2).sum(axis=(1, 2))
        n += a.shape[1] * a.shape[2]
    mean = s / n
    std = np.sqrt(sq / n - mean ** 2)
    return {"bands": list(BANDS), "mean": mean.tolist(), "std": std.tolist(),
            "n_train_files": len(paths)}


def save_stats(stats, path):
    Path(path).write_text(json.dumps(stats, indent=2))
