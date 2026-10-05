"""PyTorch Dataset over the EuroSAT manifest."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class CachedEuroSATDataset(Dataset):
    """Same contract as EuroSATDataset but reads from the uint16 memmap cache."""

    def __init__(self, cache_dir, stats_json, split: str, augment: bool = False):
        d = Path(cache_dir)
        self.arr = np.load(d / "patches.npy", mmap_mode="r")
        self.labels = np.load(d / "labels.npy")
        self.idx = np.flatnonzero(np.load(d / "splits.npy") == split)
        stats = json.loads(Path(stats_json).read_text())
        self.mean = np.array(stats["mean"], dtype="float32")[:, None, None]
        self.std = np.array(stats["std"], dtype="float32")[:, None, None] + 1e-6
        self.augment = augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = (self.arr[j].astype("float32") - self.mean) / self.std
        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            x = np.rot90(x, np.random.randint(4), axes=(1, 2))
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.labels[j])


class EuroSATDataset(Dataset):
    """Loads 13-band patches, standardised with TRAIN-split statistics."""

    def __init__(self, root, manifest_csv, stats_json, split: str, augment: bool = False):
        self.root = Path(root)
        with open(manifest_csv) as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        stats = json.loads(Path(stats_json).read_text())
        self.mean = np.array(stats["mean"], dtype="float32")[:, None, None]
        self.std = np.array(stats["std"], dtype="float32")[:, None, None] + 1e-6
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with rasterio.open(self.root / r["path"]) as src:
            x = (src.read().astype("float32") - self.mean) / self.std
        if self.augment:  # flips/90deg rotations are label-preserving for nadir imagery
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            x = np.rot90(x, np.random.randint(4), axes=(1, 2))
        return torch.from_numpy(np.ascontiguousarray(x)), int(r["label"])
