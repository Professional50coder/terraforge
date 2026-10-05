import csv
from collections import Counter

import numpy as np
import rasterio
from rasterio.transform import from_origin

from terraforge.data import eurosat


def _make_root(tmp_path, per_class=20):
    for ci, cls in enumerate(eurosat.CLASSES):
        d = tmp_path / cls
        d.mkdir()
        for k in range(per_class):
            with rasterio.open(d / f"{cls}_{k}.tif", "w", driver="GTiff", height=4,
                               width=4, count=13, dtype="uint16", crs="EPSG:32631",
                               transform=from_origin(0, 40, 10, 10)) as dst:
                dst.write(np.full((13, 4, 4), 1000 * (ci + 1), dtype="uint16"))
    return tmp_path


def test_split_is_stratified_and_reproducible(tmp_path):
    root = _make_root(tmp_path)
    rows = eurosat.index_files(root)
    s1 = eurosat.stratified_split(rows, seed=1)
    assert s1 == eurosat.stratified_split(rows, seed=1)
    for label in range(len(eurosat.CLASSES)):
        c = Counter(s for (_, l), s in zip(rows, s1) if l == label)
        assert c == {"train": 14, "val": 3, "test": 3}


def test_no_path_in_two_splits(tmp_path):
    root = _make_root(tmp_path)
    out = tmp_path / "m.csv"
    eurosat.write_manifest(root, out)
    paths = [r["path"] for r in csv.DictReader(open(out))]
    assert len(paths) == len(set(paths)) == 200


def test_stats_use_train_only(tmp_path):
    root = _make_root(tmp_path)
    out = tmp_path / "m.csv"
    eurosat.write_manifest(root, out)
    st = eurosat.band_statistics(root, out)
    assert len(st["mean"]) == 13 and st["n_train_files"] == 140
    assert all(s >= 0 for s in st["std"])
