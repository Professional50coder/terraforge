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


def test_cache_matches_files_and_stats_agree(tmp_path):
    root = _make_root(tmp_path / "raw") if (tmp_path / "raw").mkdir() is None else None
    man = tmp_path / "m.csv"
    eurosat.write_manifest(root, man)
    # patches in _make_root are 4x4; cache expects 64x64, so rebuild at 64
    for p in root.rglob("*.tif"):
        with rasterio.open(p, "w", driver="GTiff", height=64, width=64, count=13,
                           dtype="uint16", crs="EPSG:32631",
                           transform=from_origin(0, 640, 10, 10)) as dst:
            dst.write(np.full((13, 64, 64), 1000 * (eurosat.CLASSES.index(p.parent.name) + 1),
                              dtype="uint16"))
    n = eurosat.build_cache(root, man, tmp_path / "cache")
    assert n == 200
    arr = np.load(tmp_path / "cache" / "patches.npy", mmap_mode="r")
    labels = np.load(tmp_path / "cache" / "labels.npy")
    assert arr.shape == (200, 13, 64, 64) and arr.dtype == np.uint16
    assert int(arr[0, 0, 0, 0]) == 1000 * (int(labels[0]) + 1)
    a = eurosat.stats_from_cache(tmp_path / "cache")
    b = eurosat.band_statistics(root, man)
    np.testing.assert_allclose(a["mean"], b["mean"], rtol=1e-9)
    np.testing.assert_allclose(a["std"], b["std"], rtol=1e-6, atol=1e-6)


def test_stats_use_train_only(tmp_path):
    root = _make_root(tmp_path)
    out = tmp_path / "m.csv"
    eurosat.write_manifest(root, out)
    st = eurosat.band_statistics(root, out)
    assert len(st["mean"]) == 13 and st["n_train_files"] == 140
    assert all(s >= 0 for s in st["std"])
