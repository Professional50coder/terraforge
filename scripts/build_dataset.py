"""Download EuroSAT-MS (if needed), then write manifest, uint16 cache and train-only stats."""
import argparse
import urllib.request
import zipfile
from pathlib import Path

from terraforge.data import eurosat


def find_root(base: Path) -> Path:
    for p in [base, *base.rglob("*")]:
        if p.is_dir() and (p / eurosat.CLASSES[0]).is_dir():
            return p
    raise FileNotFoundError(f"EuroSAT class folders not found under {base}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-raw", action="store_true", help="keep extracted tifs after caching")
    a = ap.parse_args()
    data = Path(a.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    try:
        root = find_root(data)
    except FileNotFoundError:
        zpath = data / "EuroSATallBands.zip"
        for url in eurosat.DOWNLOAD_URLS:
            try:
                print("downloading", url, flush=True)
                urllib.request.urlretrieve(url, zpath)
                break
            except OSError as e:
                print("failed:", e, flush=True)
        else:
            raise SystemExit("all EuroSAT mirrors failed")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(data)
        zpath.unlink()
        root = find_root(data)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = eurosat.write_manifest(root, out / "manifest.csv", seed=a.seed)
    print(f"{n} patches indexed", flush=True)
    if not (out / "cache" / "splits.npy").exists():
        eurosat.build_cache(root, out / "manifest.csv", out / "cache")
    stats = eurosat.stats_from_cache(out / "cache")
    eurosat.save_stats(stats, out / "band_stats.json")
    print(f"cache ready; stats from {stats['n_train_files']} train patches", flush=True)


if __name__ == "__main__":
    main()
