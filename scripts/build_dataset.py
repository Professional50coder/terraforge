"""Download EuroSAT-MS (if needed), write split manifest and train-only band stats."""
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
    a = ap.parse_args()
    data = Path(a.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    try:
        root = find_root(data)
    except FileNotFoundError:
        zpath = data / "EuroSATallBands.zip"
        print("downloading", eurosat.DOWNLOAD_URL)
        urllib.request.urlretrieve(eurosat.DOWNLOAD_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(data)
        zpath.unlink()
        root = find_root(data)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = eurosat.write_manifest(root, out / "manifest.csv", seed=a.seed)
    stats = eurosat.band_statistics(root, out / "manifest.csv")
    eurosat.save_stats(stats, out / "band_stats.json")
    print(f"{n} patches indexed; stats from {stats['n_train_files']} train files")


if __name__ == "__main__":
    main()
