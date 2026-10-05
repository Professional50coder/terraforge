"""Fetch a small real Sentinel-2 chip around a lon/lat from cloud-optimised GeoTIFFs.

Why windowed reads: a scene is ~800 MB; a 64x64 chip at 10 m is a 640 m square. COGs
support HTTP range requests, so rasterio downloads only the needed tiles.

Known mismatches with the training data (surfaced to the caller, never hidden):
- EuroSAT is Level-1C (top-of-atmosphere); the catalog serves Level-2A (surface
  reflectance). Same sensor, different processing: a domain gap.
- L2A has no B10 (cirrus) band. It is filled with zeros and reported in `missing_bands`.
- Since processing baseline 04.00 (Jan 2022) L2A digital numbers are meant to carry a +1000
  offset. Whether a given file actually does is decided FROM THE PIXELS (see `offset_present`),
  not from metadata: for a real Paris scene the catalog declared an offset of -0.1 while the data
  contained values far below 1000 (5th percentile ~200), which under that offset would be
  negative reflectance. Trusting the metadata clipped about half the pixels to zero.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds

from terraforge.data.eurosat import BANDS

# EuroSAT band -> Earth Search (element84) sentinel-2-l2a asset key; None = not provided.
ASSET_FOR_BAND = {
    "B01": "coastal", "B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1",
    "B06": "rededge2", "B07": "rededge3", "B08": "nir", "B8A": "nir08", "B09": "nir09",
    "B10": None, "B11": "swir16", "B12": "swir22",
}
CLOUDY_SCL = (0, 1, 3, 8, 9, 10)  # no data, defective, shadow, cloud medium/high, cirrus
OFFSET_BASELINE = 4.0
OFFSET_DN = 1000


def harmonize(dn: np.ndarray, baseline: float | None) -> np.ndarray:
    """Remove the post-baseline-04.00 +1000 offset so values match older TOA scaling."""
    if baseline is not None and baseline >= OFFSET_BASELINE:
        return np.clip(dn.astype("float32") - OFFSET_DN, 0, None)
    return dn.astype("float32")


def offset_present(raw_bands: list[np.ndarray], floor: float = 800.0) -> bool:
    """True only if the +1000 offset is evidently baked into the data.

    With the offset, physically valid pixels never fall far below ~1000 DN (that would be
    negative reflectance), so the 1st percentile of non-zero pixels sits at or above ~1000.
    Without it, dark pixels (water, shadow) sit at a few hundred DN. 800 leaves margin for noise.
    """
    vals = np.concatenate([b[b > 0].ravel() for b in raw_bands]) if raw_bands else np.array([])
    return bool(vals.size and np.percentile(vals, 1) >= floor)


# GDAL HTTP tuning for cloud-optimised GeoTIFFs: skip directory listings on open, only touch .tif
# files, merge adjacent byte ranges and multiplex requests. Each open otherwise costs several
# round trips; 13 bands x several round trips was the dominant latency.
GDAL_HTTP_OPTS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
    # Latency varied from ~11 s to ~244 s for the identical request; bound every read and retry once
    # or twice, so a stalled connection fails fast instead of holding a worker for minutes.
    "GDAL_HTTP_TIMEOUT": "30",
    "GDAL_HTTP_MAX_RETRY": "2",
    "GDAL_HTTP_RETRY_DELAY": "1",
}


def read_window(href: str, lon: float, lat: float, size: int, res: float,
                resampling=Resampling.bilinear) -> np.ndarray:
    """size x size window centred on lon/lat, resampled to a `res`-metre grid."""
    with rasterio.Env(**GDAL_HTTP_OPTS), rasterio.open(href) as src:
        (x,), (y,) = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        half = size * res / 2
        win = from_bounds(x - half, y - half, x + half, y + half, transform=src.transform)
        return src.read(1, window=win, out_shape=(size, size), boundless=True,
                        fill_value=0, resampling=resampling).astype("float32")


def fetch_chip(hrefs: dict[str, str], lon: float, lat: float, baseline: float | None = None,
               size: int = 64, res: float = 10.0) -> tuple[np.ndarray, dict]:
    """Return ((13,size,size) float32 in EuroSAT band order, meta)."""
    if "scl" not in hrefs:
        raise KeyError("scene lacks the scene-classification band (scl)")
    chip = np.zeros((len(BANDS), size, size), dtype="float32")
    missing, jobs = [], {}
    for i, band in enumerate(BANDS):
        key = ASSET_FOR_BAND[band]
        if key is None or key not in hrefs:
            missing.append(band)
        else:
            jobs[i] = hrefs[key]
    # Bands are independent network reads and GDAL releases the GIL, so threads parallelise them.
    with ThreadPoolExecutor(max_workers=min(8, len(jobs) + 1)) as pool:
        scl_future = pool.submit(read_window, hrefs["scl"], lon, lat, size, res, Resampling.nearest)
        futures = {i: pool.submit(read_window, h, lon, lat, size, res) for i, h in jobs.items()}
        raw = {i: fut.result() for i, fut in futures.items()}   # a failed band raises here
        scl = scl_future.result().astype(int)
    # Remove the offset only if the baseline allows it AND the pixels show it is really there.
    removed = (baseline is None or baseline >= OFFSET_BASELINE) and offset_present(list(raw.values()))
    for i, dn in raw.items():
        chip[i] = harmonize(dn, OFFSET_BASELINE if removed else 0.0)
    cloud = float(np.isin(scl, CLOUDY_SCL).mean())
    nodata = float((scl == 0).mean())
    return chip, {"cloud_fraction": cloud, "nodata_fraction": nodata, "missing_bands": missing,
                  "offset_removed": bool(removed)}
