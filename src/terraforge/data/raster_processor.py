"""Raster I/O and spectral processing for Sentinel-2 L2A.

Design notes
------------
Sentinel-2 L2A bands are surface reflectance stored as scaled integers
(reflectance = DN / 10000). Bands come at 10/20/60 m, so anything that mixes
them must resample onto a common grid first. Every function here keeps the
CRS and affine transform attached to the data (as xarray attrs) so geospatial
meaning is never lost between steps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling

REFLECTANCE_SCALE = 10_000.0

# Sentinel-2 Scene Classification Layer (SCL) classes treated as unusable:
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8 cloud medium prob,
# 9 cloud high prob, 10 thin cirrus.
INVALID_SCL = (0, 1, 3, 8, 9, 10)


class RasterProcessor:
    """Reads GeoTIFFs into xarray and computes spectral products."""

    def __init__(self, scale: float = REFLECTANCE_SCALE):
        self.scale = scale

    def read(self, path: str | Path, names: list[str] | None = None,
             out_shape: tuple[int, int] | None = None) -> xr.DataArray:
        """Read a GeoTIFF as a (band, y, x) float32 reflectance DataArray.

        out_shape resamples on read (bilinear) so 20 m bands can be brought
        onto the 10 m grid. The affine transform is rescaled accordingly.
        """
        with rasterio.open(path) as src:
            shape = out_shape or (src.height, src.width)
            data = src.read(
                out_shape=(src.count, *shape), resampling=Resampling.bilinear
            ).astype("float32")
            transform = src.transform * src.transform.scale(
                src.width / shape[1], src.height / shape[0]
            )
            crs = src.crs
            nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        data = data / self.scale
        ys = transform.f + transform.e * (np.arange(shape[0]) + 0.5)
        xs = transform.c + transform.a * (np.arange(shape[1]) + 0.5)
        return xr.DataArray(
            data, dims=("band", "y", "x"),
            coords={"band": names or list(range(1, data.shape[0] + 1)), "y": ys, "x": xs},
            attrs={"crs": crs.to_string() if crs else None,
                   "transform": tuple(transform)[:6]},
        )

    @staticmethod
    def ndvi(nir: xr.DataArray, red: xr.DataArray) -> xr.DataArray:
        """NDVI = (NIR - Red) / (NIR + Red), in [-1, 1].

        Healthy vegetation reflects strongly in NIR and absorbs red (chlorophyll),
        so it scores high (~0.6-0.9); bare soil ~0.1-0.2; water/cloud <= 0.
        Zero denominators give NaN rather than inf.
        """
        denom = nir + red
        return ((nir - red) / denom.where(denom != 0)).rename("ndvi")

    @staticmethod
    def cloud_mask(scl: xr.DataArray) -> xr.DataArray:
        """True where the pixel is usable (clear), from the SCL band."""
        return ~xr.apply_ufunc(np.isin, scl, INVALID_SCL)

    @staticmethod
    def apply_mask(data: xr.DataArray, valid: xr.DataArray) -> xr.DataArray:
        """Set invalid pixels to NaN; keeps attrs."""
        return data.where(valid)

    @staticmethod
    def zscore(data: xr.DataArray, dim=("y", "x")) -> xr.DataArray:
        """Per-band standardisation, ignoring NaNs (for model input)."""
        return (data - data.mean(dim, skipna=True)) / data.std(dim, skipna=True)
