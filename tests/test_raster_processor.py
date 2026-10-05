import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from terraforge.data.raster_processor import RasterProcessor


def _da(a):
    return xr.DataArray(np.array(a, dtype="float32"), dims=("y", "x"))


def test_ndvi_values():
    out = RasterProcessor.ndvi(_da([[0.8, 0.2]]), _da([[0.1, 0.2]]))
    np.testing.assert_allclose(out.values, [[0.7 / 0.9, 0.0]], rtol=1e-5)


def test_ndvi_zero_denominator_is_nan():
    out = RasterProcessor.ndvi(_da([[0.0]]), _da([[0.0]]))
    assert np.isnan(out.values).all()


def test_cloud_mask_flags_clouds_and_shadows():
    scl = xr.DataArray(np.array([[4, 8, 9, 3, 5, 10]]), dims=("y", "x"))
    valid = RasterProcessor.cloud_mask(scl)
    assert valid.values.tolist() == [[True, False, False, False, True, False]]


def test_read_scales_reflectance_and_keeps_crs(tmp_path):
    p = tmp_path / "t.tif"
    with rasterio.open(p, "w", driver="GTiff", height=4, width=4, count=1,
                       dtype="uint16", crs="EPSG:32631",
                       transform=from_origin(500000, 5000000, 10, 10)) as dst:
        dst.write(np.full((1, 4, 4), 5000, dtype="uint16"))
    da = RasterProcessor().read(p, names=["red"])
    assert da.attrs["crs"] == "EPSG:32631"
    np.testing.assert_allclose(da.values, 0.5)
    assert float(da.x[0]) == 500005.0  # pixel centre, 10 m grid


def test_read_resample_rescales_grid(tmp_path):
    p = tmp_path / "t.tif"
    with rasterio.open(p, "w", driver="GTiff", height=2, width=2, count=1,
                       dtype="uint16", crs="EPSG:32631",
                       transform=from_origin(0, 20, 10, 10)) as dst:
        dst.write(np.full((1, 2, 2), 1000, dtype="uint16"))
    da = RasterProcessor().read(p, out_shape=(4, 4))
    assert da.shape == (1, 4, 4)
    assert float(da.x[1] - da.x[0]) == 5.0
