from datetime import datetime, timezone

import geopandas as gpd
import pytest
from shapely.geometry import box

from terraforge.geo import aoi, sky


def utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def test_declination_at_equinox_and_solstices():
    assert abs(sky.sun_position(utc(2024, 3, 20, 3, 6))["lat"]) < 0.4
    assert abs(sky.sun_position(utc(2024, 6, 20, 20, 51))["lat"] - 23.44) < 0.3
    assert abs(sky.sun_position(utc(2024, 12, 21, 9, 21))["lat"] + 23.44) < 0.3


def test_subsolar_longitude_tracks_utc_time():
    noon = sky.sun_position(utc(2024, 3, 20, 12, 0))["lon"]
    six = sky.sun_position(utc(2024, 3, 20, 18, 0))["lon"]
    assert abs(noon) < 5  # within the equation of time (+-4 deg) of the prime meridian
    assert abs((noon - six) - 90) < 0.1  # 6 h = 90 deg westward (equation of time drifts ~0.02)


def test_daylight_matches_geometry():
    t = utc(2024, 3, 20, 12, 0)  # sun over the Gulf of Guinea
    assert sky.is_daylight(0, 0, t) and not sky.is_daylight(0, 180, t)
    assert sky.is_daylight(80, 0, utc(2024, 6, 21, 12, 0))   # midnight sun
    assert not sky.is_daylight(80, 0, utc(2024, 12, 21, 12, 0))  # polar night


def test_terminator_is_90_degrees_from_sun():
    t = utc(2024, 9, 1, 8, 0)
    s = sky.sun_position(t)
    import math
    for lon, lat in sky.terminator(t, 36):
        la, lo, sl, so = map(math.radians, (lat, lon, s["lat"], s["lon"]))
        cosd = math.sin(la) * math.sin(sl) + math.cos(la) * math.cos(sl) * math.cos(lo - so)
        assert abs(cosd) < 1e-6  # angular distance = 90 deg


def test_moon_phase_known_dates():
    assert sky.moon_phase(utc(2024, 1, 25, 17, 54))["illumination"] > 0.98   # full moon
    assert sky.moon_phase(utc(2024, 1, 11, 11, 57))["illumination"] < 0.03   # new moon
    q = sky.moon_phase(utc(2024, 1, 18, 3, 52))                               # first quarter
    assert 0.4 < q["illumination"] < 0.6 and q["phase"] == "First Quarter"


def test_area_in_hectares_not_square_degrees():
    # ~1 km x 1 km at mid-latitude -> ~100 ha
    g = gpd.GeoSeries([box(6.0, 49.6, 6.0 + 0.01368, 49.6 + 0.009)], crs="EPSG:4326")
    assert 95 < float(aoi.area_ha(g).iloc[0]) < 105


def test_area_refuses_missing_crs():
    with pytest.raises(ValueError):
        aoi.area_ha(gpd.GeoSeries([box(0, 0, 1, 1)]))


def test_coverage_is_area_fraction_and_sorted():
    fp = gpd.GeoDataFrame({"id": ["half", "full", "none"]}, geometry=[
        box(6.0, 49.0, 6.5, 50.0), box(5.0, 48.0, 8.0, 51.0), box(20, 20, 21, 21)],
        crs="EPSG:4326")
    out = aoi.coverage_of_aoi(fp, (6.0, 49.0, 7.0, 50.0))
    assert out["id"].tolist() == ["full", "half", "none"]
    assert abs(out["coverage"].iloc[0] - 1.0) < 1e-3
    assert abs(out["coverage"].iloc[1] - 0.5) < 0.01 and out["coverage"].iloc[2] == 0
