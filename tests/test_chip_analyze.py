import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from rasterio.warp import transform as warp_transform  # noqa: E402

from terraforge.api.main import build_reference_index, create_app  # noqa: E402
from terraforge.api.sources import SyntheticSource  # noqa: E402
from terraforge.data.chip_fetcher import ASSET_FOR_BAND, fetch_chip, harmonize  # noqa: E402
from terraforge.inference.predictor import InferenceEngine  # noqa: E402

LON, LAT = 6.1, 49.6


def make_scene(tmp_path, dn=3000, scl_value=4):
    """Synthetic L2A-like scene: 10 m UTM rasters covering the point with margin."""
    (x,), (y,) = warp_transform("EPSG:4326", "EPSG:32631", [LON], [LAT])
    hrefs = {}
    for key in [k for k in ASSET_FOR_BAND.values() if k] + ["scl"]:
        p = tmp_path / f"{key}.tif"
        val = scl_value if key == "scl" else dn
        with rasterio.open(p, "w", driver="GTiff", height=400, width=400, count=1, dtype="uint16",
                           crs="EPSG:32631", transform=from_origin(x - 2000, y + 2000, 10, 10)) as d:
            d.write(np.full((1, 400, 400), val, dtype="uint16"))
        hrefs[key] = str(p)
    return hrefs


def test_harmonize_removes_offset_only_for_new_baseline():
    dn = np.array([500.0, 3000.0])
    assert harmonize(dn, 5.0).tolist() == [0.0, 2000.0]     # clipped at 0, offset removed
    assert harmonize(dn, 3.0).tolist() == [500.0, 3000.0]   # older baseline untouched
    assert harmonize(dn, None).tolist() == [500.0, 3000.0]


def test_fetch_chip_shape_band_order_missing_b10_and_cloud(tmp_path):
    chip, meta = fetch_chip(make_scene(tmp_path), LON, LAT, baseline=5.0)
    assert chip.shape == (13, 64, 64)
    assert meta["missing_bands"] == ["B10"] and not chip[10].any()   # B10 zero-filled
    assert np.allclose(chip[3], 2000.0)                              # 3000 - 1000 offset
    assert meta["cloud_fraction"] == 0.0


def test_cloud_fraction_detected(tmp_path):
    _, meta = fetch_chip(make_scene(tmp_path, scl_value=9), LON, LAT)
    assert meta["cloud_fraction"] == 1.0


def test_missing_scl_fails_loudly(tmp_path):
    h = make_scene(tmp_path)
    h.pop("scl")
    with pytest.raises(KeyError):
        fetch_chip(h, LON, LAT)


class Item:
    id, datetime = "S2_test", "2024-06-01T10:00:00Z"
    properties = {"eo:cloud_cover": 2.0, "s2:processing_baseline": "05.10"}

    def __init__(self, hrefs):
        self.assets = {k: type("A", (), {"href": v})() for k, v in hrefs.items()}


class FakeSTAC:
    def __init__(self, items):
        self.items = items

    def search(self, q):
        return self.items


def client(tmp_path, items):
    stats = {"mean": [1500.0] * 13, "std": [800.0] * 13}
    eng = InferenceEngine.load("cnn", None, stats)
    src = SyntheticSource()
    return TestClient(create_app(eng, src, build_reference_index(eng, src, 32), stac=FakeSTAC(items)))


def test_analyze_returns_prediction_with_honest_warnings(tmp_path):
    c = client(tmp_path, [Item(make_scene(tmp_path))])
    r = c.get("/analyze", params={"lat": LAT, "lon": LON})
    assert r.status_code == 200
    j = r.json()
    assert j["scene"]["id"] == "S2_test" and 0 <= j["confidence"] <= 1
    text = " ".join(j["warnings"])
    assert "Level-1C" in text and "B10" in text and "not trained" in text


def test_conformal_sets_flow_through_predict_and_analyze(tmp_path):
    stats = {"mean": [1500.0] * 13, "std": [800.0] * 13}
    # q = 1.0 admits every class: a deliberately maximal set, so the plumbing is checkable
    eng = InferenceEngine.load("cnn", None, stats,
                               conformal={"method": "lac", "q": 1.0, "alpha": 0.1})
    src = SyntheticSource()
    c = TestClient(create_app(eng, src, build_reference_index(eng, src, 16),
                              stac=FakeSTAC([Item(make_scene(tmp_path))])))
    j = c.get("/analyze", params={"lat": LAT, "lon": LON}).json()
    assert len(j["prediction_set"]) == 10 and j["label"] == j["prediction_set"][0]
    assert any("at least 90%" in w and "does not hold under" in w for w in j["warnings"])
    import base64 as b64
    x, _ = src.next_batch(2)
    body = {"shape": list(x.shape), "data_b64": b64.b64encode(x.astype("<f4").tobytes()).decode()}
    r = c.post("/predict", json=body).json()
    assert r["conformal_alpha"] == 0.1 and len(r["predictions"][0]["prediction_set"]) == 10


def test_analyze_no_scene_is_404_and_bad_coords_422(tmp_path):
    c = client(tmp_path, [])
    assert c.get("/analyze", params={"lat": LAT, "lon": LON}).status_code == 404
    assert c.get("/analyze", params={"lat": 120, "lon": 0}).status_code == 422


def test_analyze_upstream_failure_is_502_not_500(tmp_path):
    hrefs = make_scene(tmp_path)
    hrefs["red"] = str(tmp_path / "does_not_exist.tif")
    c = client(tmp_path, [Item(hrefs)])
    assert c.get("/analyze", params={"lat": LAT, "lon": LON}).status_code == 502


def test_offset_detected_from_pixels_not_trusted_from_metadata(tmp_path):
    from terraforge.data.chip_fetcher import offset_present
    rng = np.random.default_rng(0)
    with_offset = [1000 + rng.gamma(2, 400, 5000)]            # floor at ~1000: offset is baked in
    without = [rng.gamma(2, 400, 5000)]                        # dark pixels at a few hundred DN
    assert offset_present(with_offset) and not offset_present(without)
    assert not offset_present([]) and not offset_present([np.zeros(10)])


def test_fetch_chip_keeps_dark_data_intact_when_the_declared_offset_is_absent(tmp_path):
    # baseline says "offset present", but values (500 DN) prove it is not: must NOT subtract 1000
    chip, meta = fetch_chip(make_scene(tmp_path, dn=500), LON, LAT, baseline=5.12)
    assert meta["offset_removed"] is False and np.allclose(chip[3], 500.0)


def test_fetch_chip_removes_a_genuine_offset_even_without_a_baseline(tmp_path):
    chip, meta = fetch_chip(make_scene(tmp_path, dn=3000), LON, LAT, baseline=None)
    assert meta["offset_removed"] is True and np.allclose(chip[3], 2000.0)


def test_slow_imagery_returns_504_not_a_hung_request(tmp_path):
    import time

    class SlowSTAC:
        def search(self, q):
            time.sleep(1.0)
            return [Item(make_scene(tmp_path))]

    stats = {"mean": [1500.0] * 13, "std": [800.0] * 13}
    eng = InferenceEngine.load("cnn", None, stats)
    src = SyntheticSource()
    c = TestClient(create_app(eng, src, build_reference_index(eng, src, 16),
                              stac=SlowSTAC(), analyze_timeout=0.2))
    r = c.get("/analyze", params={"lat": LAT, "lon": LON})
    assert r.status_code == 504 and "slowly" in r.json()["detail"]


def test_repeat_click_reuses_the_cached_chip(tmp_path, monkeypatch):
    from terraforge.data import chip_fetcher
    calls = {"n": 0}
    real = chip_fetcher.fetch_chip

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(chip_fetcher, "fetch_chip", counting)
    c = client(tmp_path, [Item(make_scene(tmp_path))])
    for _ in range(3):
        assert c.get("/analyze", params={"lat": LAT, "lon": LON}).status_code == 200
    assert calls["n"] == 1                       # three clicks, one network fetch
    c.get("/analyze", params={"lat": LAT + 0.01, "lon": LON})
    assert calls["n"] == 2                       # a different place fetches again
