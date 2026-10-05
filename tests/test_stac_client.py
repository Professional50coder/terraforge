import pytest

from terraforge.data.stac_client import STACClient, SceneQuery


def test_query_kwargs():
    q = SceneQuery((5.9, 49.4, 6.5, 50.2), "2024-06-01", "2024-08-31", max_cloud=10)
    kw = q.to_search_kwargs()
    assert kw["datetime"] == "2024-06-01/2024-08-31"
    assert kw["query"] == {"eo:cloud_cover": {"lt": 10}}


@pytest.mark.parametrize("bbox,start,end,cloud", [
    ((6, 49, 5, 50), "2024-01-01", "2024-02-01", 10),
    ((5, 49, 6, 50), "2024-03-01", "2024-02-01", 10),
    ((5, 49, 6, 50), "2024-01-01", "2024-02-01", 120),
])
def test_invalid_queries(bbox, start, end, cloud):
    with pytest.raises(ValueError):
        SceneQuery(bbox, start, end, cloud).to_search_kwargs()


def test_missing_asset_fails_loudly():
    class Item:
        id = "x"
        assets = {"red": type("A", (), {"href": "r"})()}

    with pytest.raises(KeyError):
        STACClient.asset_hrefs(Item())
