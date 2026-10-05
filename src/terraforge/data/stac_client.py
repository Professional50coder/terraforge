"""STAC discovery for Sentinel-2 L2A.

STAC (SpatioTemporal Asset Catalog) is a JSON standard for describing
geospatial assets. Instead of hard-coding file paths we query a catalog by
area, time and quality, and get back Items whose Assets are COG URLs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pystac_client import Client

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"


@dataclass
class SceneQuery:
    bbox: tuple[float, float, float, float]  # lon/lat: minx, miny, maxx, maxy
    start: str  # ISO date
    end: str
    max_cloud: float = 20.0
    collection: str = S2_COLLECTION
    limit: int = 50

    def validate(self) -> None:
        minx, miny, maxx, maxy = self.bbox
        if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
            raise ValueError(f"invalid bbox {self.bbox}")
        if self.start > self.end:
            raise ValueError("start must not be after end")
        if not 0 <= self.max_cloud <= 100:
            raise ValueError("max_cloud must be within 0-100")

    def to_search_kwargs(self) -> dict:
        self.validate()
        return {
            "collections": [self.collection],
            "bbox": list(self.bbox),
            "datetime": f"{self.start}/{self.end}",
            "query": {"eo:cloud_cover": {"lt": self.max_cloud}},
            "max_items": self.limit,
        }


@dataclass
class STACClient:
    url: str = EARTH_SEARCH_URL
    _client: Client | None = field(default=None, repr=False)

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client.open(self.url)
        return self._client

    def search(self, query: SceneQuery) -> list:
        """Return items sorted by cloud cover, least cloudy first."""
        items = list(self.client.search(**query.to_search_kwargs()).items())
        return sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))

    @staticmethod
    def asset_hrefs(item, names=("red", "nir", "scl")) -> dict[str, str]:
        """Map asset names to COG URLs, failing loudly if one is missing."""
        missing = [n for n in names if n not in item.assets]
        if missing:
            raise KeyError(f"{item.id} lacks assets: {missing}")
        return {n: item.assets[n].href for n in names}
