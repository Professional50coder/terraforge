"""Tools the agent may call. Every tool returns plain JSON-serialisable facts.

The agent is only allowed to state what a tool returned, so each tool validates its own
inputs and raises ValueError with a message the model can read and correct.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from terraforge.geo import sky


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema (object)
    fn: Callable[..., dict]

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}

    def validate(self, args: dict) -> dict:
        props = self.parameters.get("properties", {})
        for req in self.parameters.get("required", []):
            if req not in args:
                raise ValueError(f"missing required argument '{req}'")
        clean = {}
        for k, v in args.items():
            if k not in props:
                raise ValueError(f"unknown argument '{k}'")
            t = props[k].get("type")
            if t == "number" and (isinstance(v, bool) or not isinstance(v, (int, float))):
                raise ValueError(f"argument '{k}' must be a number")
            if t == "string" and not isinstance(v, str):
                raise ValueError(f"argument '{k}' must be a string")
            if t == "array" and not isinstance(v, list):
                raise ValueError(f"argument '{k}' must be an array")
            clean[k] = v
        return clean


def _parse_time(t: str | None):
    if not t:
        return None
    try:
        return datetime.fromisoformat(t.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("time must be ISO-8601, e.g. 2024-06-21T12:00:00Z")


def sky_now(time: str | None = None) -> dict:
    dt = _parse_time(time)
    return {"sun": sky.sun_position(dt), "moon": sky.moon_phase(dt)}


def daylight_at(lat: float, lon: float, time: str | None = None) -> dict:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("lat must be in [-90,90] and lon in [-180,180]")
    return {"lat": lat, "lon": lon, "daylight": sky.is_daylight(lat, lon, _parse_time(time))}


def bbox_area(bbox: list) -> dict:
    import geopandas as gpd
    from shapely.geometry import box

    from terraforge.geo.aoi import area_ha
    if len(bbox) != 4 or not all(isinstance(v, (int, float)) for v in bbox):
        raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat]")
    if not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        raise ValueError("invalid bbox")
    g = gpd.GeoSeries([box(*bbox)], crs="EPSG:4326")
    return {"bbox": bbox, "area_ha": float(area_ha(g).iloc[0])}


def build_tools(stac_client=None) -> dict[str, Tool]:
    """stac_client is injectable so tests and offline use never touch the network."""
    def search_scenes(bbox: list, start: str, end: str, max_cloud: float = 20) -> dict:
        from terraforge.data.stac_client import SceneQuery, STACClient
        client = stac_client or STACClient()
        items = client.search(SceneQuery(tuple(bbox), start, end, max_cloud, limit=10))
        return {"count": len(items), "scenes": [
            {"id": i.id, "date": str(i.datetime)[:10],
             "cloud_pct": i.properties.get("eo:cloud_cover")} for i in items[:5]]}

    num = {"type": "number"}
    tools = [
        Tool("sky_now", "Sun subpoint and moon phase, now or at an ISO time.",
             {"type": "object", "properties": {"time": {"type": "string"}}, "required": []}, sky_now),
        Tool("daylight_at", "Whether it is daylight at a latitude/longitude.",
             {"type": "object", "properties": {"lat": num, "lon": num, "time": {"type": "string"}},
              "required": ["lat", "lon"]}, daylight_at),
        Tool("bbox_area", "Area in hectares of a lon/lat bounding box.",
             {"type": "object", "properties": {"bbox": {"type": "array"}}, "required": ["bbox"]},
             bbox_area),
        Tool("search_scenes", "Find Sentinel-2 scenes for a bbox and date range, least cloudy first.",
             {"type": "object", "properties": {"bbox": {"type": "array"}, "start": {"type": "string"},
                                               "end": {"type": "string"}, "max_cloud": num},
              "required": ["bbox", "start", "end"]}, search_scenes),
    ]
    return {t.name: t for t in tools}
