"""Vector geospatial layer (GeoPandas): footprints, AOIs, areas in hectares.

Two traps this module exists to avoid:
1. Area in degrees. Computing .area on lon/lat polygons gives square degrees, which is
   meaningless. Areas are computed after projecting to a local UTM zone (metres).
2. Antimeridian/overlap sloppiness. Coverage is the footprint-AOI intersection *area*,
   not "any overlap", so a scene that clips a corner is not mistaken for full coverage.
"""
from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box, shape


def footprints_from_items(items) -> gpd.GeoDataFrame:
    """STAC items -> GeoDataFrame (EPSG:4326) with id, datetime, cloud cover."""
    rows = [{"id": i.id, "datetime": str(i.datetime), "cloud": i.properties.get("eo:cloud_cover"),
             "geometry": shape(i.geometry)} for i in items]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def area_ha(geoms: gpd.GeoSeries) -> gpd.GeoSeries:
    """Hectares, via the best local UTM projection (input must be lon/lat)."""
    if geoms.crs is None:
        raise ValueError("geometry has no CRS; refusing to guess")
    return geoms.to_crs(geoms.estimate_utm_crs()).area / 10_000.0


def coverage_of_aoi(footprints: gpd.GeoDataFrame, aoi_bbox) -> gpd.GeoDataFrame:
    """Add `coverage` = fraction of the AOI covered by each footprint, sorted best first."""
    aoi = gpd.GeoSeries([box(*aoi_bbox)], crs="EPSG:4326")
    utm = aoi.estimate_utm_crs()
    aoi_p = aoi.to_crs(utm).iloc[0]
    fp = footprints.to_crs(utm)
    out = footprints.copy()
    out["coverage"] = [g.intersection(aoi_p).area / aoi_p.area for g in fp.geometry]
    return out.sort_values("coverage", ascending=False).reset_index(drop=True)
