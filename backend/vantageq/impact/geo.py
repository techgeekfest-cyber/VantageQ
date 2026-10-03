"""Small geometry helpers: reprojection and GeoJSON writing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shapely
from rasterio.warp import transform as warp_transform

WGS84 = 4326
UTM45N = 32645  # metric CRS for lengths/areas at Trishuli


def reproject(geom, src_epsg: int, dst_epsg: int):
    """Reproject a shapely geometry (or array of geometries) between EPSG codes."""
    if src_epsg == dst_epsg:
        return geom

    def fn(coords: np.ndarray) -> np.ndarray:
        if len(coords) == 0:
            return coords
        xs, ys = warp_transform(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", coords[:, 0], coords[:, 1])
        return np.column_stack([xs, ys])

    return shapely.transform(geom, fn)


def write_geojson(path: Path, features: list[tuple], src_epsg: int = UTM45N) -> None:
    """Write (geometry, properties) pairs as an RFC 7946 FeatureCollection in WGS84."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for geom, props in features:
        g = reproject(geom, src_epsg, WGS84)
        out.append({"type": "Feature", "geometry": shapely.geometry.mapping(g), "properties": props})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": out}, ensure_ascii=False))
