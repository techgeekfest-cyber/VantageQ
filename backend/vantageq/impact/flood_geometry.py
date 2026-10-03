"""Flood class raster -> vector flood geometry.

Method: pixels of class 2 (flood) in the model's class map form a binary mask; connected regions
smaller than `min_pixels` (8-connectivity) are removed with rasterio's sieve (this also fills holes
smaller than that); the mask is polygonised along pixel edges and dissolved into one
(Multi)Polygon. No further simplification, so the polygon follows the 10 m pixel grid exactly.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.features import shapes, sieve
from shapely.geometry import shape
from shapely.ops import unary_union

FLOOD_CLASS = 2
MIN_PIXELS = 10  # ~0.08 ha at 8.8 m ground pixels


def flood_mask(class_map: np.ndarray, flood_class: int = FLOOD_CLASS, min_pixels: int = MIN_PIXELS) -> np.ndarray:
    mask = (class_map == flood_class).astype("uint8")
    if min_pixels > 1 and mask.any():
        mask = sieve(mask, size=min_pixels, connectivity=8)
    return mask.astype(bool)


def mask_to_polygon(mask: np.ndarray, transform):
    """Dissolved (Multi)Polygon of True pixels, in the raster's CRS."""
    polys = [shape(g) for g, v in shapes(mask.astype("uint8"), mask=mask, transform=transform, connectivity=8) if v]
    return unary_union(polys)


def flood_polygon_from_raster(path, flood_class: int = FLOOD_CLASS, min_pixels: int = MIN_PIXELS):
    """Return (polygon in raster CRS, raster EPSG, raw flood pixel count, kept flood pixel count)."""
    with rasterio.open(path) as ds:
        cls = ds.read(1)
        transform, epsg = ds.transform, ds.crs.to_epsg()
    mask = flood_mask(cls, flood_class, min_pixels)
    return mask_to_polygon(mask, transform), epsg, int((cls == flood_class).sum()), int(mask.sum())
