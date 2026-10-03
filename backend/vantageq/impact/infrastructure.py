"""Road / bridge / building impact by plain geometric intersection with the flood polygon.

All geometries must be in a metric CRS (UTM). Rules (docs/TRISHULI_IMPACT_ANALYSIS.md):
- road segment (OSM way clipped to the AOI) is *potentially affected* if its length inside the
  flood polygon is >= MIN_AFFECTED_LENGTH_M;
- bridge (highway way with bridge=*) *intersects predicted flood* if any of its length
  (> MIN_OVERLAP) lies inside the flood polygon; `near_flood` records whether it lies within NEAR_M;
- building is *potentially affected* if part of its footprint (area > MIN_OVERLAP m²) lies inside the
  flood polygon, or, for a point building, if the point is covered by it.
Mere edge contact or floating-point slivers do not count (MIN_OVERLAP tolerance).
Each OSM element is counted once (keyed by type + id).
"""

from __future__ import annotations

from shapely.prepared import prep

MIN_AFFECTED_LENGTH_M = 10.0  # about one 10 m pixel; ignores corner touches
NEAR_M = 20.0
MIN_OVERLAP = 0.01  # m / m²: numerical tolerance, excludes touching edges and float slivers


def line_impact(line, flood) -> tuple[float, float]:
    """(total length, length inside flood) in metres."""
    if line.is_empty:
        return 0.0, 0.0
    inside = line.intersection(flood).length if line.intersects(flood) else 0.0
    return float(line.length), float(inside)


def _unique(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for it in items:
        key = (it["osm_type"], it["osm_id"])
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def road_impacts(roads: list[dict], flood, aoi, min_len: float = MIN_AFFECTED_LENGTH_M) -> list[dict]:
    out = []
    for r in _unique(roads):
        geom = r["geometry"].intersection(aoi)
        length, inside = line_impact(geom, flood)
        if length == 0:
            continue
        out.append(r | {"geometry": geom, "length_m": round(length, 1), "affected_length_m": round(inside, 1),
                        "affected_pct": round(100 * inside / length, 1), "affected": inside >= min_len})
    return out


def bridge_impacts(bridges: list[dict], flood, aoi, near_m: float = NEAR_M) -> list[dict]:
    out = []
    for b in _unique(bridges):
        geom = b["geometry"].intersection(aoi)
        if geom.is_empty:
            continue
        length, inside = line_impact(geom, flood)
        out.append(b | {"geometry": geom, "length_m": round(length, 1), "affected_length_m": round(inside, 1),
                        "affected_pct": round(100 * inside / length, 1) if length else None,
                        "affected": inside > MIN_OVERLAP,
                        "near_flood": bool(geom.distance(flood) <= near_m) if not flood.is_empty else False})
    return out


def building_impacts(buildings: list[dict], flood, aoi) -> list[dict]:
    pf, pa = (prep(flood) if not flood.is_empty else None), prep(aoi)
    out = []
    for b in _unique(buildings):
        g = b["geometry"]
        if not pa.intersects(g):
            continue
        touch = bool(pf.intersects(g)) if pf else False
        if g.geom_type in ("Polygon", "MultiPolygon"):
            area = float(g.area)
            inside = float(g.intersection(flood).area) if touch else 0.0
            hit = inside > MIN_OVERLAP
        else:
            area, inside = 0.0, 0.0
            hit = bool(flood.covers(g)) if touch else False
        out.append(b | {"footprint_m2": round(area, 1), "flooded_m2": round(inside, 1), "affected": hit})
    return out
