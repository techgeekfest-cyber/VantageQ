"""Pre-event OSM retrieval via an Overpass API attic query (historical snapshot), and parsing.

The snapshot date is passed as Overpass `[date:...]`, so the result reflects OSM as it was at
that instant; post-event edits are excluded. (ohsome extraction was the planned source, but it now
requires an API key; see docs/TRISHULI_IMPACT_ANALYSIS.md.)
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "VantageQ-hackathon/0.1 (educational research prototype)"

# Roads used for impact and the connectivity graph (vehicle-usable classes).
VEHICLE_HIGHWAYS = frozenset({
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
    "living_street", "service", "track", "road",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"})
# Foot paths: reported and used for bridges, but not part of the connectivity graph.
PATH_HIGHWAYS = frozenset({"path", "footway", "steps", "bridleway", "pedestrian", "cycleway"})
SETTLEMENT_PLACES = ("city", "town", "village", "hamlet", "isolated_dwelling")


def build_query(bbox_wgs84, date_iso: str, timeout: int = 180) -> str:
    w, s, e, n = bbox_wgs84
    places = "|".join(SETTLEMENT_PLACES)
    return (f'[out:json][timeout:{timeout}][date:"{date_iso}"][bbox:{s},{w},{n},{e}];\n'
            f'(way["highway"];nwr["building"]["building"!="no"];nwr["place"~"^({places})$"];);\n'
            "out body geom;")


def fetch(bbox_wgs84, date_iso: str, cache: Path) -> dict:
    """Run the attic query once and cache the raw JSON (data/external is git-ignored)."""
    if cache.exists():
        return json.loads(cache.read_text())
    data = urllib.parse.urlencode({"data": build_query(bbox_wgs84, date_iso)}).encode()
    req = urllib.request.Request(OVERPASS_URL, data=data, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = json.load(resp)
    raw["vantageq_query"] = {"bbox_wgs84": list(bbox_wgs84), "date": date_iso, "endpoint": OVERPASS_URL}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(raw))
    return raw


def _tags(e: dict) -> dict:
    return e.get("tags", {})


def _name(tags: dict) -> str | None:
    return tags.get("name:en") or tags.get("name")


def _way_line(e: dict) -> tuple[list, list]:
    coords = [(p["lon"], p["lat"]) for p in e.get("geometry") or [] if p]
    return e.get("nodes", []), coords


def _relation_polygon(e: dict):
    def lines(role):
        out = []
        for m in e.get("members", []):
            if m.get("type") == "way" and m.get("role", "outer") == role and m.get("geometry"):
                pts = [(p["lon"], p["lat"]) for p in m["geometry"] if p]
                if len(pts) >= 2:
                    out.append(LineString(pts))
        return out

    outer = unary_union(list(polygonize(lines("outer"))))
    inner = unary_union(list(polygonize(lines("inner"))))
    return outer.difference(inner) if not inner.is_empty else outer


def parse(raw: dict) -> dict[str, list[dict]]:
    """Split raw Overpass elements into roads, paths, bridges, buildings, settlements (WGS84).

    Each OSM element appears at most once per list (keyed by type + id).
    """
    out = {"roads": [], "paths": [], "bridges": [], "buildings": [], "settlements": []}
    seen: dict[str, set] = {k: set() for k in out}

    def add(kind, key, item):
        if key not in seen[kind]:
            seen[kind].add(key)
            out[kind].append(item)

    for e in raw.get("elements", []):
        t, tags, key = e["type"], _tags(e), (e["type"], e["id"])
        hw = tags.get("highway")
        if t == "way" and hw:
            nodes, coords = _way_line(e)
            if len(coords) < 2:
                continue
            item = {"osm_type": t, "osm_id": e["id"], "highway": hw, "name": _name(tags), "ref": tags.get("ref"),
                    "bridge": tags.get("bridge"), "nodes": nodes, "geometry": LineString(coords)}
            if hw in VEHICLE_HIGHWAYS:
                add("roads", key, item)
            elif hw in PATH_HIGHWAYS:
                add("paths", key, item)
            if tags.get("bridge") not in (None, "no"):
                add("bridges", key, item)
        if "building" in tags and tags["building"] != "no":
            geom = None
            if t == "node":
                geom = Point(e["lon"], e["lat"])
            elif t == "way":
                _, coords = _way_line(e)
                if len(coords) >= 4 and coords[0] == coords[-1]:
                    geom = Polygon(coords)
            elif t == "relation":
                geom = _relation_polygon(e)
            if geom is not None and not geom.is_empty:
                add("buildings", key, {"osm_type": t, "osm_id": e["id"], "building": tags["building"],
                                       "geometry": shapely.make_valid(geom)})
        if tags.get("place") in SETTLEMENT_PLACES:
            if t == "node":
                geom = Point(e["lon"], e["lat"])
            else:
                _, coords = _way_line(e)
                geom = (Polygon(coords) if len(coords) >= 4 and coords[0] == coords[-1]
                        else _relation_polygon(e) if t == "relation" else None)
                geom = geom.representative_point() if geom is not None and not geom.is_empty else None
            if geom is not None:
                add("settlements", key, {"osm_type": t, "osm_id": e["id"], "place": tags["place"],
                                         "name": _name(tags), "name_local": tags.get("name"), "geometry": geom})
    return out
