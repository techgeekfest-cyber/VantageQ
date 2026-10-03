"""Build the committed, lightweight dashboard demo snapshot from the (git-ignored) Trishuli outputs.

Reads data/processed/trishuli/ (+ infrastructure/) and writes frontend/public/demo/trishuli/:
- summary JSONs copied verbatim (all numbers shown in the dashboard come from these);
- GeoJSON layers with coordinates rounded to 6 decimals (~0.1 m, visualisation only);
- roads_reference.geojson: all roads simplified (2 m tolerance, visualisation only);
- affected_road_sections.geojson: the parts of affected roads inside the predicted flood
  (visualisation of the same overlap used in the analysis; no new analysis);
- manifest.json: event/scene/model metadata and SHA-256 of every source file.
Analytical source files are never modified. Output is deterministic for the same inputs.

Usage:
  python scripts/build_demo_snapshot.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from shapely.geometry import mapping, shape

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from vantageq.impact.geo import UTM45N, WGS84, reproject  # noqa: E402

SRC = REPO_ROOT / "data" / "processed" / "trishuli"
INFRA = SRC / "infrastructure"
OUT = REPO_ROOT / "frontend" / "public" / "demo" / "trishuli"
DECIMALS = 6
SIMPLIFY_M = 2.0

LAYERS = ["flood_extent", "affected_roads", "affected_bridges", "bridges", "affected_buildings",
          "settlements", "cutoff_settlements"]
EVENT = {"aoi_name": "Trishuli Valley (sub-AOI near Betrawati), Nepal", "event_date": "2026-08-26"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rounded(geom: dict) -> dict:
    def r(c):
        return [round(c[0], DECIMALS), round(c[1], DECIMALS)] if isinstance(c[0], (int, float)) else [r(x) for x in c]
    return {"type": geom["type"], "coordinates": r(geom["coordinates"])}


def write_json(path: Path, obj) -> int:
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return len(text) + 1


def fc(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def load(name: str) -> dict:
    return json.loads((INFRA / f"{name}.geojson").read_text())


def main() -> int:
    needed = [SRC / "trishuli_20260828_summary.json", INFRA / "infrastructure_summary.json",
              INFRA / "connectivity_summary.json", INFRA / "roads.geojson"] + [INFRA / f"{n}.geojson" for n in LAYERS]
    missing = [str(p.relative_to(REPO_ROOT)) for p in needed if not p.exists()]
    if missing:
        print("missing inputs (run scripts/infer_trishuli.py and scripts/analyze_trishuli_impact.py):", *missing, sep="\n  ")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    sizes = {}

    for name in LAYERS:
        src = load(name)
        sizes[f"{name}.geojson"] = write_json(OUT / f"{name}.geojson", fc(
            [{"type": "Feature", "geometry": rounded(f["geometry"]), "properties": f["properties"]}
             for f in src["features"]]))

    roads = load("roads")
    ref = []
    for f in roads["features"]:
        g = reproject(shape(f["geometry"]), WGS84, UTM45N).simplify(SIMPLIFY_M, preserve_topology=True)
        p = f["properties"]
        ref.append({"type": "Feature", "geometry": rounded(mapping(reproject(g, UTM45N, WGS84))),
                    "properties": {k: p.get(k) for k in ("osm_id", "highway", "name", "ref")}})
    sizes["roads_reference.geojson"] = write_json(OUT / "roads_reference.geojson", fc(ref))

    flood = reproject(shape(load("flood_extent")["features"][0]["geometry"]), WGS84, UTM45N)
    sections = []
    for f in load("affected_roads")["features"]:
        part = reproject(shape(f["geometry"]), WGS84, UTM45N).intersection(flood)
        if not part.is_empty:
            sections.append({"type": "Feature", "geometry": rounded(mapping(reproject(part, UTM45N, WGS84))),
                             "properties": f["properties"]})
    sizes["affected_road_sections.geojson"] = write_json(OUT / "affected_road_sections.geojson", fc(sections))

    for name in ("infrastructure_summary", "connectivity_summary"):
        sizes[f"{name}.json"] = write_json(OUT / f"{name}.json", json.loads((INFRA / f"{name}.json").read_text()))

    inf = json.loads((SRC / "trishuli_20260828_summary.json").read_text())
    manifest = {
        "title": "VantageQ demo snapshot: Trishuli flood, August 2026",
        "aoi_name": EVENT["aoi_name"], "aoi_wgs84": inf["sub_aoi_wgs84"], "event_date": EVENT["event_date"],
        "scenes": inf["scenes"],
        "model": {"name": "Kuro Siwo-trained U-Net (ResNet-18 encoder)", "input": "6-channel Sentinel-1 VV/VH: post, pre1, pre2",
                  "checkpoint_sha256_16": inf["checkpoint_sha256_16"], "checkpoint_epoch": inf["checkpoint_epoch"]},
        "flood_prediction": {"class_pct": inf["class_pct"], "grid": {k: inf["grid"][k] for k in ("epsg", "width", "height")}},
        "visualisation_notes": [
            f"Coordinates rounded to {DECIMALS} decimals (~0.1 m).",
            f"roads_reference.geojson simplified with {SIMPLIFY_M} m tolerance (display only).",
            "affected_road_sections.geojson = parts of affected roads inside the predicted flood (display only).",
            "All dashboard numbers come from infrastructure_summary.json and connectivity_summary.json."],
        "files": dict(sorted(sizes.items())),
        "sources_sha256": {str(p.relative_to(REPO_ROOT)): sha256(p) for p in sorted(needed)},
    }
    write_json(OUT / "manifest.json", manifest)
    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"wrote {len(list(OUT.iterdir()))} files, {total / 1024:.1f} KB, to {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
