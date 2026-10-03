"""Trishuli infrastructure impact + simplified connectivity from the frozen-model flood map.

Inputs (permitted only):
- data/processed/trishuli/trishuli_20260828_class.tif (frozen model output; not modified)
- pre-event OSM: Overpass attic snapshot at 2026-08-25T00:00:00Z, cached in data/external/osm/

Outputs: data/processed/trishuli/infrastructure/ (GeoJSON in WGS84, summaries, quicklook).
All results are "potentially affected" / "potentially cut off" indicators, not confirmed damage or
isolation. See docs/TRISHULI_IMPACT_ANALYSIS.md.

Usage:
  python scripts/analyze_trishuli_impact.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from shapely.geometry import box

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from vantageq.impact import connectivity as conn  # noqa: E402
from vantageq.impact import flood_geometry as fg  # noqa: E402
from vantageq.impact import infrastructure as infra  # noqa: E402
from vantageq.impact import osm  # noqa: E402
from vantageq.impact.geo import UTM45N, WGS84, reproject, write_geojson  # noqa: E402

AOI_BBOX = (85.17, 27.95, 85.22, 27.99)  # same sub-AOI as the inference run
OSM_DATE = "2026-08-25T00:00:00Z"  # one day before the 2026-08-26 event
CLASS_MAP = REPO_ROOT / "data" / "processed" / "trishuli" / "trishuli_20260828_class.tif"
OSM_CACHE = REPO_ROOT / "data" / "external" / "osm" / "trishuli_subaoi_20260825.json"
OUT = REPO_ROOT / "data" / "processed" / "trishuli" / "infrastructure"


def props(item: dict, keys) -> dict:
    return {k: item.get(k) for k in keys}


def to_utm(items: list[dict]) -> list[dict]:
    return [it | {"geometry": reproject(it["geometry"], WGS84, UTM45N)} for it in items]


def main() -> int:
    # Flood geometry (raster CRS -> UTM for metric intersections).
    flood_r, epsg, raw_px, kept_px = fg.flood_polygon_from_raster(CLASS_MAP)
    flood = reproject(flood_r, epsg, UTM45N)
    aoi = reproject(box(*AOI_BBOX), WGS84, UTM45N)
    print(f"flood: {raw_px} class-2 pixels, {kept_px} after removing regions < {fg.MIN_PIXELS} px; "
          f"polygon area {flood.area / 1e6:.3f} km² ({len(getattr(flood, 'geoms', [flood]))} parts)")

    # Pre-event OSM.
    raw = osm.fetch(AOI_BBOX, OSM_DATE, OSM_CACHE)
    feats = osm.parse(raw)
    print("OSM elements parsed:", {k: len(v) for k, v in feats.items()})
    roads, paths, bridges = to_utm(feats["roads"]), to_utm(feats["paths"]), to_utm(feats["bridges"])
    buildings, settlements = to_utm(feats["buildings"]), to_utm(feats["settlements"])

    road_res = infra.road_impacts(roads, flood, aoi)
    bridge_res = infra.bridge_impacts(bridges, flood, aoi)
    bldg_res = infra.building_impacts(buildings, flood, aoi)
    settle_in = [s for s in settlements if aoi.contains(s["geometry"])]

    # Connectivity graph from vehicle roads (OSM node ids -> metric coordinates).
    node_xy = {}
    for r in roads:
        for n, xy in zip(r["nodes"], r["geometry"].coords):
            node_xy[n] = xy
    g = conn.build_graph(roads, node_xy, aoi)
    conn.mark_blocked(g, flood)
    settle_res, net = conn.settlement_connectivity(g, settle_in)

    # Outputs.
    OUT.mkdir(parents=True, exist_ok=True)
    rk = ("osm_type", "osm_id", "highway", "name", "ref", "bridge", "length_m", "affected_length_m",
          "affected_pct", "affected")
    bk = rk + ("near_flood",)
    gk = ("osm_type", "osm_id", "building", "footprint_m2", "flooded_m2", "affected")
    sk = ("osm_type", "osm_id", "place", "name", "name_local", "nearest_node", "snap_distance_m", "status",
          "reachable_road_km_before", "reachable_road_km_after", "access_reduced", "blocked_edges_at_node")
    files = {
        "flood_extent.geojson": [(flood, {"class": "flood", "source": CLASS_MAP.name, "min_pixels": fg.MIN_PIXELS})],
        "roads.geojson": [(r["geometry"], props(r, rk)) for r in road_res],
        "bridges.geojson": [(b["geometry"], props(b, bk)) for b in bridge_res],
        "buildings.geojson": [(b["geometry"], props(b, gk)) for b in bldg_res],
        "settlements.geojson": [(s["geometry"], props(s, sk)) for s in settle_res],
        "affected_roads.geojson": [(r["geometry"], props(r, rk)) for r in road_res if r["affected"]],
        "affected_bridges.geojson": [(b["geometry"], props(b, bk)) for b in bridge_res if b["affected"]],
        "affected_buildings.geojson": [(b["geometry"], props(b, gk)) for b in bldg_res if b["affected"]],
        "cutoff_settlements.geojson": [(s["geometry"], props(s, sk)) for s in settle_res
                                       if s["status"] == "potentially_cut_off"],
    }
    for name, fc in files.items():
        write_geojson(OUT / name, fc)

    aff_roads = [r for r in road_res if r["affected"]]
    infra_summary = {
        "note": "Potentially affected = geometric overlap with the model's predicted flood class; "
                "not confirmed damage. No reference damage map used.",
        "aoi_wgs84": AOI_BBOX, "osm_snapshot": OSM_DATE, "osm_source": osm.OVERPASS_URL + " (attic [date:])",
        "flood_source": str(CLASS_MAP.relative_to(REPO_ROOT)), "flood_class": fg.FLOOD_CLASS,
        "flood_min_pixels": fg.MIN_PIXELS, "flood_pixels_raw": raw_px, "flood_pixels_kept": kept_px,
        "flood_area_km2_utm": round(flood.area / 1e6, 4),
        "rules": {"road": f"OSM way clipped to AOI; affected if length inside flood >= {infra.MIN_AFFECTED_LENGTH_M} m "
                          "(prototype rule, not calibrated)",
                  "bridge": "highway way with bridge!=no; 'affected' = intersects predicted flood (length inside > "
                            f"{infra.MIN_OVERLAP} m); near_flood within {infra.NEAR_M} m",
                  "building": f"affected if footprint area inside flood > {infra.MIN_OVERLAP} m² (point: covered)",
                  "status": "prototype rules; overlap with predicted flood is not confirmed damage"},
        "roads": {"segments_total": len(road_res), "segments_affected": len(aff_roads),
                  "length_km_total": round(sum(r["length_m"] for r in road_res) / 1000, 3),
                  "affected_length_km": round(sum(r["affected_length_m"] for r in aff_roads) / 1000, 3),
                  "affected_by_highway": dict(Counter(r["highway"] for r in aff_roads)),
                  "by_highway_total": dict(Counter(r["highway"] for r in road_res))},
        "paths_not_in_graph": {"segments": len(paths),
                               "length_km": round(sum(p["geometry"].intersection(aoi).length for p in paths) / 1000, 3)},
        "bridges": {"total": len(bridge_res), "affected": sum(b["affected"] for b in bridge_res),
                    "near_flood": sum(b["near_flood"] for b in bridge_res),
                    "by_highway_total": dict(Counter(b["highway"] for b in bridge_res))},
        "buildings": {"total": len(bldg_res), "affected": sum(b["affected"] for b in bldg_res),
                      "by_osm_type": dict(Counter(b["osm_type"] for b in bldg_res))},
        "settlements": {"analyzed": len(settle_res), "potentially_cut_off":
                        sum(s["status"] == "potentially_cut_off" for s in settle_res)},
    }
    conn_summary = {
        "definition": "potentially_cut_off = disconnected in the simplified road graph after removing "
                      "flood-affected edges: the settlement's nearest road node is in the main road network "
                      "(largest component by length) before removing blocked edges and not in it after. "
                      "Graph indicator only; not confirmed real-world isolation.",
        "limitations": ["footpaths (path/footway/steps/...) are excluded from the graph",
                        "graph clipped to the AOI: roads continuing beyond it (e.g. NH42 south of Bhainse) "
                        "are invisible, so alternative access outside the AOI is not considered",
                        "depends on the unvalidated model flood prediction"],
        "graph_roads": "vehicle highway classes only (footpaths excluded)", "blocked_edge_rule":
        f"edge length inside flood >= {conn.MIN_BLOCK_LENGTH_M} m (prototype rule, not calibrated)",
        "max_snap_m": conn.MAX_SNAP_M,
        "network": net, "settlements": [props(s, sk) for s in settle_res],
        "status_counts": dict(Counter(s["status"] for s in settle_res)),
    }
    (OUT / "infrastructure_summary.json").write_text(json.dumps(infra_summary, indent=1, ensure_ascii=False))
    (OUT / "connectivity_summary.json").write_text(json.dumps(conn_summary, indent=1, ensure_ascii=False))
    quicklook(flood, road_res, bridge_res, bldg_res, settle_res, aoi, OUT / "quicklook_infrastructure.png")

    print(json.dumps({k: infra_summary[k] for k in ("roads", "paths_not_in_graph", "bridges", "buildings",
                                                    "settlements")}, indent=1, ensure_ascii=False))
    print("network:", net)
    for s in settle_res:
        print(f"  settlement {s['name']!s:<22} ({s['place']}, osm {s['osm_id']}): {s['status']}, snap "
              f"{s['snap_distance_m']} m, reachable road {s['reachable_road_km_before']} -> "
              f"{s['reachable_road_km_after']} km")
    print("outputs in", OUT.relative_to(REPO_ROOT))
    return 0


def quicklook(flood, roads, bridges, buildings, settlements, aoi, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10, 9))
    for poly in getattr(flood, "geoms", [flood]):
        if not poly.is_empty:
            ax.fill(*poly.exterior.xy, color="#6baed6", alpha=0.7, lw=0)
    for b in buildings:
        geom = b["geometry"]
        c = "#e6550d" if b["affected"] else "#bdbdbd"
        if geom.geom_type == "Point":
            ax.plot(geom.x, geom.y, ".", ms=1, color=c)
        else:
            for p in getattr(geom, "geoms", [geom]):
                ax.fill(*p.exterior.xy, color=c, lw=0)
        if b["affected"]:  # footprints are too small to see at map scale; mark them
            pt = geom.representative_point()
            ax.plot(pt.x, pt.y, "o", ms=4, color="#e6550d", markeredgecolor="black", markeredgewidth=0.3, zorder=4)
    for r in roads:
        major = r["highway"] in ("primary", "secondary", "trunk")
        for line in getattr(r["geometry"], "geoms", [r["geometry"]]):
            ax.plot(*line.xy, color="#d7301f" if r["affected"] else "#525252", ls="--" if r["affected"] else "-",
                    lw=1.8 if major or r["affected"] else 0.8, zorder=3)
        if r["affected"]:  # only the part inside the predicted flood is drawn thick
            part = r["geometry"].intersection(flood)
            for line in getattr(part, "geoms", [part]):
                if line.geom_type == "LineString" and not line.is_empty:
                    ax.plot(*line.xy, color="#a50f15", lw=5, zorder=4, solid_capstyle="butt")
    for b in bridges:
        c = b["geometry"].centroid
        ax.plot(c.x, c.y, marker="X" if b["affected"] else "^", ms=9, zorder=5,
                color="#a50f15" if b["affected"] else "black", markeredgecolor="white")
    for s in settlements:
        cut = s["status"] == "potentially_cut_off"
        ax.plot(s["geometry"].x, s["geometry"].y, marker="*" if cut else "o", ms=16 if cut else 9, zorder=6,
                color="#cb181d" if cut else "#31a354", markeredgecolor="black")
        ax.annotate(f"{s['name'] or s['osm_id']}\n({s['status'].replace('_', ' ')})",
                    (s["geometry"].x, s["geometry"].y), xytext=(6, 6), textcoords="offset points", fontsize=8,
                    bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8})
    ax.plot(*aoi.exterior.xy, color="black", lw=1, ls="--")
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("UTM 45N easting (m)")
    ax.set_ylabel("northing (m)")
    ax.legend(handles=[
        Patch(color="#6baed6", label="predicted flood (frozen model)"),
        Line2D([], [], color="#a50f15", lw=5, label="road length inside predicted flood"),
        Line2D([], [], color="#d7301f", lw=1.8, ls="--", label="rest of a potentially affected road"),
        Line2D([], [], color="#525252", lw=1.5, label="road (OSM 2026-08-25)"),
        Line2D([], [], marker="X", color="#a50f15", ls="", ms=9, label="bridge intersecting predicted flood"),
        Line2D([], [], marker="^", color="black", ls="", ms=8, label="bridge"),
        Line2D([], [], marker="o", color="#e6550d", ls="", ms=5, label="potentially affected building"),
        Line2D([], [], marker="*", color="#cb181d", ls="", ms=14, label="potentially cut-off settlement"),
        Line2D([], [], marker="o", color="#31a354", ls="", ms=8, label="settlement (not cut off; see status)")],
        loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("Trishuli sub-AOI: potential infrastructure impact (indicator, not validated)\n"
                 "© OpenStreetMap contributors (ODbL), pre-event snapshot 2026-08-25", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
