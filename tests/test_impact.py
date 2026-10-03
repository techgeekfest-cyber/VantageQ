"""Impact analysis on synthetic data: flood vectorisation, intersections, graph blocking, cut-off logic."""

import sys
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from vantageq.impact import connectivity as conn  # noqa: E402
from vantageq.impact import flood_geometry as fg  # noqa: E402
from vantageq.impact import infrastructure as infra  # noqa: E402
from vantageq.impact import osm  # noqa: E402

AOI = box(-1000, -1000, 1000, 1000)


# --------------------------------------------------------------------------- flood geometry

def test_flood_raster_to_polygon_removes_specks_and_keeps_area():
    cls = np.zeros((20, 20), "uint8")
    cls[2:6, 2:6] = 2      # 16-pixel flood block
    cls[15, 15] = 2        # single-pixel speck -> removed (< 10 px)
    cls[10:12, 10:12] = 1  # permanent water is not flood
    mask = fg.flood_mask(cls, min_pixels=10)
    assert mask.sum() == 16
    poly = fg.mask_to_polygon(mask, from_origin(0, 200, 10, 10))
    assert poly.area == pytest.approx(16 * 100)
    assert poly.bounds == (20.0, 140.0, 60.0, 180.0)


def test_empty_flood_gives_empty_geometry():
    mask = fg.flood_mask(np.zeros((5, 5), "uint8"))
    assert not mask.any()
    assert fg.mask_to_polygon(mask, from_origin(0, 50, 10, 10)).is_empty


# --------------------------------------------------------------------------- intersections

FLOOD = box(0, 0, 100, 100)


def road(osm_id, coords, **kw):
    return {"osm_type": "way", "osm_id": osm_id, "highway": kw.get("highway", "unclassified"), "name": None,
            "ref": None, "bridge": kw.get("bridge"), "nodes": kw.get("nodes", []), "geometry": LineString(coords)}


def test_road_affected_length_percentage_and_threshold():
    roads = [road(1, [(-100, 50), (300, 50)]),   # 100 m of 400 m inside -> affected, 25 %
             road(2, [(95, -50), (95, 150)]),     # 100 m inside -> affected, 50 %
             road(3, [(-50, 105), (50, 105)]),    # misses
             road(4, [(-5, -50), (5, -50), (5, 3)])]  # only 3 m inside -> below 10 m threshold
    res = {r["osm_id"]: r for r in infra.road_impacts(roads, FLOOD, AOI)}
    assert res[1]["affected_length_m"] == pytest.approx(100) and res[1]["affected_pct"] == pytest.approx(25)
    assert res[1]["affected"] and res[2]["affected"]
    assert not res[3]["affected"] and res[3]["affected_length_m"] == 0
    assert res[4]["affected_length_m"] == pytest.approx(3) and not res[4]["affected"]


def test_roads_are_clipped_to_aoi_and_deduplicated():
    r = road(7, [(-2000, 50), (2000, 50)])
    res = infra.road_impacts([r, dict(r)], FLOOD, AOI)
    assert len(res) == 1 and res[0]["length_m"] == pytest.approx(2000)


def test_bridge_and_building_rules_and_no_double_counting():
    bridges = [road(10, [(50, 50), (50, 120)], bridge="yes"), road(11, [(110, 50), (130, 50)], bridge="yes"),
               road(12, [(400, 400), (420, 400)], bridge="yes")]
    b = {x["osm_id"]: x for x in infra.bridge_impacts(bridges, FLOOD, AOI)}
    assert b[10]["affected"] and b[10]["affected_length_m"] == pytest.approx(50)
    assert not b[11]["affected"] and b[11]["near_flood"]   # 10 m away, within 20 m
    assert not b[12]["affected"] and not b[12]["near_flood"]

    blds = [{"osm_type": "way", "osm_id": 1, "building": "yes", "geometry": box(90, 90, 110, 110)},
            {"osm_type": "way", "osm_id": 1, "building": "yes", "geometry": box(90, 90, 110, 110)},  # duplicate
            {"osm_type": "node", "osm_id": 2, "building": "yes", "geometry": Point(10, 10)},
            {"osm_type": "way", "osm_id": 3, "building": "yes", "geometry": box(200, 200, 210, 210)}]
    res = infra.building_impacts(blds, FLOOD, AOI)
    assert len(res) == 3
    hit = {(x["osm_type"], x["osm_id"]): x for x in res}
    assert hit[("way", 1)]["affected"] and hit[("way", 1)]["flooded_m2"] == pytest.approx(100)
    assert hit[("node", 2)]["affected"] and not hit[("way", 3)]["affected"]


# --------------------------------------------------------------------------- graph

def chain_network():
    """Main chain 1-2-3-4-5 (x = 0..4000 m), spur 3-6, separate component 7-8."""
    xy = {1: (0, 0), 2: (1000, 0), 3: (2000, 0), 4: (3000, 0), 5: (4000, 0), 6: (2000, 800),
          7: (0, 3000), 8: (500, 3000)}
    roads = [{"osm_id": 100, "nodes": [1, 2, 3, 4, 5], "highway": "primary"},
             {"osm_id": 101, "nodes": [3, 6], "highway": "track"},
             {"osm_id": 102, "nodes": [7, 8], "highway": "track"}]
    return roads, xy, box(-500, -500, 4500, 3500)


def test_graph_splits_ways_at_junctions_and_ends_only():
    roads, xy, aoi = chain_network()
    g = conn.build_graph(roads, xy, aoi)
    assert set(g.nodes) == {1, 3, 5, 6, 7, 8}  # 2 and 4 are interior, non-junction nodes
    lengths = sorted(d["length_m"] for *_, d in g.edges(data=True))
    assert lengths == pytest.approx([500, 800, 2000, 2000])


def test_graph_drops_nodes_outside_aoi():
    roads, xy, _ = chain_network()
    g = conn.build_graph(roads, xy, box(-500, -500, 2500, 1000))  # cuts the chain at x = 2500
    assert 4 not in g and 5 not in g and 7 not in g
    assert g.has_edge(1, 3) and g.has_edge(3, 6)


def test_blocked_edge_removal_and_cut_off_status():
    roads, xy, aoi = chain_network()
    g = conn.build_graph(roads, xy, aoi)
    flood = box(3400, -50, 3500, 50)  # 100 m of the 3-5 edge
    assert conn.mark_blocked(g, flood) == 1
    assert conn.without_blocked(g).number_of_edges() == g.number_of_edges() - 1
    settlements = [{"name": "east", "geometry": Point(4050, 30)},     # near node 5 -> cut off
                   {"name": "west", "geometry": Point(-40, 0)},       # near node 1 -> connected
                   {"name": "island", "geometry": Point(510, 3010)},  # separate component before
                   {"name": "far", "geometry": Point(4400, 3400)}]    # > 500 m from any node
    res, summary = conn.settlement_connectivity(g, settlements)
    status = {s["name"]: s["status"] for s in res}
    assert status == {"east": "potentially_cut_off", "west": "connected",
                      "island": "not_on_main_network_before", "far": "no_road_within_snap_distance"}
    east = next(s for s in res if s["name"] == "east")
    assert east["reachable_road_km_before"] == pytest.approx(4.8) and east["reachable_road_km_after"] == 0
    assert east["access_reduced"]
    assert summary["blocked_edges"] == 1 and summary["components_after"] == summary["components_before"] + 1


def test_short_overlap_does_not_block_edge():
    roads, xy, aoi = chain_network()
    g = conn.build_graph(roads, xy, aoi)
    assert conn.mark_blocked(g, box(3995, -5, 4005, 5)) == 0  # 5 m < 10 m threshold


# --------------------------------------------------------------------------- OSM parsing

def test_osm_parse_classifies_and_deduplicates():
    way = {"type": "way", "id": 1, "nodes": [1, 2], "tags": {"highway": "primary", "bridge": "yes", "ref": "NH42"},
           "geometry": [{"lon": 85.18, "lat": 27.96}, {"lon": 85.181, "lat": 27.961}]}
    raw = {"elements": [
        way, dict(way),
        {"type": "way", "id": 2, "nodes": [3, 4], "tags": {"highway": "footway", "bridge": "no"},
         "geometry": [{"lon": 85.18, "lat": 27.96}, {"lon": 85.19, "lat": 27.97}]},
        {"type": "way", "id": 3, "nodes": [5, 6, 7, 5], "tags": {"building": "house"},
         "geometry": [{"lon": 0, "lat": 0}, {"lon": 1, "lat": 0}, {"lon": 1, "lat": 1}, {"lon": 0, "lat": 0}]},
        {"type": "node", "id": 9, "lat": 27.97, "lon": 85.19, "tags": {"place": "hamlet", "name": "X"}},
        {"type": "node", "id": 10, "lat": 27.97, "lon": 85.19, "tags": {"place": "suburb"}},
    ]}
    f = osm.parse(raw)
    assert len(f["roads"]) == 1 and len(f["bridges"]) == 1 and f["bridges"][0]["ref"] == "NH42"
    assert len(f["paths"]) == 1  # footway with bridge=no is not a bridge
    assert len(f["buildings"]) == 1 and isinstance(f["buildings"][0]["geometry"], Polygon)
    assert [s["name"] for s in f["settlements"]] == ["X"]


def test_query_is_historical_and_bbox_limited():
    q = osm.build_query((85.17, 27.95, 85.22, 27.99), "2026-08-25T00:00:00Z")
    assert '[date:"2026-08-25T00:00:00Z"]' in q and "[bbox:27.95,85.17,27.99,85.22]" in q


def test_touching_or_sliver_overlap_is_not_counted_as_affected():
    flood = box(0, 0, 100, 100)
    blds = [{"osm_type": "way", "osm_id": 1, "building": "yes", "geometry": box(100, 0, 110, 10)},   # shares an edge
            {"osm_type": "way", "osm_id": 2, "building": "yes", "geometry": box(100 - 1e-6, 20, 110, 30)},  # sliver
            {"osm_type": "way", "osm_id": 3, "building": "yes", "geometry": box(95, 40, 105, 50)}]  # 50 m² inside
    res = {b["osm_id"]: b for b in infra.building_impacts(blds, flood, AOI)}
    assert not res[1]["affected"] and not res[2]["affected"] and res[3]["affected"]
    br = [road(20, [(100, 50), (120, 50)], bridge="yes")]  # bridge only touching the flood edge
    assert not infra.bridge_impacts(br, flood, AOI)[0]["affected"]
