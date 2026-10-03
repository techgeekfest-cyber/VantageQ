"""Simplified road-network connectivity (NetworkX). Not a routing or traffic model.

Graph: nodes = OSM nodes that are road intersections or road ends inside the AOI; edges = road
pieces between them (OSM ways split at junctions). An edge is *blocked* when its length inside the
flood polygon is >= MIN_BLOCK_LENGTH_M. The *main network* is the connected component with the
largest total road length.

A settlement is snapped to its nearest graph node (within MAX_SNAP_M). It is *potentially cut off*,
i.e. disconnected in the simplified road graph after removing flood-affected edges, if that node is in the main network before blocking and not in it after blocking, where the
post-blocking main network is the component holding most of the original main network's length.
This is a graph indicator only; it does not establish real-world isolation.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point
from shapely.prepared import prep

MIN_BLOCK_LENGTH_M = 10.0
MAX_SNAP_M = 500.0


def build_graph(roads: list[dict], node_xy: dict[int, tuple[float, float]], aoi) -> nx.MultiGraph:
    """roads: dicts with 'osm_id', 'nodes' (OSM node ids) and attributes; node_xy: id -> (x, y) metric.

    Only nodes inside `aoi` are used; a way leaving the AOI is split into its inside runs.
    """
    paoi = prep(aoi)
    inside = {n: paoi.contains(Point(xy)) for n, xy in node_xy.items()}
    runs = []
    for r in roads:
        run = []
        for n in r["nodes"] + [None]:
            if n is not None and inside.get(n, False):
                run.append(n)
            else:
                if len(run) >= 2:
                    runs.append((r, run))
                run = []
    use = Counter(n for _, run in runs for n in run)
    junction = {n for n, c in use.items() if c >= 2} | {run[0] for _, run in runs} | {run[-1] for _, run in runs}

    g = nx.MultiGraph()
    for r, run in runs:
        start = 0
        for i in range(1, len(run)):
            if run[i] in junction or i == len(run) - 1:
                piece = run[start:i + 1]
                line = LineString([node_xy[n] for n in piece])
                g.add_edge(piece[0], piece[-1], osm_id=r["osm_id"], highway=r.get("highway"), name=r.get("name"),
                           bridge=r.get("bridge"), geometry=line, length_m=float(line.length))
                start = i
    for n in g.nodes:
        g.nodes[n]["xy"] = node_xy[n]
    return g


def mark_blocked(g: nx.MultiGraph, flood, min_len: float = MIN_BLOCK_LENGTH_M) -> int:
    n = 0
    for _, _, d in g.edges(data=True):
        inside = d["geometry"].intersection(flood).length if d["geometry"].intersects(flood) else 0.0
        d["affected_length_m"] = float(inside)
        d["blocked"] = inside >= min_len
        n += d["blocked"]
    return n


def without_blocked(g: nx.MultiGraph) -> nx.MultiGraph:
    h = g.copy()
    h.remove_edges_from([(u, v, k) for u, v, k, d in g.edges(keys=True, data=True) if d.get("blocked")])
    return h


def component_lengths(g: nx.MultiGraph) -> list[tuple[set, float]]:
    comps = []
    for c in nx.connected_components(g):
        comps.append((c, sum(d["length_m"] for _, _, d in g.subgraph(c).edges(data=True))))
    return sorted(comps, key=lambda t: -t[1])


def main_component(g: nx.MultiGraph) -> set:
    comps = component_lengths(g)
    return comps[0][0] if comps else set()


def main_after(g_after: nx.MultiGraph, g_before: nx.MultiGraph, main_before: set) -> set:
    """Component of the blocked graph holding the most road length of the original main network."""
    best, best_len = set(), -1.0
    for c in nx.connected_components(g_after):
        inter = c & main_before
        length = sum(d["length_m"] for _, _, d in g_before.subgraph(inter).edges(data=True))
        if length > best_len:
            best, best_len = c, length
    return best


def nearest_node(g: nx.MultiGraph, xy) -> tuple[int | None, float]:
    if g.number_of_nodes() == 0:
        return None, float("inf")
    ids = list(g.nodes)
    pts = np.array([g.nodes[n]["xy"] for n in ids])
    d = np.hypot(pts[:, 0] - xy[0], pts[:, 1] - xy[1])
    i = int(d.argmin())
    return ids[i], float(d[i])


def _reachable_km(g: nx.MultiGraph, node) -> float:
    if node is None or node not in g:
        return 0.0
    c = nx.node_connected_component(g, node)
    return sum(d["length_m"] for _, _, d in g.subgraph(c).edges(data=True)) / 1000


def settlement_connectivity(g: nx.MultiGraph, settlements: list[dict], max_snap_m: float = MAX_SNAP_M) -> tuple[list, dict]:
    """Status per settlement ('geometry' = metric Point) plus a network summary."""
    before_main = main_component(g)
    g_after = without_blocked(g)
    after_main = main_after(g_after, g, before_main)
    results = []
    for s in settlements:
        node, dist = nearest_node(g, (s["geometry"].x, s["geometry"].y))
        base = {"nearest_node": node, "snap_distance_m": round(dist, 1)}
        if node is None or dist > max_snap_m:
            status, before, after = "no_road_within_snap_distance", 0.0, 0.0
        else:
            before, after = _reachable_km(g, node), _reachable_km(g_after, node)
            if node not in before_main:
                status = "not_on_main_network_before"
            elif node not in after_main:
                status = "potentially_cut_off"
            else:
                status = "connected"
        blocked_near = sorted({d["osm_id"] for u, v, d in g.edges(node, data=True) if d.get("blocked")}) if node in g else []
        results.append(s | base | {"status": status, "reachable_road_km_before": round(before, 2),
                                   "reachable_road_km_after": round(after, 2),
                                   "access_reduced": after < before - 1e-9,
                                   "blocked_edges_at_node": blocked_near})
    summary = {
        "nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
        "blocked_edges": sum(1 for *_, d in g.edges(data=True) if d.get("blocked")),
        "components_before": nx.number_connected_components(g),
        "components_after": nx.number_connected_components(g_after),
        "main_network_km_before": round(sum(d["length_m"] for *_, d in g.subgraph(before_main).edges(data=True)) / 1000, 3),
        "main_network_km_after": round(sum(d["length_m"] for *_, d in g_after.subgraph(after_main).edges(data=True)) / 1000, 3),
        "nodes_leaving_main_network": len(before_main - after_main),
    }
    return results, summary
