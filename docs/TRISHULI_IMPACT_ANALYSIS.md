# Trishuli Infrastructure Impact and Connectivity (Experiment 1)

Run date: **2026-10-04**. Script: `scripts/analyze_trishuli_impact.py`. Code:
`backend/vantageq/impact/` (`osm.py`, `flood_geometry.py`, `infrastructure.py`, `connectivity.py`,
`geo.py`).

> Every result here is an **indicator** derived from (a) the frozen model's *unvalidated* flood
> prediction and (b) pre-event OSM. "Potentially affected" means geometric overlap with predicted
> flood, not confirmed damage. **"Potentially cut off" is a simplified graph-based indicator, not
> confirmed ground isolation.** No EMSR927, Copernicus EMS, UNOSAT or other damage map was used for
> any input, threshold or check.

---

## 1. Historical OSM source

**Planned:** ohsome API. **Found on 2026-10-04:**

| Endpoint | Result |
|---|---|
| `GET/POST https://api.ohsome.org/v1/elements/geometry` (and `/elements/centroid`) | **HTTP 403** |
| `https://api.ohsome.org/v1/elements/count` | works anonymously (data until 2026-07-27T09:00Z) |
| `POST https://api.heigit.org/ohsome-api/v2/extraction/features` | **HTTP 401**: v2 requires a free API key (docs: v2 migration guide); returns GeoParquet |

The project lead chose an **Overpass API attic query** instead (no account needed):

- Endpoint: `https://overpass-api.de/api/interpreter`
- Query (exact; built by `osm.build_query`):
  ```
  [out:json][timeout:180][date:"2026-08-25T00:00:00Z"][bbox:27.95,85.17,27.99,85.22];
  (way["highway"];nwr["building"]["building"!="no"];nwr["place"~"^(city|town|village|hamlet|isolated_dwelling)$"];);
  out body geom;
  ```
- `[date:...]` makes Overpass return OSM **as it existed at that instant**, so later edits are excluded.
- Cross-check: Overpass at 2026-08-25 returned 210 highway ways in the AOI; ohsome (anonymous count)
  at 2026-07-27 reported 118 vehicle-road + 92 path ways = 210. These are consistent.
- Raw response cached at `data/external/osm/trishuli_subaoi_20260825.json` (2.7 MB, git-ignored).

## 2. Snapshot date

**2026-08-25T00:00:00Z**, one day before the 2026-08-26 event (event time of day unknown, so the
previous midnight is used).

## 3. AOI

Same sub-AOI as the inference run: **85.17–85.22 E, 27.95–27.99 N** (WGS84), ≈ 4.9 × 4.4 km.
All features are clipped to it. Metric work (lengths, areas, distances) is done in **UTM 45N
(EPSG:32645)**; GeoJSON outputs are WGS84.

## 4. Features retrieved (counts in AOI)

| Feature | OSM selection | Count |
|---|---|---:|
| Roads (impact + graph) | ways, `highway` ∈ motorway…tertiary, unclassified, residential, living_street, service, track, road, *_link | **118** ways, 73.8 km (unclassified 88, track 15, primary 7, service 4, residential 3, living_street 1) |
| Foot paths (not in graph) | `highway` ∈ path, footway, steps, bridleway, pedestrian, cycleway | 92 ways, 34.0 km |
| Bridges | any `highway` way with `bridge=*`, `bridge≠no` (roads **and** paths) | **10** (path 5, unclassified 2, primary 2, service 1) |
| Buildings | nodes, ways, relations with `building=*`, `building≠no` | **4,877** (all closed ways) |
| Settlements | `place` ∈ city, town, village, hamlet, isolated_dwelling | **3** hamlets: Betrawati, Bhainse, Naubisephat |

Each OSM element is counted once (keyed by type + id).

## 5. Flood → vector (`flood_geometry.py`)

- Input: `data/processed/trishuli/trishuli_20260828_class.tif` (read only; SHA-256 prefix
  `7fed5e5f4e881563`, unchanged before/after).
- Mask = class **2 (flood)**, the model's argmax. Permanent water (class 1) is **not** used.
- Connected regions < **10 pixels** (8-connectivity) removed with `rasterio.features.sieve` (also
  fills holes < 10 px). Result: 1,547 → 1,546 pixels.
- Polygonised along pixel edges (EPSG:3857), dissolved, reprojected to UTM 45N; no smoothing.
  Result: **0.120 km²** in 5 parts. Saved as `flood_extent.geojson`.

## 6. Road rule (prototype rule)

Each road way is clipped to the AOI and intersected with the flood polygon (in UTM 45N metres). It is
**potentially affected** if its length inside the flood polygon is **≥ 10 m**. The 10 m threshold
(about one pixel; ignores corner touches) is a **prototype rule, not calibrated** against any
reference. Reported per way: total length, affected length, affected %. Graph edges use the same
10 m rule to become *blocked* (§9). Ways are split into graph edges only inside the connectivity
module; the reported road counts are per OSM way.

## 7. Bridge rule

A bridge way is reported as a **bridge intersecting predicted flood** (field `affected`) if some of
its length (> 0.01 m) lies inside the flood polygon. `near_flood` = within 20 m. This is overlap with
the *predicted* flood class; it does **not** mean a bridge is damaged or impassable. Bridges cross
water by design, so if the river under a bridge is predicted as *flood*, the bridge is flagged.
This rule alone is weak evidence of bridge impact.

## 8. Building rule

A building is **potentially affected** if part of its footprint (area **> 0.01 m²**) lies inside the
flood polygon. A point building would need to be covered by it; all 4,877 here are footprints.
Mere edge contact or floating-point slivers do not count (see §14). Reported per building:
footprint area and area inside flood.

## 9. Settlement connectivity (`connectivity.py`)

- Graph (NetworkX `MultiGraph`): **nodes** = OSM nodes that are road junctions or road ends inside
  the AOI; **edges** = vehicle-road pieces between them (ways split at junctions, using OSM node
  IDs). Foot paths are excluded. Ways leaving the AOI are cut at the last inside node.
- **Blocked edge:** length inside flood ≥ 10 m. Blocked edges are removed.
- **Main network:** connected component with the largest total road length. After blocking, it is
  the component holding most of the original main network's length.
- Each settlement is snapped to its nearest graph node (≤ 500 m).
- Status:
  - `potentially_cut_off`: **disconnected in the simplified road graph after removing
    flood-affected edges**, i.e. node in the main network before blocking, **not** in it after;
  - `connected`: in the main network before and after;
  - `not_on_main_network_before`: its road component was already separate from the main network
    in the clipped OSM graph (not caused by flooding);
  - `no_road_within_snap_distance`.
- Also reported: reachable road length from the settlement's node before/after (`access_reduced`).

**"Potentially cut off" is a simplified graph-based indicator, not confirmed ground isolation.**

## 10. Results

| Quantity | Value |
|---|---:|
| Road segments (ways) total / potentially affected | 118 / **3** |
| Road length total / inside predicted flood (affected ways) | 73.8 km / **0.189 km** |
| Bridges total / intersecting predicted flood / within 20 m | 10 / **2** / 2 |
| Buildings total / potentially affected | 4,877 / **36** (2,046 m² of 2,701 m² footprint inside flood) |
| Settlements analysed / potentially cut off | 3 / **2** |

Potentially affected roads (all primary road NH42 / NH18, Pasang Lhamu Highway):

| OSM way | Name | Length | Inside flood |
|---|---|---:|---:|
| 341461646 | Falaakhu River Bridge (bridge) | 67 m | 67 m (100 %) |
| 345313941 | Pasang Lhamu Highway | 390 m | 30 m (7.6 %) |
| 379104232 | Pasang Lhamu Highway | 1,961 m | 92 m (4.7 %) |

Bridges intersecting predicted flood: **Falaakhu River Bridge** (NH42, way 341461646) and footbridge
**"Tupche Aama ko pool"** (path, way 281810148); both lie entirely inside the predicted flood polygon.

Network: 205 nodes, 192 edges, 22 components before (clipping fragments the network), **3 blocked
edges**, 25 components after; main network 31.1 km → 28.4 km; 14 nodes leave it.

| Settlement | Snap distance | Status | Reachable road before → after |
|---|---:|---|---|
| Betrawati (hamlet) | 35 m | **potentially_cut_off** | 31.13 → 0.70 km |
| Bhainse (hamlet) | 28 m | **potentially_cut_off** | 31.13 → 0.39 km |
| Naubisephat (hamlet) | 187 m | not_on_main_network_before | 1.45 → 1.45 km |

**Why Betrawati and Bhainse are potentially cut off (verified edge by edge).** Inside the AOI the
NH42/NH18 highway forms a chain; after removing the 3 blocked edges the original main network
(81 nodes, 31.1 km) splits as:

```
MAIN network after blocking (67 nodes, 28.4 km; north across the Falaakhu river)
   ⟷ [blocked] Falaakhu River Bridge, way 341461646 (67 m of 67 m inside predicted flood)
   ⟷ single junction node 268330848
   ⟷ [blocked] Pasang Lhamu Highway piece, way 345313941 (30 m inside flood)
   ⟷ Betrawati's piece (10 nodes, 0.70 km of road)
   ⟷ [blocked] Pasang Lhamu Highway piece, way 379104232 (92 m inside flood)
   ⟷ Bhainse's piece (3 nodes, 0.39 km): dead end at the AOI's south-west boundary
```

- **Betrawati** (snapped 35 m to node 13186041199): its only in-AOI route to the main network crosses
  the bridge and the 30 m highway piece, both blocked in series. Southwards it only reaches
  Bhainse's piece, itself separated by the blocked 92 m piece.
- **Bhainse** (snapped 28 m to node 2532475279): its route runs through Betrawati's piece, so it is
  cut by the 92 m piece (and the two edges beyond). South of Bhainse the highway leaves the AOI;
  that continuation is not in the graph.
- Unblocking any **one** of the three edges leaves both statuses unchanged (descriptive check, not
  tuning; outputs unchanged).

## 11. Limitations

- The flood map is an **unvalidated** model prediction (test flood recall 0.32 on Kuro Siwo; strong
  terrain domain gap). Missed flood → missed impacts; false flood → false impacts.
- **Bridges over rivers** are flagged whenever the river is predicted as flood (§7). The
  Falaakhu River Bridge result should be read in that light.
- **AOI clipping:** the graph knows only roads inside the 4.9 × 4.4 km box. NH42 continues south of
  Bhainse beyond the AOI, so Bhainse's "cut-off" status is especially uncertain. Bhainse lies on the AOI's
  south-west edge; routes leaving the AOI are invisible, so it may have alternative access outside.
- Only **vehicle roads** form the graph; 92 foot paths (34 km), often the real access in hill
  villages, are excluded.
- **Only 3 settlements** are mapped in OSM in this AOI, all hamlets. Thousands of buildings are not
  represented by any settlement point.
- OSM completeness/accuracy in rural Nepal is uncertain; building footprints are mostly
  `building=yes`.
- Overlap ≠ damage: a building or road touching predicted flood is not confirmed damaged or impassable.
- Overpass attic data reflect OSM history as stored by Overpass; a single snapshot was used.

## 12. Output checks (2026-10-04 review)

- All 9 GeoJSON files: RFC 7946 WGS84, valid geometries, no duplicate OSM IDs.
- Road lengths (UTM 45N) agree with independent geodesic (haversine) lengths within 0.1–1.2 m.
- Flood area: polygon 0.1201 km² vs 1,546 px × 78.0 m² = 0.1206 km².
- Roads are clipped to the AOI; reprojecting the AOI box through its corners lets roads overshoot
  the lon/lat box by ≤ 0.25 m (negligible). Buildings straddling the boundary are kept whole
  (≤ 13 m beyond the box).

## 13. Outputs (`data/processed/trishuli/infrastructure/`, git-ignored)

`flood_extent.geojson`, `roads.geojson`, `bridges.geojson`, `buildings.geojson`, `settlements.geojson`,
`affected_roads.geojson`, `affected_bridges.geojson`, `affected_buildings.geojson`,
`cutoff_settlements.geojson`, `infrastructure_summary.json`, `connectivity_summary.json`,
`quicklook_infrastructure.png`.

```bash
.venv/bin/pip install -r requirements.txt     # adds shapely, networkx
.venv/bin/python scripts/analyze_trishuli_impact.py
```

## 14. Correction during review

The first run counted 37 affected buildings. One (OSM way 886277383) only shared a floating-point
sliver with the flood polygon (≈ 0.001 m², reported as 0.0 m²), because the rule used `intersects()`,
which is also true for edge contact. The rule now requires a positive overlap (> 0.01 m² for
footprints, > 0.01 m for bridges); a regression test covers it. Count: **37 → 36**; nothing else
changed.

## 15. Attribution

Map data © OpenStreetMap contributors, available under the Open Database License (ODbL 1.0),
retrieved via the Overpass API (historical attic query). Flood map derived from modified
Copernicus Sentinel-1 data (2026) processed by the VantageQ team.
