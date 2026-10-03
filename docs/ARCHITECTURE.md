# VantageQ — Architecture

Status: **initial design, nothing implemented yet**. Update this document when decisions change.
Items marked **[VERIFY]** are not yet confirmed and must be checked before code depends on them.

---

## 1. Purpose and scope

VantageQ is a practical prototype for IIT Mandi Multimodal AI Hackathon 2026, Track B ("Mapping
Flood Damage from Space"). The goal is a **complete, reliable, demoable** system that can qualify
for the second round. Fewer features that work well beat many that work partly.

Given an **area of interest (AOI)** and an **event date**, it produces:

1. A flood extent map and a simple before/after change map
2. Potentially affected buildings, roads and bridges (pre-event OSM)
3. Settlements potentially cut off from the nearest town or hospital
4. A simple interactive map dashboard
5. A one-page situation report
6. Reproducible evaluation results

The AI component is **Sentinel-1 SAR flood segmentation** with a U-Net.

It is not an operational system. All impact outputs are worded as *potentially affected*.

### 1.1 Priority order

Work proceeds strictly in this order. A later item does not start until earlier items work
end-to-end on the target AOI.

1. Working Sentinel-1 before/after data pipeline
2. Flood segmentation AI
3. Flood/change map
4. OSM infrastructure impact estimation
5. Cut-off settlement analysis with a simple road graph
6. Interactive dashboard
7. One-page situation report
8. EMSR927 comparison/evaluation
9. *(Bonus)* DEM flood-path tracing — only if 1–8 already work

### 1.2 In scope

- U-Net flood segmentation, trained on a **manageable subset** of Kuro Siwo
- Simple Sentinel-1 log-ratio change map from the same orbit track
- GeoPandas/Shapely intersections for infrastructure impact
- NetworkX shortest-path connectivity on a simple road graph
- One FastAPI backend, one Next.js frontend, files on disk
- Template-generated report from pipeline outputs

### 1.3 Explicitly out of scope

- Complex or novel model architectures, ensembles, foundation models
- Bulk dataset downloads before we know exactly what is needed
- Microservices, task queues, databases (unless clearly necessary)
- Agents or chatbots
- Uncertainty frameworks beyond the model's own output probability
- Advanced routing (weighted costs, travel time, optimisation)
- Elaborate fallback systems not required for the live demo
- Sentinel-2 and Copernicus DEM as required inputs (both **optional/secondary** until the core
  pipeline works)

---

## 2. System components

```
  AOI + event date
        │
        ▼
  backend/ (one Python package, FastAPI)
    satellite      find + download S1 (S2 optional)
    preprocessing  prepare S1 before/after stack
    inference      U-Net flood segmentation
    change         S1 log-ratio change map
    osm            pre-event OSM extraction
    damage         building / road / bridge intersection
    network        road graph + settlement connectivity
    reporting      summary.json -> one-page report
    api            serves run outputs to the frontend
        │
        │  files in data/processed/runs/<run_id>/
        ├──────────────────────┬────────────────────────┐
        ▼                      ▼                        ▼
  frontend/ dashboard     report (HTML/PDF)      evaluation/ (separate;
                                                  EMSR927 comparison)

  ml/  trains the U-Net on a Kuro Siwo subset -> checkpoint used by backend/inference
```

| Directory     | Responsibility |
|---------------|----------------|
| `backend/`    | Production pipeline and API. Reads only permitted inputs. |
| `ml/`         | Dataset loading, U-Net, training, model metrics. Produces a checkpoint. |
| `evaluation/` | **Only** place allowed to read EMSR927. Compares finished outputs to it. |
| `frontend/`   | Dashboard. Displays outputs; computes no numbers. |
| `scripts/`    | Thin CLI entry points. |
| `tests/`      | pytest tests, including one data-boundary guard test. |
| `data/`       | Local data, git-ignored. `data/eval_only/` is quarantined (see §4). |
| `docs/`       | Documentation. |

The pipeline is a plain sequence of Python functions run in one process. Outputs are files.

---

## 3. Data flow (production pipeline)

Each step writes files into one **run directory** (`data/processed/runs/<run_id>/`).

1. **Request** — AOI (GeoJSON polygon, EPSG:4326) + event date.
2. **Scene selection** (`satellite`)
   - Find Sentinel-1 GRD IW (VV+VH) scenes over the AOI before and after the event.
   - Pick one post-event scene and a pre-event scene **from the same relative orbit and
     direction**, so before/after pixels share viewing geometry.
   - If no same-track pair exists, stop with a clear message. (No automatic fallback modes.)
   - Catalogue: Copernicus Data Space Ecosystem or Microsoft Planetary Computer. **[VERIFY]** which
     is simplest, and whether an analysis-ready (terrain-corrected) S1 product is available, which
     would avoid running our own SAR processing chain.
   - Sentinel-2: optional, for visual context only if a cloud-free scene happens to exist.
3. **Preprocessing** (`preprocessing`) — calibrated backscatter in dB, terrain-corrected, both
   dates on one grid, cut into tiles. Should match Kuro Siwo's preprocessing as closely as
   practical **[VERIFY]**. Terrain correction may need Copernicus DEM; otherwise DEM stays optional.
4. **Flood segmentation** (`inference`) — U-Net on before + after VV/VH tiles → flood mask and
   flood probability. Tiles stitched back together.
5. **Change map** (`change`) — log-ratio of after vs before backscatter, simple threshold.
   Labelled "potential change", lower confidence than the flood mask.
6. **Pre-event OSM** (`osm`) — buildings, roads, bridges, settlements, hospitals as of a date
   **before** the event (e.g. Overpass `[date:...]` query). The snapshot date is recorded.
7. **Infrastructure impact** (`damage`) — GeoPandas/Shapely intersections:
   - Building intersects flood/change area → potentially affected
   - Road segment overlaps flood/change area → potentially affected (with overlapping length)
   - Bridge (`bridge=*`) within a small buffer of flood/change area → potentially affected
8. **Connectivity** (`network`) — see §6.
9. **Summary** — `summary.json` holds every count and area. It is the **only source of numbers**
   for the dashboard and report.
10. **Report** (`reporting`) — Jinja2 template filled from `summary.json`, a map image and a fixed
    limitations section.
11. **API** (`api`) — FastAPI serves the run files.

Each run also writes `manifest.json`: request, scene IDs (with dates, relative orbit,
direction), OSM snapshot date, model checkpoint name and git commit. This keeps runs reproducible
and shows that only permitted inputs were used.

---

## 4. Production vs evaluation-only separation

Production may use only Sentinel-1, Sentinel-2, Copernicus DEM and pre-event OSM (plus Kuro Siwo /
Sen1Floods11 for training). **EMSR927, other Copernicus EMS products, UNOSAT or any published
damage map, and post-event OSM edits are never production inputs.**

Kept simple:

- Reference maps live only in `data/eval_only/`.
- Only `evaluation/` reads them. `backend/` and `ml/` never import `evaluation/` and never
  reference `eval_only`. One pytest test checks this by scanning the source.
- OSM queries always carry a date before the event.
- Evaluation runs **after** a production run and only reads its output files.
- **No tuning on EMSR927.** Thresholds and model choices are fixed on training/validation data.
  Any change made after looking at EMSR927 results will be disclosed.
- The AOI and event date come from the hackathon brief, not from EMS map extents.

---

## 5. Flood segmentation (AI component)

- **Model:** standard U-Net with a pretrained ResNet encoder (`segmentation_models_pytorch`).
- **Input:** before VV, VH + after VV, VH (dB, normalised).
- **Output:** flood / not-flood (or Kuro Siwo's classes if simpler to keep) **[VERIFY]** labels.
- **Loss:** cross-entropy + Dice.
- **Reference baseline:** a threshold on the after-image / log-ratio. The U-Net must beat it.
- **Data:** a manageable Kuro Siwo subset chosen after inspecting metadata (structure, format,
  labels, size, licence, official splits). Sen1Floods11 is optional. No bulk download until we know
  exactly what we need.
- **Splits:** official splits where they exist, plus a hold-out of whole events/regions —
  never a random tile split. Any mountain/Himalayan events are kept as an unseen test set.
- **Metrics:** IoU, Dice/F1, precision, recall, flood-area error. Saved as JSON; README tables are
  copied from those files.

---

## 6. Cut-off settlement analysis

Deliberately simple:

1. Build a NetworkX graph from pre-event OSM roads (edges carry OSM id, length, bridge flag),
   over an area slightly larger than the AOI so the edge of the map does not create false
   isolation.
2. Snap each settlement (`place=village|hamlet|town|…`) and each target (`amenity=hospital`,
   `place=town|city`) to its nearest road node.
3. **Before:** shortest path from each settlement to the nearest target.
4. **After:** remove road segments and bridges marked potentially affected; recompute.
5. Status:
   - `connected` — still reachable
   - `potentially_cut_off` — reachable before, not after
   - `not_connected_before` — no path even before the event (OSM gap, not flood impact)
6. Output per settlement: name, status, nearest target, route length before/after, IDs of
   affected roads/bridges on the original route, and a short reason.

No weighted costs, travel times or routing optimisation.

---

## 7. Outputs and interfaces

```
runs/<run_id>/
  manifest.json
  flood_mask.tif            flood_probability.tif     change_mask.tif
  flood_extent.geojson
  buildings_affected.geojson  roads_affected.geojson  bridges_affected.geojson
  settlements.geojson         (with connectivity status)
  summary.json
  report.html / report.pdf
```

API: `POST /runs`, `GET /runs/{id}/summary`, `GET /runs/{id}/layers/{name}`,
`GET /runs/{id}/report`.

**Dashboard** (kept simple): large map, layer toggles, event statistics, infrastructure impact
counts, cut-off settlement list with details. All values come from `summary.json` and layer files.

**Numbers:** every number in the dashboard or report comes from pipeline outputs. No LLM is used
to generate report content.

---

## 8. Technology

| Area | Choice |
|------|--------|
| Backend | Python 3.11, FastAPI, pydantic |
| Rasters | Rasterio, NumPy, SciPy/scikit-image (only as needed) |
| Vectors | GeoPandas, Shapely, PyProj |
| Graph | NetworkX |
| ML | PyTorch, segmentation_models_pytorch |
| S1 processing | **[VERIFY]** analysis-ready product if available; otherwise a minimal processing chain |
| Report | Jinja2 → HTML (→ PDF) |
| Frontend | Next.js, TypeScript, MapLibre GL JS |
| Storage | Files on disk. No database. |
| Tests | pytest |

A library is added only when the module that uses it is implemented.

---

## 9. Assumptions

- The hackathon brief gives the target event, AOI and date (Indian Himalaya).
- A same-track Sentinel-1 before/after pair exists for the AOI.
- Kuro Siwo labels fit a flood / not-flood formulation.
- Pre-event OSM can be fetched for a specific past date.
- One Colab/Kaggle-class GPU is enough for training.
- One AOI is processed at a time.

---

## 10. Major risks

| Risk | Mitigation (simple) |
|------|---------------------|
| Our S1 preprocessing differs from Kuro Siwo's → poor transfer | Match their chain; compare value histograms |
| Steep terrain (layover/shadow, narrow valleys) | Mask unreliable pixels; state in limitations |
| Flood recedes before the next S1 pass | Show acquisition dates; state in limitations |
| No same-track pair in the time window | Check early (first task); choose window accordingly |
| SAR ambiguities (wet soil, vegetation, urban areas) | Before/after input; state in limitations |
| Kuro Siwo size / licence / access | Inspect metadata first; train on a subset |
| Few Himalayan scenes in training data | Hold out any mountain events; report honestly |
| Incomplete rural OSM | Report as limitation; `not_connected_before` status |
| Overlap ≠ damage | "Potentially affected" wording everywhere |
| Time budget | Strict priority order (§1.1); bonus last |

---

## 11. Limitations to state in the report

Sentinel revisit frequency; cloud limits on optical imagery; SAR ambiguities and terrain effects;
OSM completeness and accuracy; uncertainty in impact estimates; spatial overlap cannot confirm
structural damage; simple connectivity model; prototype, non-operational.
