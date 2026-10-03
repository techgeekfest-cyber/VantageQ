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
   - Search with the **CDSE STAC API** (`sentinel-1-grd`; public, no auth), which exposes
     `sat:relative_orbit` and `sat:orbit_state`.
   - Per track (same relative orbit **and** direction), keep scenes covering the AOI; take the
     **first post-event scene and the two latest pre-event scenes** (Kuro Siwo's 3-image format).
     Use the track whose post-event scene is closest to the event.
   - If no same-track triplet exists, stop with a clear message. (No automatic fallback modes.)
   - Sentinel-2: optional, for visual context only if a cloud-free scene happens to exist.
   - See `docs/DATA_SOURCES.md` §2–3.
3. **Preprocessing** (`preprocessing`) — fetch AOI-clipped, orthorectified **linear σ⁰** VV/VH at
   10 m from the **Sentinel Hub Process API on CDSE** (`SIGMA0_ELLIPSOID`, `COPERNICUS_30` DEM,
   Lee speckle filter), plus its `shadowMask`/`dataMask`. This approximates Kuro Siwo's SNAP chain
   (σ⁰ linear, Lee Sigma, SRTM terrain correction). Then clamp to [0, 0.15], normalise with Kuro
   Siwo's mean/std, and tile at 224 px. **Required check:** our σ⁰ histograms must roughly match
   Kuro Siwo's statistics; if not, run Kuro Siwo's own SNAP graph instead. No separate DEM download
   is needed for this step.
4. **Flood segmentation** (`inference`) — U-Net on pre1 + pre2 + post VV/VH tiles → classes
   no-water / permanent water / flood, plus flood probability. Tiles stitched back together;
   shadow/no-data pixels marked as no-data, not dry.
5. **Change map** (`change`) — log-ratio of after vs before backscatter, simple threshold.
   Labelled "potential change", lower confidence than the flood mask.
6. **Pre-event OSM** (`osm`) — buildings, roads, bridges, settlements, hospitals as of a date
   **before** the event, via the **ohsome API** `time` parameter (Overpass `[date:...]` as
   alternative). The snapshot date is recorded. Roads are fetched for an area larger than the AOI
   so routes can reach real towns/hospitals.
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
- **Input:** 6 channels: pre1 VV/VH, pre2 VV/VH, post VV/VH. **Linear σ⁰**, clamped to
  [0, 0.15], normalised with Kuro Siwo's mean/std. 224 × 224 tiles. DEM channel off (as in the
  official baseline).
- **Output:** Kuro Siwo's 3 classes (0 no water, 1 permanent water, 2 flood). The flood map uses
  class 2; permanent water is shown separately.
- **Loss:** cross-entropy (official default); add Dice only if flood recall is poor.
- **Reference baseline:** a threshold on the after-image / log-ratio. The U-Net must beat it.
- **Starting point:** ResNet-18 encoder, Adam lr 1e-3, cosine schedule — Kuro Siwo's official
  U-Net config, so our numbers are comparable with the paper.
- **Data:** Kuro Siwo GRD webdataset, **streamed** from Hugging Face and filtered by event — one or
  two `train_GRD` shards plus one `test_GRD` shard (tens of GB at most, not the ~1–2 TB full
  release). Sen1Floods11 is not needed initially. See `docs/DATA_SOURCES.md` §1.
- **Splits:** Kuro Siwo's official **event-held-out** train/val/test activations. Never a random
  tile split.
- **Himalayan generalisation:** Kuro Siwo contains **no Himalayan/steep-terrain events**. We report
  the official test events, event `1111007` (Nepal lowlands) separately, and the final Trishuli
  comparison against EMSR927 (evaluation-only). Steep-terrain performance is unknown until then.
- **Metrics:** per-class F1, mIoU and binary-water F1 (Kuro Siwo protocol), plus flood-class IoU,
  precision, recall and flood-area error. Saved as JSON; README tables are copied from those files.

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
| S1 search | CDSE STAC API (`sentinel-1-grd`) |
| S1 processing | Sentinel Hub Process API on CDSE (orthorectified σ⁰); SNAP + Kuro Siwo graph only if the radiometric check fails |
| OSM history | ohsome API (Overpass attic as alternative) |
| Report | Jinja2 → HTML (→ PDF) |
| Frontend | Next.js, TypeScript, MapLibre GL JS |
| Storage | Files on disk. No database. |
| Tests | pytest |

A library is added only when the module that uses it is implemented.

---

## 9. Assumptions

- The hackathon brief gives the target event, AOI and date (Indian Himalaya).
- A same-track Sentinel-1 before/after pair exists for the AOI.
- Kuro Siwo labels (no water / permanent water / flood) are used as-is.
- Pre-event OSM comes from ohsome; its current snapshot (2026-07-27) is pre-event for Trishuli.
- A free CDSE account with Sentinel Hub access has enough quota for a few AOI requests.
- One Colab/Kaggle-class GPU is enough for training.
- One AOI is processed at a time.

---

## 10. Major risks

| Risk | Mitigation (simple) |
|------|---------------------|
| Our S1 preprocessing differs from Kuro Siwo's → poor transfer | Match their chain; compare value histograms |
| Steep terrain (layover/shadow, narrow valleys) | Mask unreliable pixels; state in limitations |
| Flood recedes before the next S1 pass | Pick the closest post-event track (+2 days for Trishuli); show acquisition dates |
| No same-track pair in the time window | Check early (first task); choose window accordingly |
| SAR ambiguities (wet soil, vegetation, urban areas) | Before/after input; state in limitations |
| Kuro Siwo size / licence / access | Inspect metadata first; train on a subset |
| No Himalayan scenes in Kuro Siwo | Report official test + Nepal-lowland event; state clearly; EMSR927 is the only mountain check |
| Incomplete rural OSM | Report as limitation; `not_connected_before` status |
| Overlap ≠ damage | "Potentially affected" wording everywhere |
| Time budget | Strict priority order (§1.1); bonus last |

---

## 11. Limitations to state in the report

Sentinel revisit frequency; cloud limits on optical imagery; SAR ambiguities and terrain effects;
OSM completeness and accuracy; uncertainty in impact estimates; spatial overlap cannot confirm
structural damage; simple connectivity model; prototype, non-operational.
