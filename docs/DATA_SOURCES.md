# VantageQ — Data Sources

Findings from the data inspection on **2026-10-03**. Only metadata, repository files,
documentation, catalogue queries and one ~4 MB range-read of a single Kuro Siwo shard were used.
No bulk data was downloaded. No EMSR927 or other published damage map was accessed.

Items marked **[VERIFY]** are still open (see §9).

---

## 1. Kuro Siwo (training dataset)

Sources: GitHub `Orion-AI-Lab/KuroSiwo` (code), Hugging Face `orion-ai-lab/Kuro-Siwo-Webdataset`
and `orion-ai-lab/Kuro-Siwo-GeoTIFFs` (data), NeurIPS 2024 Datasets & Benchmarks paper
(arXiv 2311.12056v3).

### 1.1 Content

- 43 flood events, 2015–2022, six continents. Labelled set: **67,490 time series**
  (202,470 SAR images). Large unlabelled set (533,847 time series) — not needed.
- Each sample = **3 Sentinel-1 acquisitions**: 2 pre-event (`sec1`, `sec2`) + 1 post-event
  (`flood`), plus a DEM tile, label mask and valid-pixel mask.
- All three acquisitions share the same orbit direction (paper: "belong to either the descending
  or ascending imaging geometry"). In the inspected sample the two pre-event images were 12 days
  apart (2020-04-29, 2020-05-11) and the post-event image months later (2020-10-14).
- Labels: manual photo-interpretation by SAR experts. For events that had Copernicus EMS products,
  the EMS shapefiles were the starting point for annotation. All Kuro Siwo events are from
  2015–2022 (activation IDs ≤ 567 plus non-EMS IDs `1111xxx`), so **EMSR927 is not part of it**.
  Kuro Siwo is an explicitly listed training dataset, so using it for training is permitted.

### 1.2 Sample format (verified from one sample)

| Field | Shape / type | Notes |
|-------|--------------|-------|
| `flood_vv`, `flood_vh` | (1, 224, 224) float32 | Post-event VV / VH |
| `sec1_vv`, `sec1_vh`, `sec2_vv`, `sec2_vh` | (1, 224, 224) float32 | Two pre-event images |
| `dem` | (1, 224, 224) float32 | Elevation (m) |
| `mask` | (1, 224, 224) float32 | **0 = no water, 1 = permanent water, 2 = flood** |
| `valid_mask` | (1, 224, 224) float32 | 1 = valid pixel |
| `info.json` | JSON | Event ID, AOI ID, S1 product UUIDs and dates, `pflood`, `pwater`, tile geometry (EPSG:3857) |

- Patch size **224 × 224 px** at ~10 m.
- Backscatter is **linear σ⁰ (not dB)**: sample VV median ≈ 0.09, VH ≈ 0.02.
- Official code clamps inputs to `[0, 0.15]` and normalises with
  `mean = [0.0953, 0.0264]`, `std = [0.0427, 0.0215]` (VV, VH).
- GeoTIFF version: per-tile files named `MS1_IVV…`, `SL1_IVV…`, `SL2_IVV…`, `MK0_MLU…` (label),
  `MK0_MNA…` (valid mask), `MK0_DEM…`.

### 1.3 Size

| Hugging Face repo / folder | Size |
|----------------------------|------|
| Webdataset `train_GRD` (5 shards, 3.1–10.9 GB each) | **46.8 GB** |
| Webdataset `test_GRD` (12 shards) | 122.8 GB |
| Webdataset `train_SLC` | 98.0 GB |
| Webdataset `unlabelled/` | 1,514.5 GB |
| GeoTIFFs `GRD/` (35 tars) | 699.1 GB |
| GeoTIFFs `SLC/` | 316.3 GB |

About 1.8 MB per labelled GRD sample. The full dataset is far beyond our needs.

### 1.4 Official split and protocol

- **Event-held-out.** 10 test events chosen across continents/climate zones; the other 33 events
  are train/validation. No sample overlap between event groups.
- From `configs/train/data_config.json` (activation IDs):
  - train: `130, 470, 555, 118, 174, 324, 421, 554, 427, 518, 502, 498, 497, 496, 492, 147, 267,
    273, 275, 417, 567, 1111011, 1111004, 1111009, 1111010, 1111006, 1111005`
  - val: `514, 559, 279, 520, 437, 1111003, 1111008`
  - test: `321, 561, 445, 562, 411, 1111002, 277, 1111007, 205, 1111013`
- Metrics: per-class F1 (no-water, permanent water, flood), mean IoU, and F1 for a merged binary
  "water" class.
- Reported GRD reference result, U-Net / ResNet-50 with all 3 acquisitions: F1-flood 80.12 %,
  F1-permanent-water 78.24 %, mIoU 76.20 %, F1-water 83.85 %.
- Official baseline config: `segmentation_models_pytorch` U-Net, ResNet-18 encoder (ImageNet
  weights), Adam lr 1e-3, cosine schedule, cross-entropy, 3 classes, DEM/slope off by default.

### 1.5 GRD vs SLC

- **GRD:** SNAP-processed, terrain-corrected σ⁰ amplitude. Matches what we can produce
  operationally. **Use GRD.**
- **SLC:** complex data (phase + amplitude), heavier processing, coherence-based. Out of scope.

### 1.6 GRD preprocessing chain (from `configs/grd_preprocessing.xml`, ESA SNAP graph)

1. Apply-Orbit-File (Sentinel Precise)
2. Subset
3. ThermalNoiseRemoval
4. Remove-GRD-Border-Noise (borderLimit 500, trimThreshold 50)
5. Land-Sea-Mask
6. Calibration → **σ⁰, linear** (`outputImageScaleInDb = false`)
7. Speckle-Filter → **Lee Sigma**, 7×7 window, 3×3 target, sigma 0.9
8. Terrain-Correction → **SRTM 1 Sec** DEM, bilinear resampling, **10 m**, no radiometric
   normalisation (plain σ⁰, not γ⁰ RTC), output CRS **WGS 84 / Pseudo-Mercator (EPSG:3857)**
9. Write GeoTIFF

### 1.7 Can we reproduce it?

Yes, closely enough:

- **Exact option:** run the same SNAP graph (`gpt grd_preprocessing.xml`) on our own GRD scenes.
  Faithful, but SNAP is a heavy Java install and slow.
- **Simple option (recommended first):** Sentinel Hub Process API on CDSE (§2.3) with
  `backCoeff = SIGMA0_ELLIPSOID`, `orthorectify = true`, `demInstance = COPERNICUS_30`,
  `speckleFilter = LEE`, linear output, 10 m. Differences: Lee instead of Lee Sigma, Copernicus DEM
  instead of SRTM. Acceptable for a prototype **if** our VV/VH value distributions roughly match
  Kuro Siwo's statistics (VV mean ≈ 0.095, VH ≈ 0.026 before clamping). This check is required
  before inference. If it fails, switch to the SNAP graph.

### 1.8 Is a practical subset feasible?

Yes:

- Webdataset shards can be **streamed over HTTP** and filtered on `info.json` (event ID, `pflood`)
  without storing the whole shard.
- Initial target: one or two `train_GRD` shards (~3–11 GB each) plus one `test_GRD` shard, keeping
  flood-positive tiles and a share of negatives. Several thousand tiles are enough for a ResNet-18
  U-Net baseline on one Colab/Kaggle-class GPU.
- Which events each shard contains is not documented **[VERIFY]**. A small streaming script can list
  `actid` per shard before deciding.

### 1.9 Himalayan generalisation

Kuro Siwo has **no Himalayan or steep-mountain events**. The "Nepal" event (`1111007`, AOI
"Patna") is a lowland Terai/Gangetic-plain event and sits in the official **test** set. Therefore:

- We can measure event-held-out generalisation (official test events) but **not**
  Himalayan-specific performance before the final EMSR927 evaluation.
- Event `1111007` is the closest proxy and should be reported separately.
- Steep-terrain effects (layover/shadow, narrow valleys) are untested by the training data.

### 1.10 Licence and attribution

- Dataset: **CC BY 4.0** (Hugging Face metadata `license:cc-by-4.0`; README says "CC BY").
  Attribution: cite the NeurIPS 2024 paper (Bountos et al., "Kuro Siwo: 33 billion m² under the
  water…").
- Code repository: **MIT**. If we copy code, keep the MIT notice.

---

## 2. Sentinel-1 acquisition

### 2.1 Search: CDSE STAC (public, no authentication needed for search)

- Endpoint: `https://stac.dataspace.copernicus.eu/v1/`, collection **`sentinel-1-grd`**.
- Queryable properties include `sat:relative_orbit`, `sat:orbit_state`, `platform`,
  `product:type`, `sar:instrument_mode`, `sar:polarizations`, `datetime`.
- Each item also carries `sat:absolute_orbit`, start/end time, and VV/VH COG assets.
- Example:
  `GET /v1/search?collections=sentinel-1-grd&bbox=<w,s,e,n>&datetime=<start>/<end>`
- The CDSE OData catalogue (`https://catalogue.dataspace.copernicus.eu/odata/v1/Products`) gives
  the same information (`relativeOrbitNumber`, `orbitDirection`, footprint) and was used to check
  AOI coverage.

### 2.2 Selecting compatible before/after scenes

Simple rule:

1. Search the AOI from about −36 days to +14 days around the event.
2. Group by (`sat:relative_orbit`, `sat:orbit_state`) and keep scenes whose footprint covers the AOI.
3. For each track, take the **first post-event scene** and the **two most recent pre-event
   scenes** (Kuro Siwo format).
4. Choose the track whose post-event scene is closest to the event date; keep a second track as
   a manual alternative.

### 2.3 Download / processing

- STAC asset `href`s are `s3://eodata/...` and need CDSE S3 credentials to download.
- **Simplest route:** Sentinel Hub **Process API on CDSE**. It returns a clipped, orthorectified
  σ⁰ GeoTIFF for an AOI and time range, so we avoid downloading full scenes (~1 GB each) and running
  SNAP. Relevant options: `backCoeff` (`SIGMA0_ELLIPSOID`), `orthorectify`, `demInstance`
  (`COPERNICUS_30`), `speckleFilter` (`LEE`, window size), `upsampling`, resolution `HIGH` (10 m).
  Extra bands: `shadowMask`, `localIncidenceAngle`, `dataMask` — useful as a layover/shadow mask.
  It requires a free CDSE account and OAuth client. Monthly quota limits **[VERIFY]**.
- To pin an exact acquisition, filter `orbitDirection` and use a narrow `timeRange` around the
  scene time (Sentinel Hub's data filter does not expose relative orbit directly; the STAC search
  does).
- **Alternative:** download the `IW_GRDH_1S-COG` product via CDSE S3 and run Kuro Siwo's SNAP
  graph.

### 2.4 Constellation status (observed)

Over the Trishuli area, only **one satellite** acquires at a time: **Sentinel-1D** in 2026,
Sentinel-1A in the same window of 2025. Each track repeats every **12 days**, giving about three
acquisitions per 12 days across three tracks.

---

## 3. Trishuli case study — data availability (catalogue check only)

- AOI used for the check: approximate bounding box of the Trishuli valley from Bidur to
  Rasuwagadhi, **85.10–85.42 E, 27.85–28.30 N**. This is our approximation; the final AOI must come
  from the hackathon brief **[VERIFY]**.
- Event date: **2026-08-26** (time of day unknown).

Sentinel-1 GRD (IW, VV+VH, all S1D) intersecting the AOI, August–September 2026:

| Track (relative orbit) | Direction | AOI coverage | Pre-event scenes | First post-event | Post − event |
|---|---|---|---|---|---|
| **85** | Ascending (≈12:21 UTC) | **100 %** | 2026-08-04, **2026-08-16** | **2026-08-28** | **+2 days** |
| **19** | Descending (≈00:18 UTC) | **100 %** | 2026-08-12, **2026-08-24** | 2026-09-05 | +10 days |
| 121 | Descending (≈00:10 UTC) | ~32 % | 2026-08-07, 2026-08-19 | 2026-08-31 | +5 days |

Product names (primary pair, track 85 ascending):

- Pre 1: `S1D_IW_GRDH_1SDV_20260804T122140_20260804T122205_003976_00737A_*`
- Pre 2: `S1D_IW_GRDH_1SDV_20260816T122141_20260816T122206_004151_007980_*`
- Post: `S1D_IW_GRDH_1SDV_20260828T122141_20260828T122206_004326_007FA4_*`

Assessment:

- **Feasible.** Track 85 gives a full-coverage, same-track triplet in Kuro Siwo format, with the
  post-event image about 2 days after the event. All products are online.
- Track 19 (descending) gives an independent full-coverage triplet with the opposite look
  direction. That helps in steep valleys, where layover/shadow differ between geometries, but its
  post image is 10 days later, so receding water may be missed.
- Caveats: the flood may have peaked and receded within two days; narrow, steep valleys are hard
  for 10 m SAR; Kuro Siwo contains no comparable terrain (§1.9).

---

## 4. Sentinel-2 (optional, secondary)

- CDSE STAC collection `sentinel-2-l2a`. Two tiles overlap the AOI (orbits R076 and R119).
- August–September 2026 tile-level cloud cover was mostly 50–99 %. Clearer scenes: **pre-event**
  2026-08-11/12 (13–28 % on some tiles); **post-event** none below ~38 % until 2026-09-20/21
  (16 %) and 2026-09-28 (2–8 %), i.e. 3–5 weeks after the event.
- Tile-level cloud is not AOI-level cloud; the SCL mask would be needed per scene.
- Conclusion: S2 is not usable for the main analysis window. Keep it **optional**, for visual
  context or late debris/scar mapping only.
- Licence: free and open under Copernicus terms; attribution as in §7.

---

## 5. Copernicus DEM

- GLO-30 (30 m) available globally; Sentinel Hub uses it internally (`COPERNICUS_30`) for
  orthorectification, so the core pipeline does **not** need to download a DEM.
- Separate DEM download is only needed for the bonus flow-path tracing or for slope context.
  Sources: CDSE (`COP-DEM` collections) or the public AWS Open Data bucket **[VERIFY]**.
- Kuro Siwo's terrain correction used SRTM 1″. The difference is minor at 10 m output.

---

## 6. OpenStreetMap (pre-event)

### 6.1 ohsome API (originally recommended for the historical snapshot)

> **Update 2026-10-04:** ohsome v1 extraction endpoints (`/elements/geometry`, `/elements/centroid`)
> now return HTTP 403. The v2 API (`https://api.heigit.org/ohsome-api/v2/`, extraction at
> `POST /extraction/features`, GeoParquet output) requires a free API key. Anonymous
> `/v1/elements/count` still works. The impact analysis therefore uses an **Overpass attic query**
> (§6.2) at 2026-08-25T00:00:00Z; see `docs/TRISHULI_IMPACT_ANALYSIS.md`.

- `https://api.ohsome.org/v1/` returns OSM elements as they existed at a given `time`.
- The current database ends at **2026-07-27T09:00Z**. That is **before** the event, so any ohsome
  snapshot is automatically pre-event. We will use **2026-07-27** as the snapshot date and record
  it in the run manifest.
- Endpoint for features: `/elements/geometry` (with `bboxes`, `time`, `filter`, `properties=tags`).

Counts at 2026-07-27 in the AOI box:

| Feature | Count |
|---------|-------|
| Buildings (polygons) | 103,500 |
| Highway ways (main classes + track/service), total length | ≈ 2,056 km |
| Highway ways tagged `bridge=yes` | 202 |
| Settlements (`place` = city/town/village/hamlet) | 158 |
| Towns/cities, wider box incl. Kathmandu | 30 |
| Hospitals/clinics, wider box incl. Kathmandu | 797 |

The wider box includes the Kathmandu valley, so hospital counts are dominated by Kathmandu. The
connectivity analysis needs a road graph larger than the flood AOI to reach real hospitals/towns.

### 6.2 Overpass API (alternative)

The Overpass `[date:"YYYY-MM-DDThh:mm:ssZ"]` attic query also returns historical data and could
provide a snapshot closer to the event (e.g. 2026-08-25). The public instance was reachable. It is
heavier on large areas; use it only if ohsome's July snapshot proves insufficient.

### 6.3 Licence

ODbL 1.0. Attribution: "© OpenStreetMap contributors". Derived databases must also be ODbL.

---

## 7. Licensing and attribution summary

| Source | Licence | Required attribution |
|--------|---------|----------------------|
| Sentinel-1 / Sentinel-2 | Copernicus free & open data | "Contains modified Copernicus Sentinel data 2026, processed by the VantageQ team" |
| Copernicus DEM GLO-30 | Copernicus DEM GLO-30 licence (ESA), Article 6(b) | "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved"; dataset citation: https://doi.org/10.5270/ESA-c5d3d65 |
| Kuro Siwo dataset | CC BY 4.0 | Cite Bountos et al., NeurIPS 2024 |
| Kuro Siwo code | MIT | Keep copyright notice if code is reused |
| OpenStreetMap / ohsome | ODbL 1.0 | "© OpenStreetMap contributors" |
| Sen1Floods11 (optional) | **[VERIFY]** | Cite Bonafilia et al., CVPR-W 2020 |

Copernicus DEM notice verified on 2026-10-04 against the Copernicus DEM GLO-30 licence (ESA),
Article 6, as linked from the CDSE/Sentinel Hub DEM documentation, and the CDSE Copernicus DEM
collection page (https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).
Article 6(a) ("© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under
COPERNICUS by the European Union and ESA; all rights reserved") applies when distributing the DEM
itself. VantageQ does not distribute the DEM: COPERNICUS_30 is used inside Sentinel Hub to
orthorectify Sentinel-1, and results produced with it are shown. The Article 6(b) "produced using"
notice for adapted/modified data is therefore used.

---

## 8. Evaluation-only data

EMSR927 has **not** been accessed. It will be obtained later, stored in `data/eval_only/`, and
used only by `evaluation/` after a production run.

---

## 9. Unresolved issues

1. **Official AOI** for the Trishuli case study (current bbox is our approximation).
2. **Sentinel Hub quotas** on CDSE for our account (processing units / requests per month).
3. **Radiometric match** between Sentinel Hub σ⁰ and Kuro Siwo σ⁰ — histogram check on the actual
   scenes.
4. **Event content per webdataset shard** — needed to pick a subset that respects the official
   event split.
5. **Himalayan validation data:** none in Kuro Siwo. Options: accept and report it, or hand-label
   a few small tiles from our own permitted imagery (not from EMS maps).
6. **Bridge coverage:** 202 tagged road bridges; completeness unknown.
7. **Sen1Floods11 licence** (only if used). (Copernicus DEM attribution wording: resolved 2026-10-04, see §7.)
