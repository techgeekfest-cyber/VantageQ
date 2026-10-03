# VantageQ

Satellite flood-impact intelligence prototype for the **IIT Mandi Multimodal AI Hackathon 2026,
Track B — Mapping Flood Damage from Space**.

> **Status:** working research prototype. Sentinel-1 retrieval, a frozen flood-segmentation model,
> Trishuli inference, pre-event OSM impact and connectivity analysis, and a static map dashboard are
> implemented. The situation report and EMSR927 evaluation are not built yet. See
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## What it will do

Given an area of interest (AOI) and a flood event date, VantageQ will:

1. Map flood extent (Sentinel-1 SAR flood segmentation) and potential change/debris areas
2. Estimate potentially affected buildings, roads and bridges using pre-event OpenStreetMap
3. Identify settlements potentially cut off from the nearest town or hospital
4. Show results on an interactive map dashboard
5. Generate a one-page situation report
6. Provide reproducible evaluation results

## Data rules

Production inputs are limited to **Sentinel-1, Sentinel-2, Copernicus DEM and pre-event
OpenStreetMap**. Training uses listed datasets (Kuro Siwo; optionally Sen1Floods11).

Copernicus EMS products (including **EMSR927**), UNOSAT and other published damage maps, and
post-event OSM edits are **never** production inputs. EMSR927 is used only in the separate
`evaluation/` workflow, after the system has produced its own output.

## Repository layout

```
backend/      Python analysis pipeline (Sentinel-1 retrieval, impact, connectivity)
ml/           Flood segmentation: datasets, model, training, model evaluation
evaluation/   Evaluation-only comparison against EMSR927 (isolated)
frontend/     Next.js + TypeScript + MapLibre dashboard
scripts/      CLI entry points
data/         Local data (git-ignored); data/eval_only/ is quarantined
docs/         Architecture and project documentation
tests/        Tests, including data-boundary guard tests
```

## Current Demo

**Trishuli Valley, Nepal: flood event of 26 August 2026** (one ~4.9 × 4.4 km sub-AOI near Betrawati).

- **Predicted flood:** a frozen U-Net trained on the Kuro Siwo dataset, applied to Sentinel-1 VV/VH
  (post-event 2026-08-28, pre-event 2026-08-16 and 2026-08-04, same track). Predicted flood extent:
  0.120 km². The prediction is **not validated** against ground truth for Trishuli; on held-out
  Kuro Siwo test events the model reached flood IoU 0.31 (`docs/TEST_EVALUATION.md`).
- **Infrastructure:** historical OpenStreetMap snapshot from 2026-08-25 (pre-event). 3 road ways
  potentially affected (189 m of road inside the predicted flood), 2 bridges intersecting the
  predicted flood, 36 buildings potentially affected.
- **Connectivity:** in a simplified road graph, 2 of 3 mapped settlements (Betrawati, Bhainse) are
  **potentially cut off**, i.e. disconnected after removing flood-affected road edges. This is a graph
  indicator, not confirmed real-world isolation; footpaths and roads outside the AOI are not in the graph.

Details: `docs/TRISHULI_INFERENCE.md`, `docs/TRISHULI_IMPACT_ANALYSIS.md`.

## Running the Dashboard

```bash
cd frontend
npm install
npm run dev
```

Then open <http://localhost:3000>.

- The dashboard is a static Next.js + MapLibre app with no backend or database. Its analytical
  layers and every number it shows come from a **small committed Trishuli demo snapshot**
  (`frontend/public/demo/trishuli/`, ~100 KB), built from the pipeline outputs by
  `scripts/build_demo_snapshot.py`.
- Large raw datasets (Sentinel-1 rasters, Kuro Siwo samples, raw OSM responses), model checkpoints
  and full analysis outputs remain **git-ignored** and are not needed to run the dashboard.
- The basemap uses **live OpenStreetMap tiles** for orientation only (internet required). They show
  current OSM, whereas the analysis uses the 2026-08-25 historical snapshot.

More: `frontend/README.md`.

## Limitations (summary)

This is an educational, non-operational prototype. Results are limited by Sentinel revisit
frequency, cloud cover for optical imagery, SAR ambiguities (terrain, vegetation, urban areas),
OSM completeness and accuracy, and uncertainty in damage estimation. Spatial overlap with flood
extent does **not** confirm structural damage; outputs are reported as *potentially affected*.
Connectivity analysis depends on OSM road data and simple blocking rules. A full limitations
section will accompany the situation report.

## Responsible use

Built for education and research. It does not use victim imagery and must not be used as a basis
for operational decisions without validation by qualified responders.

## Attribution

Contains modified Copernicus Sentinel data (processed by the VantageQ team).
Map data © OpenStreetMap contributors, available under the ODbL.
Copernicus DEM (used to orthorectify Sentinel-1): produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014
and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA;
all rights reserved. Kuro Siwo dataset (CC BY 4.0): Bountos et al., NeurIPS 2024. Full list and
licences: `docs/DATA_SOURCES.md` §7.

## AI-assisted development disclosure

This project was developed with the help of an AI coding assistant (Claude Code, using Claude
Opus 5.5 by Anthropic) for tasks such as scaffolding, drafting documentation and writing code.
The project authors directed the design, reviewed the generated material, and are responsible for
understanding, testing and validating all submitted code, results and claims.
