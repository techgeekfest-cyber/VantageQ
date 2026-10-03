# VantageQ

Satellite flood-impact intelligence prototype for the **IIT Mandi Multimodal AI Hackathon 2026,
Track B — Mapping Flood Damage from Space**.

> **Status:** working research prototype. Sentinel-1 retrieval, a frozen flood-segmentation model,
> Trishuli inference, pre-event OSM impact and connectivity analysis, a static map dashboard and a
> one-page situation report are implemented. The EMSR927 comparison (evaluation-only) is not built
> yet. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## What it does

Given an area of interest (AOI) and a flood event date, the current prototype:

1. Finds same-track Sentinel-1 GRD scenes before and after the event (Copernicus Data Space) and
   predicts flood extent with a U-Net trained on the Kuro Siwo dataset
2. Identifies potentially affected roads, bridges and buildings by overlap with the predicted flood,
   using a historical (pre-event) OpenStreetMap snapshot
3. Flags settlements potentially cut off from the main road network in a simplified road graph
4. Shows the results on an interactive map dashboard
5. Generates a one-page situation report from the same structured outputs
6. Records reproducible model evaluation on held-out Kuro Siwo events (`docs/TEST_EVALUATION.md`)

Not yet built: before/after change (debris) mapping as a product layer, routing to specific
towns/hospitals, and the evaluation-only EMSR927 comparison.

## Data rules

Production inputs are limited to **Sentinel-1, Sentinel-2, Copernicus DEM and pre-event
OpenStreetMap**. Training uses listed datasets (Kuro Siwo; optionally Sen1Floods11).

Copernicus EMS products (including **EMSR927**), UNOSAT and other published damage maps, and
post-event OSM edits are **never** production inputs. EMSR927 is used only in the separate
`evaluation/` workflow (not yet built), after the system has produced its own output.

## Repository layout

```
backend/      Python analysis pipeline (Sentinel-1 retrieval, impact, connectivity)
ml/           Flood segmentation: datasets, model, training, model evaluation
evaluation/   Reserved for the evaluation-only EMSR927 comparison (isolated; not yet built)
frontend/     Next.js + TypeScript + MapLibre dashboard
scripts/      CLI entry points
data/         Local data (git-ignored); data/eval_only/ is quarantined
docs/         Architecture, data sources, experiment records; docs/output/ = situation report
tests/        pytest suite (frontend tests live in frontend/src)
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

## Situation Report

One-page PDF (A4): [`docs/output/VantageQ_Trishuli_Situation_Report.pdf`](docs/output/VantageQ_Trishuli_Situation_Report.pdf)
(HTML version alongside). It is generated deterministically from the same structured outputs as the
dashboard; every number is read from the analysis JSON:

```bash
.venv/bin/python scripts/generate_situation_report.py
```

Regeneration needs the (git-ignored) pipeline outputs in `data/processed/trishuli/`; the committed
PDF can be read without them.

## Limitations (summary)

This is an educational, non-operational prototype. Results are limited by Sentinel revisit
frequency, cloud cover for optical imagery, SAR ambiguities (terrain, vegetation, urban areas),
OSM completeness and accuracy, and uncertainty in damage estimation. Spatial overlap with flood
extent does **not** confirm structural damage; outputs are reported as *potentially affected*.
Connectivity analysis depends on OSM road data and simple blocking rules. Limitations are listed
in the situation report, on the dashboard and in `docs/TRISHULI_IMPACT_ANALYSIS.md`.

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
