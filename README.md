# VantageQ

Satellite flood-impact intelligence prototype for the **IIT Mandi Multimodal AI Hackathon 2026,
Track B — Mapping Flood Damage from Space**.

> **Status:** repository skeleton and architecture only. No pipeline, model or dashboard is
> implemented yet. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

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
backend/      Python + FastAPI production pipeline and API
ml/           Flood segmentation: datasets, model, training, model evaluation
evaluation/   Evaluation-only comparison against EMSR927 (isolated)
frontend/     Next.js + TypeScript + MapLibre dashboard
scripts/      CLI entry points
data/         Local data (git-ignored); data/eval_only/ is quarantined
docs/         Architecture and project documentation
tests/        Tests, including data-boundary guard tests
```

## Getting started

Not yet runnable. Setup instructions will be added as modules are implemented.

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
Dataset and DEM attributions will be added as each source is integrated (see `docs/`).

## AI-assisted development disclosure

This project was developed with the help of an AI coding assistant (Claude Code, using Claude
Opus 5.5 by Anthropic) for tasks such as scaffolding, drafting documentation and writing code.
The project authors directed the design, reviewed the generated material, and are responsible for
understanding, testing and validating all submitted code, results and claims.
