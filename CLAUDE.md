# CLAUDE.md

Guidance for AI coding assistants working in this repository. Read `docs/ARCHITECTURE.md` first.

## Project

VantageQ — flood-impact mapping prototype (IIT Mandi Hackathon 2026, Track B). AOI + event date →
flood extent, potentially affected infrastructure, cut-off settlements, dashboard, one-page report,
evaluation. AI component: Sentinel-1 SAR flood segmentation (U-Net).

## Scope: practical hackathon prototype — do NOT overengineer

Goal: complete, reliable, demoable, good enough for round two. Prefer a few reliable features over
many partial ones. Keep everything simple and explainable.

Priority order (finish earlier items before later ones):

1. Sentinel-1 before/after data pipeline
2. Flood segmentation AI
3. Flood/change map
4. OSM infrastructure impact
5. Cut-off settlements (simple road graph)
6. Dashboard
7. One-page situation report
8. EMSR927 evaluation
9. Bonus DEM flood-path tracing — only if 1–8 work

Do:
- Standard U-Net, trained on a manageable Kuro Siwo subset
- Plain GeoPandas/Shapely intersections for impact
- NetworkX shortest paths for connectivity
- One FastAPI backend, one Next.js frontend, files on disk
- Sentinel-2 and Copernicus DEM optional/secondary until the core works

Don't:
- Complex model architectures
- Bulk dataset downloads before requirements are confirmed
- Microservices, databases (unless clearly necessary), task queues
- Agents/chatbots, or LLM-generated report content
- Uncertainty frameworks, advanced routing/optimisation
- Elaborate fallback systems not needed for the live demo

Dashboard = big map, layer toggles, event stats, impact counts, cut-off settlement list. Nothing more
until that works.

## Hard rules (do not violate)

1. **Permitted production inputs only:** Sentinel-1, Sentinel-2, Copernicus DEM, OSM from *before*
   the event date. Training may use Kuro Siwo and (optionally) Sen1Floods11.
2. **Never** use Copernicus EMS maps (incl. EMSR927), UNOSAT or any published damage map, or
   post-event OSM edits in `backend/` or `ml/`. These live only in `data/eval_only/` and are read
   only by `evaluation/`.
3. `backend/` and `ml/` must never import from `evaluation/` or reference `data/eval_only/`.
4. Never tune thresholds, models or parameters using EMSR927 results.
5. OSM queries must be dated strictly before the event date.
6. Sentinel-1 before/after comparison only within the **same relative orbit and direction**.
7. Every number in the dashboard or report comes from run outputs (`summary.json` and run files).
   No hard-coded or invented numbers.
8. Wording: "potentially affected" / "potentially cut off". Never "destroyed" or "confirmed damaged".
9. No fake/demo data presented as real results. Test fixtures must be clearly synthetic and kept
   under `tests/`.

## Working conventions

- Ask before downloading large datasets or starting training.
- Core logic in `backend/` / `ml/`; thin CLIs in `scripts/`.
- Every pipeline run writes `manifest.json` with input provenance.
- Add a library only when the module using it is implemented.
- Python 3.11, type hints, pytest.
- Update `docs/ARCHITECTURE.md` when a design decision changes.
