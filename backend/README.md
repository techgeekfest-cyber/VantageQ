# backend/

Python + FastAPI production pipeline. Not yet implemented.

Planned package layout (see `docs/ARCHITECTURE.md` §2–3):

```
backend/vantageq/
  satellite/       scene discovery + acquisition (S1, S2, Copernicus DEM)
  preprocessing/   calibration, terrain correction, co-registration, tiling
  inference/       flood segmentation inference (loads checkpoint from ml/)
  change/          before/after change analysis
  osm/             time-bounded pre-event OSM extraction
  damage/          building / road / bridge overlay
  network/         road graph + settlement connectivity
  dem/             (bonus) downstream flow-path tracing
  reporting/       summary.json -> situation report
  api/             FastAPI routes
```

Must never import from `evaluation/` or read `data/eval_only/`.
