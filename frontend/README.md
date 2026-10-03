# frontend/

VantageQ MVP dashboard: Next.js (static export) + TypeScript + MapLibre GL JS. No backend API, no
database, no authentication. It reads only the committed demo snapshot in
`public/demo/trishuli/` and computes no numbers of its own: every figure comes from
`infrastructure_summary.json` / `connectivity_summary.json`.

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
npm test             # vitest (jsdom)
npm run typecheck
npm run build        # static site in out/ (serve with any static file server)
```

`predev`/`prebuild` copy MapLibre's web worker into `public/maplibre/` (git-ignored); without it
no GeoJSON layer renders.

Regenerate the demo snapshot from the (git-ignored) pipeline outputs:

```bash
.venv/bin/python scripts/build_demo_snapshot.py
```

Code layout: `src/lib/data.ts` (loading, validation, derived values), `src/lib/layers.ts` (layer
definitions for map + legend), `src/lib/format.ts` (wording, popups), `src/components/`
(Dashboard, FloodMap, Legend, KpiCards, SettlementPanel, MethodologyPanel).

The map opens on the whole analysis AOI; **Zoom to impacts** fits it to the extent of the impact
layers (predicted flood, flooded road sections, bridges intersecting predicted flood, potentially
affected buildings, potentially cut-off settlements), derived from the loaded GeoJSON.

Basemap: live OpenStreetMap standard raster tiles (current, for orientation only; internet
required). Analytical layers come only from the committed snapshot, built from the 2026-08-25
historical OSM snapshot.
