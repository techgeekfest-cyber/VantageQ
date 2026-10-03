# data/

Nothing in this directory is committed except this README and `.gitkeep` files.

| Directory     | Contents                                                                 | Production pipeline may read? |
|---------------|--------------------------------------------------------------------------|-------------------------------|
| `raw/`        | Sentinel-1/2 scenes, Copernicus DEM tiles, pre-event OSM extracts        | Yes                           |
| `interim/`    | Preprocessed/co-registered rasters, cached intermediate products         | Yes                           |
| `processed/`  | System outputs (flood masks, impact GeoJSON, connectivity results)       | Yes (writes)                  |
| `external/`   | Training datasets (Kuro Siwo, optionally Sen1Floods11)                   | Training only (`ml/`)         |
| `eval_only/`  | EMSR927 and any other published damage/flood maps                        | **No — evaluation only**      |

`eval_only/` is the quarantine for reference products. Only code under `evaluation/` may read
from it. See `docs/ARCHITECTURE.md` §4.
