# Sentinel-1 Smoke Test (Milestone 1)

Run date: **2026-10-03**. Script: `scripts/test_sentinel1.py`. Tests:
`tests/test_sentinel1_selection.py`.

**Result: catalogue stage succeeded. Raster retrieval not yet run** because no Sentinel Hub
credentials are configured. The process endpoint returns HTTP 401 without a token, so this is a
hard requirement, not a script issue. No raster statistics exist yet.

No EMSR927 or other published damage map was accessed. No full scene was downloaded.

---

## 1. How to run

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/test_sentinel1.py --metadata-only   # catalogue only (no credentials)
.venv/bin/python scripts/test_sentinel1.py                   # + retrieval + stats (needs .env)
.venv/bin/python scripts/test_sentinel1.py --stats FILE.tif  # stats for an existing GeoTIFF
.venv/bin/python -m pytest
```

Outputs go to `data/interim/smoke_s1/` (git-ignored).

---

## 2. Catalogue search (succeeded)

| Item | Value |
|------|-------|
| API | CDSE STAC API, `https://stac.dataspace.copernicus.eu/v1/search`, collection `sentinel-1-grd` |
| Authentication | None needed for search |
| AOI (search) | 85.10–85.42 E, 27.85–28.30 N (approximate Trishuli valley, our reconnaissance box) |
| Time window | 2026-07-21 to 2026-09-09 (event −36 d / +14 d) |
| Filters (client-side) | `sar:instrument_mode = IW`, `sar:polarizations ⊇ {VV, VH}`, `sat:orbit_state = ascending`, `sat:relative_orbit = 85`, footprint covers ≥ 99 % of the AOI |
| Selection rule | First scene strictly after the event day; two latest strictly before it; scenes on the event day itself are skipped as ambiguous (event time unknown) |

12 items were returned (all Sentinel-1D, IW, VV+VH): tracks 85 ascending, 19 descending (both
100 % AOI coverage) and 121 descending (~32 %). Four matched the filter.

### Selected scenes (identifiers as returned by the STAC catalogue)

| Role | Acquisition (UTC) | Platform | Direction | Rel. orbit | Abs. orbit | STAC ID |
|------|-------------------|----------|-----------|-----------:|-----------:|---------|
| Pre 1 | 2026-08-04 12:21:40 | S1D | Ascending | 85 | 3976 | `S1D_IW_GRDH_1SDV_20260804T122140_20260804T122205_003976_00737A_38EB_COG` |
| Pre 2 | 2026-08-16 12:21:41 | S1D | Ascending | 85 | 4151 | `S1D_IW_GRDH_1SDV_20260816T122141_20260816T122206_004151_007980_B091_COG` |
| **Post** | **2026-08-28 12:21:41** | S1D | Ascending | 85 | 4326 | `S1D_IW_GRDH_1SDV_20260828T122141_20260828T122206_004326_007FA4_C73B_COG` |

Also matched but not selected: 2026-07-23 12:21:40 (`…_003801_006D5D_2C7A_COG`).

The STAC IDs are for the COG product variant. Their last 4-character suffix differs from the
non-COG SAFE names in the OData catalogue (e.g. `…_007FA4_01B4`), but datatake ID, absolute orbit
and timing match. These are the same acquisitions recorded in `docs/DATA_SOURCES.md` §3.

---

## 3. Raster retrieval (not run, credentials missing)

### Planned request (implemented, untested against the live API)

| Item | Value |
|------|-------|
| API | Sentinel Hub Process API on CDSE, `POST https://sh.dataspace.copernicus.eu/api/v1/process` |
| Scene | Post-event, 2026-08-28 (time filter = scene start − 1 min … end + 1 min) |
| Sub-AOI | 85.17–85.22 E, 27.95–27.99 N (~5 × 4.4 km, Trishuli river near Betrawati), requested in EPSG:32645 (UTM 45N) |
| Data filter | `acquisitionMode IW`, `polarization DV`, `orbitDirection ASCENDING`, `resolution HIGH` |
| Processing | `backCoeff SIGMA0_ELLIPSOID`, `orthorectify true`, `demInstance COPERNICUS_30`, `speckleFilter LEE 7×7`, `upsampling BILINEAR` |
| Bands | `VV`, `VH` (linear σ⁰, FLOAT32), `shadowMask`, `dataMask` |
| Output | 10 m GeoTIFF, about 490 × 445 px, 4 bands: `data/interim/smoke_s1/s1_sigma0_20260828_relorb85_subaoi.tif` |

Kuro Siwo used Lee Sigma (7×7 window) and SRTM terrain correction in SNAP. Our request uses plain
Lee (7×7) and Copernicus DEM, so the two chains are close but not identical.

### Authentication needed

1. Log in to the CDSE Sentinel Hub dashboard: <https://shapps.dataspace.copernicus.eu/dashboard>
   (free CDSE account).
2. User settings → **OAuth clients** → create a client. Copy the client ID and secret (the secret
   is shown once).
3. `cp .env.example .env` and set `SH_CLIENT_ID` and `SH_CLIENT_SECRET`. `.env` is git-ignored;
   never commit it.
4. The script exchanges these for a token at
   `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token`
   (`grant_type=client_credentials`).

Checked on 2026-10-03: the Process API returns **401** without a token, and the token endpoint
returns **401 `invalid_client`** for unknown clients.

---

## 4. Raster statistics

**Not yet available.** Once retrieval works, the script prints and saves
(`*.stats.json`, `*.hist.png`):

- shape, CRS, resolution, no-data %, shadow %
- VV and VH min, max, mean, median, std, p1/p5/p25/p50/p75/p95/p99, median in dB
- % of pixels above Kuro Siwo's clamp (0.15) and the mean after clamping
- Kuro Siwo reference values: VV mean 0.0953 / std 0.0427, VH mean 0.0264 / std 0.0215
  (`configs/train/data_config.json`)

The statistics code is exercised on a synthetic raster in the tests.

What to expect: linear values mostly in ~0.001–0.5, VV median roughly 0.03–0.1 (≈ −15 to −10 dB)
and VH several times lower. Steep slopes facing the radar will be much brighter; shadowed slopes
near the noise floor. A rough match with Kuro Siwo's statistics only shows the units and scale are
consistent. **It does not prove model compatibility.** Our mountain terrain differs strongly from
Kuro Siwo's mostly lowland events.

---

## 5. Next step

1. Add `SH_CLIENT_ID` / `SH_CLIENT_SECRET` to `.env` and rerun
   `.venv/bin/python scripts/test_sentinel1.py`.
2. Record the raster statistics and histogram in §4 of this file.
3. If the values look reasonable, fetch the two pre-event scenes for the same sub-AOI and compare
   (same track, so pixels should align).
