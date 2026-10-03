# Trishuli Inference Experiment 1

Run date: **2026-10-04**. The frozen Kuro Siwo U-Net applied to the Trishuli August 2026 event on
one small sub-AOI.

> **This is an inference experiment, not a validated assessment.** No labels exist for Trishuli
> in this project's inputs; no accuracy is computed. EMSR927 and other published maps were not
> used, not compared and not used to modify anything. The model was not retrained or tuned.

---

## 1. Scenes (Sentinel-1D, IW, VV+VH, relative orbit 85, ascending)

Found through the CDSE STAC catalogue (`backend/vantageq/satellite/sentinel1.py`) and checked
against the expected dates.

| Role | Acquisition (UTC) | STAC ID |
|---|---|---|
| post | 2026-08-28 12:21:41 | `S1D_IW_GRDH_1SDV_20260828T122141_20260828T122206_004326_007FA4_C73B_COG` |
| pre1 (most recent) | 2026-08-16 12:21:41 | `S1D_IW_GRDH_1SDV_20260816T122141_20260816T122206_004151_007980_B091_COG` |
| pre2 | 2026-08-04 12:21:40 | `S1D_IW_GRDH_1SDV_20260804T122140_20260804T122205_003976_00737A_38EB_COG` |

## 2. Retrieval and grid

- Sentinel Hub Process API on CDSE, per date: `SIGMA0_ELLIPSOID` (linear), `orthorectify: true`,
  `COPERNICUS_30` DEM, Lee 7×7, bilinear, bands VV, VH, dataMask.
- **One shared grid** for all dates (`make_grid`): sub-AOI 85.17–85.22 E, 27.95–27.99 N → EPSG:3857,
  bounds snapped to a (0, 0)-origin 10-unit grid. The request sets explicit `width`/`height`.

| Item | Value |
|---|---|
| CRS | EPSG:3857 |
| Bounds | 9,481,080 / 3,242,670 / 9,486,650 / 3,247,720 |
| Size | **557 × 505** px (w × h) |
| Transform | `(10, 0, 9481080, 0, −10, 3247720)` |
| Pixel size | exactly 10 × 10 map units (≈ 8.83 m on the ground at 27.97° N) |

`check_same_grid` confirmed all three inputs and all four outputs share CRS, transform, size and
bounds. Inputs are cached at `data/interim/trishuli/s1_{post,pre1,pre2}_<date>_relorb85_epsg3857.tif`
(6.6 MB, git-ignored).

## 3. Input quality (linear σ⁰)

| Date | No-data | VV mean / median / p99 | VV > 0.15 | VH mean / median | VH > 0.15 |
|---|---:|---|---:|---|---:|
| post 08-28 | 0 % | 0.323 / 0.204 / 1.57 | 63.1 % | 0.066 / 0.046 | 9.1 % |
| pre1 08-16 | 0 % | 0.327 / 0.209 / 1.58 | 64.5 % | 0.067 / 0.046 | 9.1 % |
| pre2 08-04 | 0 % | 0.343 / 0.209 / 1.73 | 65.2 % | 0.070 / 0.047 | 10.8 % |

As in the smoke test, ~2/3 of VV pixels exceed Kuro Siwo's 0.15 clamp, caused by steep terrain in
un-flattened σ⁰ (docs/KUROSIWO_PREPROCESSING.md §7). All three dates are statistically similar.

## 4. Preprocessing and inference

- `ml/preprocessing.py` only: clamp [0, 0.15], NaN → 0.15, VV `(x − 0.0953)/0.0427`,
  VH `(x − 0.0264)/0.0215`, **global fixed statistics (no per-image normalisation)**, channels
  `[post VV, post VH, pre1 VV, pre1 VH, pre2 VV, pre2 VH]` via `ml/inference.py::sar_inputs_from_dates`.
  Model input: **(6, 505, 557) float32**.
- Checkpoint: `ml/checkpoints/kurosiwo_unet_r18_best.pt`, SHA-256 prefix `0e28f6a44b774c5b`,
  epoch 17, loaded with `load_frozen_model` (no ImageNet download, eval mode). Device: MPS.
- Tiling (`ml/inference.py::predict_tiled`): native resolution, **224 × 224** tiles, 64-px overlap
  (stride 160), 32-px reflect padding on every side (no resizing), softmax probabilities averaged
  over overlapping tiles; last tile aligned to the edge.
- Tiling check: a second pass with a different tiling (overlap 112, margin 80) differs by a mean
  |Δ P(flood)| of **0.0029**. A unit test confirms tiling exactly reproduces untiled output for a
  pixel-wise model.

## 5. Results

| Predicted class (of 281,285 valid pixels) | % | Pixels |
|---|---:|---:|
| no water | 99.063 | 278,648 |
| permanent water | 0.388 | 1,090 |
| **flood** | **0.550** | **1,547** (≈ 0.12 km², cos-latitude corrected) |

P(flood): mean 0.013, median 0.0015, p90 0.018, p99 0.201, max 0.929; 0.489 % of pixels > 0.5,
0.007 % > 0.9.

### Outputs (git-ignored, `data/processed/trishuli/`)

| File | Content |
|---|---|
| `trishuli_20260828_class.tif` | uint8: 0 no water, 1 permanent water, 2 flood, 255 no data |
| `trishuli_20260828_prob_flood.tif` | float32 P(flood) |
| `trishuli_20260828_prob_permanent_water.tif` | float32 P(permanent water) |
| `trishuli_20260828_prob_water.tif` | float32 P(permanent water) + P(flood) |
| `trishuli_20260828_summary.json` | scenes, grid, input stats, class %, probability summary, hashes |
| `quicklook_prediction.png` | pre1 VV, post VV, predicted classes overlay, P(flood), P(water) |
| `quicklook_change.png` | post − mean(pre) VV dB, pre1 − pre2 VV dB (no-event reference), class map |

## 6. Observations (visual + descriptive, not accuracy)

- **No border artefacts or tile seams.** Predicted water is lower in the 10-px border (0.25 %)
  than in the interior (0.99 %); no tile-grid pattern is visible; tiling diagnostic 0.003.
- **No image-wide flood hallucination.** 99.1 % no water.
- **Flood predictions coincide with event-specific darkening.** In the post − pre change map there
  is a dark (−6 dB) blob at the centre and a dark linear trail running SW along the valley floor,
  plus a bright linear feature running NE. None of these appear in the pre1 − pre2 reference.
  Pixels > 6 dB darker than the pre-event mean: 0.55 % of the AOI (vs 0.04 % between the two
  pre-event dates). 64.5 % of them are predicted flood. Predicted-flood pixels darkened on average
  by 6.7 dB (median VV −9.3 → −15.3 dB), which is consistent with new open water. It is not proof.
- **Likely missed changes.** Most of the SW dark linear trail is not predicted as flood. Of pixels
  > 3 dB darker, only 16.4 % are predicted flood. The bright NE feature (brighter after) is not
  water-like and is not predicted; it may relate to debris or surface change, but nothing here
  establishes that.
- **"Permanent water" predictions look terrain-following.** They form thin streaks on dark slopes,
  dark on both dates (median −16.7 dB pre, −17.4 dB post, mean change −0.9 dB), not along the
  river. This is consistent with radar-shadow / back-slope darkness rather than water. It is
  unverified without a DEM or shadow mask.
- Consistent with the test evaluation (high flood precision, low recall, weak flood vs
  permanent-water separation), but those Kuro Siwo numbers do not transfer to this terrain.

## 7. Limitations

- One small sub-AOI (~4.9 × 4.5 km), one track, one post date (2 days after the event).
- No ground truth used; nothing here is an accuracy statement.
- Strong domain gap: steep Himalayan terrain is absent from Kuro Siwo; ~64 % of VV pixels saturate
  at the clamp.
- No layover/shadow or slope mask yet, so terrain false positives are not suppressed.
- 10 m SAR on a narrow valley: small or short-lived water may be missed; the flood may have receded
  before 2026-08-28.
- Flood area uses a cos(latitude) correction for Web Mercator pixels; approximate.

## 8. Commands

```bash
.venv/bin/python scripts/infer_trishuli.py   # fetches the 3 scenes if not cached (needs .env), then infers
.venv/bin/python -m pytest -q
```

Code added for this run: `backend/vantageq/satellite/sentinel1.py` (search, selection, shared
grid, identical-grid check, Process API retrieval; moved from `scripts/test_sentinel1.py`, which
now uses it), `ml/inference.py` (date → channel mapping, frozen-checkpoint loading, tiled
inference), `scripts/infer_trishuli.py`, `tests/test_trishuli_inference.py`.
