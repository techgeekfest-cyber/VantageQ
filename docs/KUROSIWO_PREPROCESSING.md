# Kuro Siwo — Actual Preprocessing and Normalisation

Source of truth: official repository `Orion-AI-Lab/KuroSiwo`, `main` at commit `4347ed173c`
(2026-07-16). Read: `configs/grd_preprocessing.xml`, `configs/train/data_config.json`,
`configs/config.json`, `configs/method/unet/unet.json`, `dataset/Dataset.py`,
`training/segmentation_trainer.py`, `utilities/utilities.py`. One labelled GRD webdataset sample was
range-read (~4 MB) to check stored values. No bulk data was downloaded.

---

## 1. Product and representation

| Stage | Representation | Evidence |
|-------|----------------|----------|
| Source | Sentinel-1 **GRD, IW, VV+VH** | `grd_preprocessing.xml`, paper |
| After SNAP calibration | **σ⁰, linear power** | `Calibration`: `outputSigmaBand=true`, `outputImageScaleInDb=false` |
| After terrain correction | **σ⁰, linear, ellipsoid-based** (no radiometric terrain flattening) | `Terrain-Correction`: `applyRadiometricNormalization=false` |
| Stored files (GeoTIFF / webdataset `.npy`) | float32 linear σ⁰, **unclamped** | Sample: VV 0.003–1.03, VH 0.002–12.2 |
| Model input | clamped, then standardised (see §3) | `Dataset.concat`, `Dataset.scale_img` |

No dB conversion anywhere in the GRD path.

## 2. SNAP GRD chain (`configs/grd_preprocessing.xml`, in order)

1. `Apply-Orbit-File`: Sentinel Precise orbits
2. `Subset`
3. `ThermalNoiseRemoval`
4. `Remove-GRD-Border-Noise` (borderLimit 500, trimThreshold 50)
5. `Land-Sea-Mask` (SRTM-based; sea set to no-data)
6. `Calibration` → σ⁰ linear
7. `Speckle-Filter` → **Lee Sigma**, window 7×7, target 3×3, sigma 0.9, ENL estimated
8. `Terrain-Correction` → Range-Doppler, **SRTM 1 Sec HGT** (EGM applied), bilinear DEM and image
   resampling, `pixelSpacingInMeter 10.0`, CRS **EPSG:3857 (Web Mercator)**, aligned to a standard
   grid, no layover/shadow mask saved, **no radiometric normalisation**
9. `Write` GeoTIFF

There is **no incidence-angle correction** and **no γ⁰ / RTC**.

**Spatial grid:** pixels are 10 **Web Mercator units**. In the inspected sample's `info.json`, a
224-px tile spans 2240 units. A Mercator unit is cos(latitude) ground metres, so ground pixel size
varies by event (≈ 9.9 m near the equator, ≈ 6.4 m at 50° N). At Trishuli (28° N) it is ≈ 8.8 m.
Tiles are 224 × 224 px.

## 3. Normalisation in code (effective official config)

`data_config.json` is parsed with `pyjson5`, which keeps the **last** value of duplicate keys.
Verified: `dem=false`, `slope=false`, `clamp_input=0.15`, `scale_input="normalize"`,
`uint8=false`, `reverse_scaling=false`, `channels=["vv","vh"]`,
`inputs=["pre_event_1","pre_event_2","post_event"]`.

Per sample (`Dataset.__getitem__` → `concat` → `scale_img`):

1. Read each date's VV and VH as float32.
2. Stack `[VV, VH]` per date.
3. **Clamp** to `[0, 0.15]` (`torch.clamp`), then **NaN → 0.15** (`torch.nan_to_num`). Same cap for
   VV and VH.
4. **Standardise** with fixed global per-channel statistics
   (`torchvision.transforms.Normalize`):
   `x' = (x − mean) / std`, with mean = `[0.0953, 0.0264]` and std = `[0.0427, 0.0215]` for `[VV, VH]`.
5. The same constants apply to all three dates. Each date is normalised **independently but
   identically**; there is no per-image or per-event statistic in this mode. (A `min-max` mode with
   per-event minima exists but is not used by the official config.)
6. Concatenate for the U-Net in this **exact channel order**:
   `[post VV, post VH, pre1 VV, pre1 VH, pre2 VV, pre2 VH]` (6 channels;
   `segmentation_trainer.py`: `cat(image, pre_event, pre_event_2)`).

Other details:

- **Labels:** 0 no water, 1 permanent water, 2 flood. Loss is `CrossEntropyLoss(ignore_index=3)`,
  but the loader never sets 3. `valid_mask` is only passed to the scaler (unused in `normalize`
  mode), so **invalid pixels are not excluded from the loss**. They keep their stored value
  (no-data = 0.0 per `info.json`), or 0.15 if NaN.
- Augmentations off by default (`data_augmentations=false`).
- Whether the published mean/std were computed before or after clamping is not documented.

## 4. The three temporal inputs

- `MS1` / `flood` = **post-event** acquisition (the "master").
- `SL1` / `pre_event_1`, `SL2` / `pre_event_2` = two **pre-event** acquisitions.
- In the inspected sample, SL1 (`crank 1`) is the more recent pre-event date (2020-05-11) and SL2
  (`crank 2`) the older one (2020-04-29), 12 days apart. Both were months before the event.
- Paper: all three acquisitions have the same orbit direction (asc or desc). Post-event lag
  averages 3.6 days (median 1).

## 5. What VantageQ can reproduce

| Kuro Siwo step | Reproducible with Sentinel Hub (CDSE)? | Notes |
|---|---|---|
| GRD IW VV/VH | Yes | Same product |
| Thermal noise removal | Yes | Applied by default (CDSE docs) |
| Border noise removal | Effectively yes | Border areas not served; irrelevant for an interior AOI |
| Precise orbit file | Approximately | SH uses orbits bundled in the product; negligible for 10 m GRD |
| Calibration σ⁰ linear | **Yes** | `backCoeff: SIGMA0_ELLIPSOID` |
| Speckle filter, before terrain correction | Approximately | SH filters on source data before orthorectification (same order), but offers **Lee**, not **Lee Sigma** |
| Range-Doppler TC, no radiometric normalisation | **Yes** | `orthorectify: true`, σ⁰ ellipsoid |
| DEM: SRTM 1″ | Approximately | SH uses COPERNICUS_30; minor geometric difference |
| Bilinear resampling | Yes | `upsampling: BILINEAR` |
| EPSG:3857 grid at 10 map units | **Yes, verified** | Smoke test 2026-10-03: EPSG:3857 output, exactly 10 × 10 units, bounds snapped to a (0, 0)-origin 10-unit grid, 0 % no-data (`scripts/test_sentinel1.py --epsg 3857`) |
| 224 × 224 tiling | Yes | Our code |
| Clamp 0.15 + mean/std standardisation + channel order | **Yes, exactly** | Pure arithmetic from the config |
| Land-sea mask | Not needed | Inland AOI |

## 6. What needs Kuro Siwo data

- Training itself (subset streaming, as planned).
- Checking how much a Kuro Siwo-trained model degrades on **steep terrain**. Kuro Siwo has
  essentially none, so even its data can only show behaviour on mild relief.
- Confirming the SL1/SL2 ordering rule beyond one sample, and whether raw label files ever contain
  a no-data code. Both are cheap to check while streaming the training subset.
- Comparing our σ⁰ histograms against per-event Kuro Siwo histograms (instead of one global
  mean/std).

---

## 7. Is the current smoke-test preprocessing aligned?

Current request: `SIGMA0_ELLIPSOID` + orthorectify + `COPERNICUS_30` + Lee 7×7 + 10 m UTM 45N.

**Radiometrically yes, closely.** Both chains give linear, ellipsoid-referenced σ⁰ with thermal
noise removed, speckle-filtered before Range-Doppler terrain correction, without radiometric
flattening. Remaining differences:

1. **Speckle filter:** Lee 7×7 vs Lee Sigma 7×7 (3×3 target). Plain Lee smooths more and blurs
   edges slightly. Minor; no Lee Sigma option in SH.
2. **Grid:** UTM, 10 m ground, vs EPSG:3857, 10 Mercator units (≈ 8.8 m ground at Trishuli). The
   UTM output is about 13 % coarser per pixel, so features look about 12 % smaller to the model.
   **Fixed with `--epsg 3857`** (verified; σ⁰ statistics unchanged vs UTM within ~1 %).
3. **DEM / orbits:** COPERNICUS_30 vs SRTM 1″; bundled vs precise orbits. Minor.

**The brightness mismatch seen in the smoke test is not a processing difference.** Our VV median of
0.20 vs Kuro Siwo's mean of 0.095 comes from the scene: steep Himalayan slopes facing the
ascending radar are bright in un-flattened σ⁰, exactly as Kuro Siwo's own chain would produce.
Kuro Siwo's training events are mostly lowland, so this is a **domain gap, not a units or
processing bug**. Flood/water pixels are the dark end (σ⁰ ≲ 0.03), which the 0.15 clamp does not
affect. Bright slopes will saturate at 0.15. A matching histogram would not prove compatibility, and
a mismatched one doesn't by itself rule it out.

---

## VantageQ implementation recommendation

Simplest pipeline with a reasonable chance of transfer. Do **not** add RTC/γ⁰: it would move our
inputs away from what the model was trained on.

1. **Scenes:** same relative orbit and direction; post = first acquisition after the event; pre1 =
   latest pre-event, pre2 = the one before (Kuro Siwo SL1/SL2 order). Trishuli: Track 85,
   post 2026-08-28, pre1 2026-08-16, pre2 2026-08-04.
2. **Retrieval (Sentinel Hub on CDSE), per date:** `SIGMA0_ELLIPSOID`, `orthorectify: true`,
   `COPERNICUS_30`, `speckleFilter LEE 7×7`, `upsampling BILINEAR`, bands `VV, VH, dataMask`,
   FLOAT32, **output in EPSG:3857 at 10 × 10 map units** with identical bounds for all three dates.
   This keeps the current smoke-test settings and changes only the CRS.
3. **Model input:** clamp each band to [0, 0.15], NaN → 0.15, then
   `(x − [0.0953, 0.0264]) / [0.0427, 0.0215]`, stacked as
   `[post VV, post VH, pre1 VV, pre1 VH, pre2 VV, pre2 VH]`. Tile at 224 px with overlap and stitch.
   Use exactly the code path we train with, so training and inference share one function.
4. **Training:** Kuro Siwo official U-Net config (ResNet-18, 3 classes, `dem=false`) on a streamed
   subset, unchanged normalisation.
5. **Terrain safeguard (post-processing only, not model input):** mark pixels as "not assessable"
   where the Copernicus DEM slope is steep (threshold to be chosen on Kuro Siwo validation data, not
   EMSR927), and where `dataMask = 0`. A layover/shadow mask is optional later: Sentinel Hub only
   provides `shadowMask` with `GAMMA0_TERRAIN`, so it would need a separate auxiliary request.
6. **Sanity check before trusting outputs:** compare the dark-end statistics (p1–p25) of our
   pre-event scenes with Kuro Siwo samples, and visually inspect predicted water against the
   river course in pre-event imagery (permanent water should be found).
