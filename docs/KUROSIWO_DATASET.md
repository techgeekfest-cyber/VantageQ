# Kuro Siwo — Dataset Structure and Minimal Subset

Inspected 2026-10-03 against `Orion-AI-Lab/KuroSiwo` `main` (`4347ed173c`) and the Hugging Face
datasets `orion-ai-lab/Kuro-Siwo-Webdataset` / `Kuro-Siwo-GeoTIFFs`. Preprocessing and
normalisation details: `docs/KUROSIWO_PREPROCESSING.md`. No bulk download; 3 samples fetched.

---

## 1. Three distributions of the same data

| Distribution | Container | Labelled only? | Used by official code? |
|---|---|---|---|
| Dropbox GRD release (`download_kuro_siwo.sh`, 11 × `.tar.gz` + `catalogue.gpkg`) | GeoTIFF directories | mixed | **Yes** (`dataset/Dataset.py`) |
| HF `Kuro-Siwo-GeoTIFFs/GRD/*.tar` (35 tars, 12–20 GB each, 699 GB) | GeoTIFF directories | **mixed** (5 of the first 27 folders had no label/DEM) | Layout differs slightly from the Dropbox one (see §2) |
| HF `Kuro-Siwo-Webdataset/train_GRD`, `test_GRD` (5 + 12 shards, 3–11 GB each) | WebDataset `.tar` of `.npy` + `info.json` | **Yes** | No (the official repo has no webdataset loader) |

**Chosen source for VantageQ: the HF labelled webdataset.** Every sample is labelled. Samples are
stored sequentially, so the first N can be streamed without downloading the shard. Its
`train_GRD` / `test_GRD` folders do **not** follow the official event split (e.g. `train_GRD`
contains val event 1111003 and test events 1111007, 1111013), so samples must be filtered by
`info.json` `actid` (see `docs/TRAINING_SMOKE_TEST.md`).

## 2. Official loader format (GeoTIFF directories)

- Root `<root>/data/<actid>/<aoiid:02>/<grid_id>/` (Dropbox layout, from the index `path` field,
  e.g. `118/01/0600fb56…`). The HF GeoTIFF tars add a 2-character prefix level:
  `data/<actid>/<aoi>/<grid_id[:2]>/<grid_id>/`.
- Files per grid (224 × 224, EPSG:3857, 10 units, no-data = 0):

| File prefix | Content | dtype |
|---|---|---|
| `MS1_IVV_<act>_<aoi>_<date>.tif`, `MS1_IVH_…` | **post-event** VV / VH σ⁰ linear | float32 |
| `SL1_IVV_…`, `SL1_IVH_…` | **pre-event 1** (more recent) | float32 |
| `SL2_IVV_…`, `SL2_IVH_…` | **pre-event 2** (older) | float32 |
| `MK0_MLU_…` | label | — |
| `MK0_MNA_…` | valid-pixel mask (1 valid) | uint8 |
| `MK0_DEM_…` | DEM | float32 |
| `info.json` | grid metadata | JSON |

- **Index:** `pickle/KuroV2_grid_dict.gz` (train/val) and `pickle/KuroV2_grid_dict_test_0_100.gz`
  (val/test), loaded with `compress_pickle`. A dict `grid_id → {path, info, clz, clz_name}`;
  31,707 entries in the train pickle. **`clz` is the climate zone (1–3), not a label class.** The
  pickle contains several events; `Dataset.__init__` keeps only records whose `info.actid` is in
  the split's activation list (`train_acts` / `val_acts` / `test_acts`).
  Inspected safely with a restricted unpickler (builtins only).
- **Reading** (`Dataset.__getitem__`): `os.listdir(<root>/data/<path>)`, match files by prefix
  (`MS1_IVV`, `SL1_IVH`, `MK0_MLU`, …), read with `cv2.imread(..., IMREAD_ANYDEPTH)`.

## 3. Webdataset sample format (verified locally)

Tar members `<key>.<field>`, e.g. `000000.flood_vv.npy`:

| Field | Official equivalent | Verified |
|---|---|---|
| `flood_vv`, `flood_vh` | MS1 post-event | (1, 224, 224) float32, linear σ⁰ |
| `sec1_vv`, `sec1_vh` | SL1 pre-event 1 | (1, 224, 224) float32 |
| `sec2_vv`, `sec2_vh` | SL2 pre-event 2 | (1, 224, 224) float32 |
| `dem` | MK0_DEM | (1, 224, 224) float32, metres |
| `mask` | MK0_MLU | (1, 224, 224) float32, values 0–2 per the dataset card; 3 also observed (see §4) |
| `valid_mask` | MK0_MNA | (1, 224, 224) float32, {0, 1} |
| `info.json` | info.json | event `actid`, `aoiid`, acquisition dates/IDs per MS1/SL1/SL2, `pflood`, `pwater`, `geom` |

About 1.8 MB per sample (9 arrays × 200 KB + JSON).

## 4. Label encoding

| Value | Class |
|---|---|
| 0 | no water |
| 1 | permanent water |
| 2 | flood |
| 3 | not a class; official loss/metrics use `ignore_index=3` |

Verified in the 57-sample training subset (`docs/TRAINING_SMOKE_TEST.md`):

- Value 3 **occurs in the data**, although the dataset card lists only 0–2 (87,173 pixels).
- All invalid pixels observed (`valid_mask = 0`, σ⁰ = 0) had label 3 (35,126 of 35,126).
- A further 52,047 **valid** pixels also had label 3. **Why is not established**; neither the
  official code nor the dataset card explains it.
- The official loader does not remap labels. Its loss uses `ignore_index=3`, so all label-3 pixels
  are excluded from the loss. VantageQ uses labels as stored, matching the official code.

## 5. Loader flow → model input

1. Stack each date as `[VV, VH]`.
2. Clamp to `[0, 0.15]`, then NaN → 0.15.
3. Normalise with `mean [0.0953, 0.0264]`, `std [0.0427, 0.0215]`.
4. Concatenate **`[post VV, post VH, pre1 VV, pre1 VH, pre2 VV, pre2 VH]`** → (6, 224, 224)
   (`segmentation_trainer.py`: `cat(image, pre_event)`, then `cat(…, pre_event_2)`).

`ml/preprocessing.py::build_model_input` is the single numpy port of this (used by the dataset,
the inspection script and future Trishuli inference), checked by `tests/test_kurosiwo_sample.py`. Clamped pixels map to 1.281 (VV) and 5.749 (VH), as observed.

## 6. Minimal subset and results

- **Files needed:** the first N samples of `train_GRD/shard-00000.tar` (10.91 GB shard, **not**
  downloaded in full). With N = 3: **5.5 MB streamed, 5.3 MB on disk** in
  `data/external/kurosiwo_smoke/` (git-ignored).
- Samples obtained: `000000`, `000001`, `000002`, all event **470** (Togo, Oti River, official
  **train** split). Post 2020-10-14, pre1 2020-05-11, pre2 2020-04-29. 100 % valid pixels, no NaN.

| Sample | no water (0) | permanent water (1) | flood (2) |
|---|---:|---:|---:|
| 000000 | 49,297 (98.25 %) | 0 | 879 (1.75 %) |
| 000001 | 28,306 (56.41 %) | 0 | 21,870 (43.59 %) |
| 000002 | 27,982 (55.77 %) | 0 | 22,194 (44.23 %) |

No permanent-water pixels in these three. Class 1 was later observed in the multi-event training
subset (129,644 pixels; `docs/TRAINING_SMOKE_TEST.md`).

**For real training**, stream a few hundred to ~2,000 samples (≈ 0.5–4 GB) across events.
Shards appear ordered by event (first GeoTIFF tar was all event 1111009; first webdataset samples
all event 470), so the first samples of one shard cover only one or two events. Sampling across
the 5 train shards and the val events is needed before any meaningful metric.

## 7. Commands used

```bash
# Index pickle (9 MB, from the repo), inspected with a restricted unpickler
curl -sL https://raw.githubusercontent.com/Orion-AI-Lab/KuroSiwo/main/pickle/KuroV2_grid_dict.gz -o KuroV2_grid_dict.gz
# Shard sizes (metadata only)
curl -s https://huggingface.co/api/datasets/orion-ai-lab/Kuro-Siwo-Webdataset/tree/main/train_GRD
# Fetch + inspect 3 labelled samples (stream stops after 3 samples; hard cap 30 MB)
.venv/bin/python scripts/kurosiwo_sample_smoke.py --n 3
.venv/bin/python scripts/kurosiwo_sample_smoke.py --inspect-only
```

Licence: CC BY 4.0. Cite Bountos et al., NeurIPS 2024 (see `docs/DATA_SOURCES.md` §1.10).
