# Training Pipeline Smoke Test

Run date: **2026-10-03**. Purpose: prove the end-to-end training path works (data → preprocessing
→ U-Net → loss → backward → optimizer step).

> **This is not an accuracy experiment.** 57 samples, 16 optimizer steps, no validation, no
> checkpoint saved. Nothing here shows that the model detects floods. The model is **not trained**.

---

## 1. Subset selection

Script: `scripts/build_kurosiwo_subset.py` (logic in `ml/kurosiwo.py`).

1. **Find events without downloading shards.** Each of the 5 labelled `train_GRD` shards
   (3.1–10.9 GB) was probed every 1 GB. A ~200 KB HTTP range read finds a tar header, then
   512-byte header hops reach the nearest sample's `info.json` and its `actid` (event ID).
   48 probes fetched **10.2 MB** in total.
2. **Filter to the official train split.** The Hugging Face `train_GRD` shards also contain
   official **val** event 1111003 and **test** events 1111007 and 1111013. The HF split does not
   follow `configs/train/data_config.json`, so selection always filters by event ID.
3. **Stream a few samples per event.** For the first 8 official-train events found, stream from
   that event's probed offset and keep up to 8 complete samples of that event, then stop reading
   (per-event byte cap).

Events found by probing (1 GB spacing; smaller events can be missed):

| Shard | Events (in order) |
|---|---|
| 00000 | 470, 497, 502, 518, 555, *1111003 (val)*, 1111004 |
| 00001 | 1111004, 1111005, 1111006, *1111007 (test)*, 1111009 |
| 00002 | 1111009 |
| 00003 | 1111010, 1111011, *1111013 (test)* |
| 00004 | *1111013 (test)* |

## 2. Subset obtained

| Event | Region | Samples |
|---|---|---:|
| 470 | Togo | 8 |
| 497 | Germany | 8 |
| 502 | Ireland | 8 |
| 518 | Belgium | 1 (probe landed near the event's end) |
| 555 | Spain | 8 |
| 1111004 | USA (Houston) | 8 |
| 1111005 | Madagascar | 8 |
| 1111006 | Madagascar | 8 |
| **Total** | **8 events** | **57** |

- Downloaded: **123.0 MB streamed** + 10.2 MB probing. On disk: **100 MB** in
  `data/external/kurosiwo_subset/` (git-ignored).
- Label pixels: no water 2,264,038; permanent water 129,644; flood 379,177; **ignore (3) 87,173**.

**Label 3 occurs in the data** (not on the dataset card). Verified in this subset: all 35,126
invalid pixels (`valid_mask = 0`) had label 3, and another 52,047 valid pixels also had label 3.
The reason for label 3 on valid pixels is not established. The official
`CrossEntropyLoss(ignore_index=3)` excludes all label-3 pixels. The model predicts only classes
0–2 (`NUM_CLASSES = 3`). This corrects an earlier statement in `docs/KUROSIWO_PREPROCESSING.md` /
`docs/KUROSIWO_DATASET.md`, now updated.

## 3. Preprocessing and model

- **Single shared preprocessing:** `ml/preprocessing.py::build_model_input`. Clamp linear σ⁰ to
  [0, 0.15], NaN → 0.15, VV `(x − 0.0953)/0.0427`, VH `(x − 0.0264)/0.0215`, order
  `[post VV, post VH, pre1 VV, pre1 VH, pre2 VV, pre2 VH]`. Used by the dataset and the
  sample-inspection script, and intended for Trishuli inference.
- **Dataset:** `ml/dataset.py::KuroSiwoDataset` → `X` float32 `[6, 224, 224]`, `y` int64
  `[224, 224]`, labels as stored.
- **Model:** `ml/model.py::build_unet`, i.e. `smp.Unet("resnet18", encoder_weights="imagenet",
  in_channels=6, classes=3)`, the official Kuro Siwo U-Net config. No architecture change. smp
  adapts the ImageNet first convolution to 6 channels; the weights (~45 MB) are downloaded once
  to the Hugging Face cache.
- **Training settings (official where defined):** Adam, lr 1e-3, `CrossEntropyLoss(ignore_index=3)`,
  batch 8, shuffle, seed 0, 2 epochs. No augmentation, no LR schedule (smoke test only).
  `train_smoke.py` refuses to run if any sample's event is outside the official train split.

## 4. Result

| Item | Value |
|---|---|
| Device | Apple M4, PyTorch 2.14.1, **MPS** |
| X batch | `(8, 6, 224, 224)` float32 |
| y batch | `(8, 224, 224)` int64 |
| Model output | `(8, 3, 224, 224)` |
| Forward pass | OK |
| Backward pass | OK (finite, non-zero gradients on the first conv) |
| Optimizer steps | 16 (2 epochs × 8 batches) |
| Initial loss (first batch, before any step) | **1.6469** |
| Epoch mean loss | 1.0975 → 0.6306 |
| Final batch loss | **0.7068** |
| Runtime | 3.3 s training, 3.8 s total (first run 12.5 s incl. MPS warm-up + weight download) |

The falling loss only shows that gradients flow and parameters update. On 57 samples it says
nothing about generalisation. Results are stored in `ml/runs/train_smoke/result.json`
(git-ignored).

## 5. Limitations

- Tiny, unbalanced subset: one event has a single sample; samples within an event are
  neighbouring tiles; events are mostly lowland; no Himalayan terrain.
- No validation or test evaluation; no metrics computed.
- Probing at 1 GB spacing finds 11 of the 27 official train events; smaller events are skipped.
- No augmentation, LR schedule or class weighting; settings are not tuned.

## 6. Commands

```bash
.venv/bin/pip install -r requirements.txt          # adds torch, segmentation-models-pytorch
.venv/bin/python scripts/build_kurosiwo_subset.py --dry-run   # probe + estimate (10 MB)
.venv/bin/python scripts/build_kurosiwo_subset.py             # stream 8 events x 8 samples
.venv/bin/python scripts/train_smoke.py                       # 2 epochs, auto device
.venv/bin/python -m pytest -q
```
