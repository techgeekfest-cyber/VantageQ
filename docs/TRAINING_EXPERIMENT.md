# Training Experiment 1: Kuro Siwo U-Net, Event-Separated

Run date: **2026-10-04**. First real learning experiment: official Kuro Siwo U-Net trained on a
small event-separated subset, evaluated on held-out official **validation** events.
No Trishuli data, EMSR927 or other published map was used.

> Read §6 before quoting any number. The validation set is small and dominated by one event, and
> the checkpoint was selected on the same validation set. These are first-signal numbers, not a
> benchmark.

---

## 1. Data

### Selection

1. **Probe** all 17 labelled HF shards (5 `train_GRD`, 12 `test_GRD`) every 0.5 GB: 343 probes,
   **72.7 MB** of range reads, 6 min (8 threads). 24 of 27 official train events and 4 of 7 val
   events were located; val events 514, 520, 559 (small) were not hit at this spacing.
2. **Choose events.** Train events picked for spread over continents and climate zones; all four
   located val events used.
3. **Stream** up to 18 (train) / 20 (val) samples per event, from up to 2 separate probe
   positions inside each event so samples are not all adjacent tiles. Split membership is
   enforced against the official lists; samples are de-duplicated by `grid_id` (HF `train_GRD`
   and `test_GRD` overlap in events).

Script: `scripts/build_kurosiwo_experiment_data.py`. Output: `data/external/kurosiwo_experiment/`
(git-ignored), with `manifest.json` (split, event, grid, shard, offset per sample) and `probes.json`.

### Train: 252 samples, 14 official TRAIN events

| Event | Region | Samples |
|---|---|---:|
| 118 | Spain | 18 |
| 130 | Myanmar | 18 |
| 147 | UK (Cumbria) | 18 |
| 273 | Albania | 18 |
| 275 | Croatia | 18 |
| 324 | France | 18 |
| 427 | Sweden | 18 |
| 470 | Togo | 18 |
| 502 | Ireland | 18 |
| 555 | Spain (Ebro) | 18 |
| 1111004 | USA (Houston) | 18 |
| 1111005 | Madagascar | 18 |
| 1111009 | Pakistan | 18 |
| 1111011 | Philippines | 18 |

Label pixels: no water 10,328,119 · permanent water 307,131 · flood 1,540,465 · label 3 468,637.

### Validation: 65 samples, 4 official VAL events

| Event | Region | Samples | Flood GT px | Perm. water GT px |
|---|---|---:|---:|---:|
| 279 | Spain | 20 | **0** | 25,987 |
| 437 | France | 20 | 4,614 | 1,735 |
| 1111003 | Djibouti | 5 (small event; byte cap reached) | 560 | 3,937 |
| 1111008 | Nicaragua | 20 | **88,771** | 17,278 |

Label pixels: no water 2,849,217 · permanent water 48,937 · flood 93,945 · label 3 269,341.
No event or grid appears in both splits (checked by the training script, which refuses otherwise).

**Size:** 622.3 MB streamed + 72.7 MB probing; **559 MB on disk** (train 444 MB, val 115 MB).

## 2. Preprocessing and model (unchanged, shared code)

- `ml/preprocessing.py`: clamp [0, 0.15], NaN → 0.15, VV `(x − 0.0953)/0.0427`,
  VH `(x − 0.0264)/0.0215`, channels `[post VV, post VH, pre1 VV, pre1 VH, pre2 VV, pre2 VH]`.
- `ml/dataset.py`: X float32 `[6, 224, 224]`, y int64 `[224, 224]`, labels as stored.
- `ml/model.py`: `smp.Unet("resnet18", encoder_weights="imagenet", in_channels=6, classes=3)`.
- Classes 0 no water, 1 permanent water, 2 flood; label 3 ignored (`ignore_index=3`).

## 3. Training configuration

| Item | Value |
|---|---|
| Device | Apple M4, PyTorch 2.14.1, MPS |
| Optimizer | Adam, lr 1e-3 (official) |
| Schedule | Cosine annealing over 20 epochs (official uses cosine) |
| Loss | `CrossEntropyLoss(ignore_index=3)` (official) |
| Batch size | 16 |
| Epochs | 20 |
| Augmentation | Random horizontal/vertical flip per batch (not in the official default; added because the train set is small) |
| Seed | 0 (Python, NumPy, torch, DataLoader). MPS kernels are not guaranteed bit-deterministic. |
| Checkpoint criterion | Lowest validation loss |
| Runtime | **110 s** training (≈ 5.3 s/epoch), 1 min 55 s end-to-end |

## 4. Metrics

Computed by `ml/metrics.py` from a confusion matrix over pixels whose label is not 3. Label-3
pixels are excluded from loss and metrics and counted separately. Validation loss is mean
cross-entropy over non-ignored pixels.

**Best checkpoint: epoch 17, validation loss 0.0937.** Validation pixels: 2,992,099 valid,
269,341 ignored.

| Class | IoU | F1 / Dice | Precision | Recall | GT pixels |
|---|---:|---:|---:|---:|---:|
| no water | 0.966 | 0.983 | 0.984 | 0.982 | 2,849,217 |
| permanent water | 0.317 | 0.481 | 0.588 | 0.407 | 48,937 |
| **flood** | **0.560** | **0.718** | **0.657** | **0.791** | 93,945 |
| mean (3 classes) | 0.614 | 0.727 | | | |
| binary water (1 ∪ 2) | 0.497 | 0.664 | | | |

Pixel accuracy 0.967 (not meaningful; dominated by no-water).

**Flood by validation event:**

| Event | Flood IoU | F1 | Precision | Recall | TP / FP / FN |
|---|---:|---:|---:|---:|---|
| 279 Spain | 0 | 0 | 0 | n/a (no flood GT) | 0 / 27,376 / 0 |
| 437 France | 0.667 | 0.800 | 0.917 | 0.710 | 3,275 / 295 / 1,339 |
| 1111003 Djibouti | 0.008 | 0.017 | 0.011 | 0.032 | 18 / 1,558 / 542 |
| 1111008 Nicaragua | 0.722 | 0.839 | 0.882 | 0.799 | 70,972 / 9,535 / 17,799 |

Learning curve: validation flood IoU 0.07 (epoch 1) → 0.31 (3) → 0.51 (7) → 0.56 (17); train
loss 0.84 → 0.15; validation loss 2.50 → 0.094. Full history: `ml/runs/kurosiwo_experiment/history.json`.

## 5. Outputs

- Checkpoint: `ml/checkpoints/kurosiwo_unet_r18_best.pt` (57 MB, git-ignored): `model_state`,
  `config`, `epoch`, `val_loss`, `val_metrics`.
- Metrics: `ml/runs/kurosiwo_experiment/metrics.json`, `history.json` (git-ignored).
- Predictions: `ml/runs/kurosiwo_experiment/predictions/*.png`, 8 images (2 per validation event,
  those with the most flood pixels). Each shows pre1 VV (dB), post VV (dB), ground truth and
  prediction.

## 6. Interpretation and limitations

- **The model learns something real.** Flood metrics on held-out events are far above trivial
  (an all-no-water predictor scores flood IoU 0). Predictions on 1111008 follow the flood extent
  closely (see PNG).
- **Validation flood metrics are dominated by one event.** 94.5 % of flood GT pixels come from
  1111008, so the overall flood IoU 0.560 mostly reflects Nicaragua. Flood performance is
  **poor on Djibouti** (IoU 0.008) and only measured on 4,614 pixels for France.
- **False positives where there is no flood.** Event 279 has no flood GT, yet 27,376 pixels were
  predicted as flood. The visual example shows them on dark, sloped terrain, a relevant warning
  for steep Himalayan scenes.
- **Optimistic estimate.** The checkpoint was chosen on the same validation set used for
  reporting, and there is no separate held-out test evaluation yet.
- **Small and partial.** 252 train samples (≈ 1 % of Kuro Siwo's labelled training data), 14 of
  27 train events, 4 of 7 val events; samples come from 1–2 contiguous runs per event.
- **Not comparable to the paper.** Kuro Siwo reports flood F1 80.1 % (U-Net ResNet-50, full data,
  official test set). Our numbers use different data size, encoder, split and evaluation set.
- Permanent water is weak (IoU 0.317), and binary water F1 (0.664) is lower than flood F1 because
  permanent water and flood are confused.
- No Himalayan or steep-terrain events in training or validation. **Nothing here measures Trishuli
  performance.**

## 7. Commands

```bash
.venv/bin/python scripts/build_kurosiwo_experiment_data.py --dry-run   # plan + estimate
.venv/bin/python scripts/build_kurosiwo_experiment_data.py             # probe (cached) + stream
.venv/bin/python scripts/train_kurosiwo.py                             # 20 epochs, batch 16, seed 0
.venv/bin/python -m pytest -q
```

Probing was first run with an equivalent one-off call of `ml.kurosiwo.probe_shard(shard, 0.5e9,
workers=8)` over all shards, writing the same `probes.json` the builder caches.
