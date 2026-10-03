# Held-out Test Evaluation: Frozen Kuro Siwo U-Net

Run date: **2026-10-04**.

**This evaluation was performed after model training and was not used for model selection or
hyperparameter tuning.** The checkpoint was frozen before any test data was downloaded. It was
evaluated once, with no retraining, tuning or architecture change. No Trishuli data, EMSR927 or
other published map was used.

---

## 1. Frozen checkpoint

| Item | Value |
|---|---|
| File | `ml/checkpoints/kurosiwo_unet_r18_best.pt` (git-ignored, 57 MB) |
| SHA-256 (first 16 hex) | `0e28f6a44b774c5b` (unchanged after evaluation) |
| Origin | `docs/TRAINING_EXPERIMENT.md`: epoch 17 of 20, chosen on lowest **validation** loss (0.0937) |
| Model | `smp.Unet("resnet18", in_channels=6, classes=3)`, weights loaded from the checkpoint |
| Preprocessing | unchanged `ml/preprocessing.py` (clamp 0.15, fixed VV/VH normalisation, 6-channel order) |

## 2. Test data

Official **TEST** events only (`TEST_ACTS` in `ml/kurosiwo.py`), located with the cached shard
probes and streamed with `scripts/build_kurosiwo_experiment_data.py --splits test`
(≤ 2 positions per event, de-duplicated by `grid_id`).

| Event | Region | Samples | Flood GT px | Perm. water GT px | Ignored (label 3) px | Pre2 / Pre1 / Post |
|---|---|---:|---:|---:|---:|---|
| 277 | Greece | 16 | 33,924 | 17,393 | 0 | 2017-07-30 / 2017-08-29 / 2018-03-27 |
| 321 | Honduras | 16 | 7,534 | 15,838 | 23,520 | 2017-12-11 / 2017-12-23 / 2018-10-07 |
| 445 | Romania | 15 | 11,991 | 12,934 | 80,878 | 2020-06-02 / 2020-06-14 / 2020-06-26 |
| 562 | Australia (Burketown) | 15 | **199,188** | 14,792 | 49,236 | 2021-08-07 / 2021-08-19 / 2022-02-03 |
| 1111007 | Nepal (lowlands, "Patna") | 16 | 92,205 | 33,575 | 7,329 | 2019-07-20 / 2019-08-01 / 2019-09-18 |
| 1111013 | USA (St Louis) | 16 | 9,806 | 8,684 | 0 | 2017-04-10 / 2017-04-22 / 2017-05-04 |
| **Total** | **6 events** | **94** | 354,648 | 103,216 | 160,963 | |

Official test events **not** included (not located at 0.5 GB probe spacing): 205, 411, 561, 1111002.

Leakage checks (the script refuses otherwise): every test event is in `TEST_ACTS`; no test event
is among the checkpoint's train/val events; no test `grid_id` appears in the train/val subsets.

**Size:** 175.1 MB streamed (no new probing; cached probes reused). **166 MB** on disk in
`data/external/kurosiwo_experiment/test/` (git-ignored).

## 3. Results (94 samples, 6 events)

Metrics from `ml/metrics.py`: confusion matrix over pixels whose label is not 3. Valid pixels:
4,555,581. Ignored pixels: 160,963. Test loss = mean cross-entropy over valid pixels.

| | Value |
|---|---:|
| **Test loss** | **0.1663** (validation: 0.0937) |
| Mean IoU (3 classes) | 0.484 |
| Mean F1 / Dice | 0.587 |
| Mean precision | 0.685 |
| Mean recall | 0.589 |

| Class | IoU | F1 / Dice | Precision | Recall | TP / FP / FN |
|---|---:|---:|---:|---:|---|
| no water | 0.960 | 0.979 | 0.964 | 0.995 | 4,076,758 / 150,616 / 20,959 |
| permanent water | 0.185 | 0.312 | 0.239 | 0.448 | 46,249 / 146,947 / 56,967 |
| **flood** | **0.307** | **0.470** | **0.852** | **0.324** | 115,000 / 20,011 / 239,648 |
| binary water (1 ∪ 2) | 0.642 | 0.782 | 0.936 | 0.671 | |

Confusion matrix (rows ground truth, columns prediction; no water / permanent water / flood) is
in `ml/runs/kurosiwo_test_eval/metrics.json`. Of the 354,648 flood GT pixels: 115,000 predicted
flood, **140,548 predicted permanent water**, 99,100 predicted no water.

### Per event

IoU/F1/recall would be "n/a" for an event without flood GT; every test event has flood GT.

| Event | Flood IoU | Flood F1 | Flood P | Flood R | Flood TP / FP / FN | Flood GT → perm. water | Perm. water IoU | mIoU |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| 277 Greece | 0.673 | 0.805 | 0.760 | 0.855 | 29,021 / 9,189 / 4,903 | 0.3 % | 0.258 | 0.632 |
| 321 Honduras | 0.192 | 0.322 | 0.664 | 0.213 | 1,604 / 810 / 5,930 | 4.1 % | 0.269 | 0.480 |
| 445 Romania | 0.691 | 0.817 | 0.750 | 0.898 | 10,769 / 3,599 / 1,222 | 2.2 % | 0.288 | 0.653 |
| 562 Australia | **0.066** | 0.124 | 0.923 | **0.067** | 13,296 / 1,111 / 185,892 | **69.7 %** | 0.065 | 0.342 |
| 1111007 Nepal lowlands | 0.603 | 0.752 | 0.920 | 0.636 | 58,624 / 5,078 / 33,581 | 0.9 % | 0.604 | 0.714 |
| 1111013 USA | 0.168 | 0.288 | 0.883 | 0.172 | 1,686 / 224 / 8,120 | 1.4 % | 0.183 | 0.444 |

Event-mean (macro) flood IoU over the 6 events: **0.399**.

## 4. Interpretation

- **The model generalises partially.** On 3 of 6 unseen events (Greece, Romania, Nepal lowlands)
  flood IoU is 0.60–0.69. On the other 3 it is poor (0.07–0.19).
- **Precision is high, recall is low.** When the model predicts flood it is usually right
  (P 0.852), but it misses about two thirds of flooded pixels (R 0.324).
- **One event dominates the total.** Australia 562 holds 56 % of all flood GT pixels. There the
  model mostly **detects the water but labels it permanent water** (69.7 % of flood GT). Binary
  water F1 is 0.853 for this event, yet flood IoU is 0.066. Its pre-event images are from the
  dry season (Aug 2021); the cause of the confusion is not established. Excluding 562, flood
  recall would be about 0.65 (101,704 / 155,460). That is an observation, not a re-scored result.
- **Honduras and USA are under-detected** (flood recall 0.21 / 0.17), mostly flood → no water.
- **Worse than validation, as expected for genuinely held-out events.** Test loss 0.166 vs
  validation 0.094; flood IoU 0.307 vs 0.560 (the validation figure was optimistic, chosen on the
  same set).

## 5. Limitations

- Small: 94 samples, 6 of 10 official test events, ≤ 2 contiguous runs of tiles per event. Per-event
  numbers rest on 15–16 neighbouring tiles each and can swing with tile choice.
- The model itself is small-data (252 training samples, ResNet-18). Not comparable with the paper
  (flood F1 80.1 % with full data, ResNet-50, full official test set).
- The flood vs permanent-water distinction is a major weakness. For VantageQ's flood map,
  "new water" (flood ∪ permanent water, minus a pre-event water reference) may need separate
  consideration; not decided here.
- No steep/Himalayan terrain in any split; nothing here measures Trishuli performance.

## 6. Outputs and commands

- Metrics: `ml/runs/kurosiwo_test_eval/metrics.json` (overall, per event, confusion matrices,
  checkpoint hash). Git-ignored.
- Predictions: `ml/runs/kurosiwo_test_eval/predictions/*.png`, 12 images, 2 per test event (the
  tiles with the most flood pixels), each showing pre1 VV, post VV, ground truth and prediction.

```bash
.venv/bin/python scripts/build_kurosiwo_experiment_data.py --splits test --dry-run   # estimate: ≈171 MB
.venv/bin/python scripts/build_kurosiwo_experiment_data.py --splits test             # stream 94 samples
.venv/bin/python scripts/evaluate_kurosiwo_test.py                                   # one evaluation
```

Code changes made for this evaluation (no model or preprocessing change): `evaluate`,
`describe`, `save_predictions` moved unchanged from `scripts/train_kurosiwo.py` into
`ml/eval_utils.py` (shared); `ml/metrics.py` now reports IoU/F1/recall as NaN for a class with no
ground truth, and adds mean precision/recall and the confusion matrix.
