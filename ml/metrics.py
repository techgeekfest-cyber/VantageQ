"""Segmentation metrics from a confusion matrix, excluding ignore-index pixels.

Rows = ground truth, columns = prediction, over classes 0..NUM_CLASSES-1. Pixels whose label is
IGNORE_INDEX are not counted anywhere (they are reported separately as `ignored_pixels`).
"""

from __future__ import annotations

import numpy as np

from ml.kurosiwo import IGNORE_INDEX, LABELS, NUM_CLASSES

FLOOD = 2
WATER = (1, 2)  # permanent water + flood, Kuro Siwo's binary "water" task


def confusion(pred: np.ndarray, target: np.ndarray, num_classes: int = NUM_CLASSES,
              ignore_index: int = IGNORE_INDEX) -> tuple[np.ndarray, int]:
    """Return (num_classes x num_classes confusion matrix, number of ignored pixels)."""
    pred, target = np.asarray(pred).ravel(), np.asarray(target).ravel()
    keep = target != ignore_index
    cm = np.bincount(num_classes * target[keep].astype(np.int64) + pred[keep].astype(np.int64),
                     minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm, int((~keep).sum())


def _scores(tp: float, fp: float, fn: float) -> dict:
    def div(a, b):
        return float(a / b) if b else float("nan")
    return {"iou": div(tp, tp + fp + fn), "f1": div(2 * tp, 2 * tp + fp + fn),
            "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def summarize(cm: np.ndarray, ignored: int = 0) -> dict:
    """Per-class, mean and binary-water metrics from a confusion matrix."""
    per_class = {}
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        per_class[LABELS.get(c, str(c))] = _scores(tp, cm[:, c].sum() - tp, cm[c, :].sum() - tp)
    w = list(WATER)
    tp_w = cm[np.ix_(w, w)].sum()
    water = _scores(tp_w, cm[:, w].sum() - tp_w, cm[w, :].sum() - tp_w)
    ious = [m["iou"] for m in per_class.values()]
    f1s = [m["f1"] for m in per_class.values()]
    total = int(cm.sum())
    return {
        "valid_pixels": total,
        "ignored_pixels": int(ignored),
        "class_pixels": {LABELS.get(c, str(c)): int(cm[c, :].sum()) for c in range(cm.shape[0])},
        "pixel_accuracy": float(np.trace(cm) / total) if total else float("nan"),
        "mean_iou": float(np.nanmean(ious)),
        "mean_f1": float(np.nanmean(f1s)),
        "per_class": per_class,
        "flood": per_class[LABELS[FLOOD]],
        "water_binary": water,
    }
