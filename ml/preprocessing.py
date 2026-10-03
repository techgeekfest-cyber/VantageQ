"""The single model-input preprocessing used for Kuro Siwo training, validation and Trishuli inference.

Reproduces the official Kuro Siwo loader (docs/KUROSIWO_PREPROCESSING.md):
clamp linear sigma0 to [0, 0.15], NaN -> 0.15, then (x - mean) / std with fixed global VV/VH
statistics, stacked as [post VV, post VH, pre1 VV, pre1 VH, pre2 VV, pre2 VH].
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

CLAMP = 0.15
MEAN = {"vv": 0.0953, "vh": 0.0264}
STD = {"vv": 0.0427, "vh": 0.0215}

# Canonical input names (pre1 = most recent pre-event acquisition, pre2 = the one before it).
CHANNELS = ("post_vv", "post_vh", "pre1_vv", "pre1_vh", "pre2_vv", "pre2_vh")
CHANNEL_LABELS = ("post VV", "post VH", "pre1 VV", "pre1 VH", "pre2 VV", "pre2 VH")


def normalize_band(x: np.ndarray, pol: str) -> np.ndarray:
    """Clamp one linear-sigma0 band to [0, CLAMP], replace NaN by CLAMP, standardise."""
    x = np.clip(np.asarray(x, dtype="float32"), 0.0, CLAMP)
    x = np.nan_to_num(x, nan=CLAMP)
    return (x - MEAN[pol]) / STD[pol]


def build_model_input(sar: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return float32 [6, H, W] model input from linear-sigma0 bands keyed by CHANNELS."""
    return np.stack([normalize_band(sar[name], name[-2:]) for name in CHANNELS]).astype("float32")
