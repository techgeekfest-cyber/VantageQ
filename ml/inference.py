"""Inference helpers: frozen-checkpoint loading, per-date channel assembly, tiled prediction.

Tiling keeps the native raster resolution: the 6-channel input is reflect-padded (never resized),
cut into TILE x TILE windows with overlap, and softmax probabilities are averaged where windows
overlap. The last window on each axis is aligned to the padded edge so every pixel is covered.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from ml.model import build_unet
from ml.preprocessing import CHANNELS

TILE = 224  # training tile size


def sar_inputs_from_dates(post: Mapping[str, np.ndarray], pre1: Mapping[str, np.ndarray],
                          pre2: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map per-date {'VV', 'VH'} linear-sigma0 arrays to ml.preprocessing CHANNELS keys.

    pre1 is the most recent pre-event acquisition, pre2 the one before (Kuro Siwo SL1 / SL2).
    """
    dates = {"post": post, "pre1": pre1, "pre2": pre2}
    return {name: dates[name.split("_")[0]][name.split("_")[1].upper()] for name in CHANNELS}


def load_frozen_model(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Build the U-Net without downloading encoder weights and load the checkpoint's state."""
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_unet(encoder_weights=None)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


def _starts(length: int, tile: int, stride: int) -> list[int]:
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


@torch.no_grad()
def predict_tiled(model: torch.nn.Module, x: np.ndarray, device: torch.device, tile: int = TILE,
                  overlap: int = 64, margin: int = 32, batch_size: int = 8) -> np.ndarray:
    """Softmax class probabilities [C, H, W] for a model input x [6, H, W] of any size.

    `margin` pixels of reflect padding on every side give border pixels image-like context;
    extra bottom/right padding makes the padded image at least one tile large.
    """
    if not 0 <= overlap < tile:
        raise ValueError("overlap must be in [0, tile)")
    _, h, w = x.shape
    pad_h, pad_w = max(tile - (h + 2 * margin), 0), max(tile - (w + 2 * margin), 0)
    xp = np.pad(x, ((0, 0), (margin, margin + pad_h), (margin, margin + pad_w)), mode="reflect")
    hp, wp = xp.shape[1:]
    stride = tile - overlap
    windows = [(r, c) for r in _starts(hp, tile, stride) for c in _starts(wp, tile, stride)]

    model.eval()
    acc, count = None, np.zeros((hp, wp), np.float32)
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        xb = torch.from_numpy(np.stack([xp[:, r:r + tile, c:c + tile] for r, c in batch])).to(device)
        probs = torch.softmax(model(xb), dim=1).cpu().numpy()
        if acc is None:
            acc = np.zeros((probs.shape[1], hp, wp), np.float32)
        for (r, c), p in zip(batch, probs):
            acc[:, r:r + tile, c:c + tile] += p
            count[r:r + tile, c:c + tile] += 1
    out = acc / count
    return out[:, margin:margin + h, margin:margin + w]
