"""PyTorch dataset over saved Kuro Siwo samples, using the shared model-input preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ml.kurosiwo import load_sample, sample_dirs, sar_inputs
from ml.preprocessing import build_model_input


class KuroSiwoDataset(torch.utils.data.Dataset):
    """Returns X float32 [6, 224, 224] and y int64 [224, 224].

    y: 0 no water, 1 permanent water, 2 flood; 3 also occurs and is excluded by the loss
    (ignore_index=3). Labels are used as stored, like the official loader.
    """

    def __init__(self, root: Path):
        self.dirs = sample_dirs(Path(root))

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = load_sample(self.dirs[i])
        x = build_model_input(sar_inputs(s))
        y = s["mask"].astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)
