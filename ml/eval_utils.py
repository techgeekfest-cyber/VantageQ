"""Shared evaluation helpers: loss + confusion-matrix metrics per split/event, prediction PNGs.

Used by scripts/train_kurosiwo.py (validation) and scripts/evaluate_kurosiwo_test.py (test).
Named eval_utils to avoid confusion with the top-level evaluation/ folder (EMSR927, quarantined).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from ml.dataset import KuroSiwoDataset
from ml.kurosiwo import IGNORE_INDEX, load_sample
from ml.metrics import confusion, summarize


def describe(ds: KuroSiwoDataset) -> tuple[Counter, Counter, set]:
    """Events (sample counts), label pixel counts and grid IDs of a dataset."""
    events, labels, grids = Counter(), Counter(), set()
    for d in ds.dirs:
        s = load_sample(d)
        events[s["info"]["actid"]] += 1
        grids.add(s["info"]["grid_id"])
        v, c = np.unique(s["mask"], return_counts=True)
        labels.update(dict(zip(v.astype(int).tolist(), c.tolist())))
    return events, labels, grids


@torch.no_grad()
def evaluate(model, loader, device, by_event: list[int] | None = None) -> tuple[float, dict, dict]:
    """Return (mean loss over non-ignored pixels, overall metrics, per-event metrics)."""
    model.eval()
    loss_sum, n_pix = 0.0, 0
    cm_total, ignored_total = None, 0
    cm_event, ign_event = defaultdict(lambda: 0), defaultdict(int)
    i = 0
    for x, y in loader:
        out = model(x.to(device))
        yd = y.to(device)
        loss_sum += float(torch.nn.functional.cross_entropy(out, yd, ignore_index=IGNORE_INDEX, reduction="sum"))
        n_pix += int((yd != IGNORE_INDEX).sum())
        pred = out.argmax(1).cpu().numpy()
        for p, t in zip(pred, y.numpy()):
            cm, ign = confusion(p, t)
            cm_total = cm if cm_total is None else cm_total + cm
            ignored_total += ign
            if by_event is not None:
                cm_event[by_event[i]] = cm_event[by_event[i]] + cm
                ign_event[by_event[i]] += ign
            i += 1
    per_event = {int(a): summarize(cm_event[a], ign_event[a]) for a in cm_event}
    return loss_sum / max(n_pix, 1), summarize(cm_total, ignored_total), per_event


def save_predictions(model, ds: KuroSiwoDataset, device, out_dir: Path, per_event: int = 2,
                     split: str = "val", repo_root: Path | None = None) -> list[str]:
    """Save simple PNGs for the samples with the most flood pixels in each event (1 per row)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap(["#f2f2f2", "#2c7fb8", "#d7301f", "#9e9e9e"])  # no water, perm., flood, ignore
    by_event = defaultdict(list)
    for i, d in enumerate(ds.dirs):
        s = load_sample(d)
        by_event[s["info"]["actid"]].append((int((s["mask"] == 2).sum()), i))
    chosen = [i for a in sorted(by_event) for _, i in sorted(by_event[a], reverse=True)[:per_event]]

    model.eval()
    paths = []
    for i in chosen:
        s = load_sample(ds.dirs[i])
        x, y = ds[i]
        with torch.no_grad():
            pred = model(x[None].to(device)).argmax(1)[0].cpu().numpy()
        y = y.numpy()
        pred_vis = np.where(y == IGNORE_INDEX, IGNORE_INDEX, pred)
        db = lambda a: 10 * np.log10(np.clip(a, 1e-4, None))  # noqa: E731
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
        ax[0].imshow(db(s["sec1_vv"]), cmap="gray", vmin=-25, vmax=0)
        ax[0].set_title("pre1 VV (dB)")
        ax[1].imshow(db(s["flood_vv"]), cmap="gray", vmin=-25, vmax=0)
        ax[1].set_title("post VV (dB)")
        ax[2].imshow(y, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax[2].set_title("ground truth")
        ax[3].imshow(pred_vis, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax[3].set_title("prediction (ignore px grey)")
        for a in ax:
            a.axis("off")
        fig.suptitle(f"{split} event {s['info']['actid']}  {ds.dirs[i].name[:30]}   "
                     "white: no water  blue: permanent water  red: flood  grey: ignore (3)", fontsize=8)
        fig.tight_layout()
        p = out_dir / f"{ds.dirs[i].name}.png"
        fig.savefig(p, dpi=90)
        plt.close(fig)
        paths.append(str(p.relative_to(repo_root)) if repo_root else str(p))
    return paths


def fmt(m: dict) -> str:
    f = m["flood"]
    return (f"mIoU {m['mean_iou']:.3f}  flood IoU {f['iou']:.3f} F1 {f['f1']:.3f} "
            f"P {f['precision']:.3f} R {f['recall']:.3f}  water F1 {m['water_binary']['f1']:.3f}")
