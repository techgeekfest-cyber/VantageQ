"""First real Kuro Siwo training experiment: event-separated train/val, metrics, best checkpoint.

Train and validation samples must come from disjoint official TRAIN / VAL events (checked).
Model: official U-Net (ml/model.py). Preprocessing: ml/preprocessing.py via ml/dataset.py.
Validation metrics exclude label 3 (ignore_index). The best checkpoint is chosen by validation
loss. See docs/TRAINING_EXPERIMENT.md.

Usage:
  python scripts/train_kurosiwo.py                     # 20 epochs, batch 16, auto device
  python scripts/train_kurosiwo.py --epochs 10 --no-flip
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.dataset import KuroSiwoDataset  # noqa: E402
from ml.kurosiwo import IGNORE_INDEX, LABEL_NAMES, load_sample, split_of  # noqa: E402
from ml.metrics import confusion, summarize  # noqa: E402
from ml.model import build_unet  # noqa: E402

DATA = REPO_ROOT / "data" / "external" / "kurosiwo_experiment"
RUN_DIR = REPO_ROOT / "ml" / "runs" / "kurosiwo_experiment"
CHECKPOINT = REPO_ROOT / "ml" / "checkpoints" / "kurosiwo_unet_r18_best.pt"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe(ds: KuroSiwoDataset) -> tuple[Counter, Counter, set]:
    events, labels, grids = Counter(), Counter(), set()
    for d in ds.dirs:
        s = load_sample(d)
        events[s["info"]["actid"]] += 1
        grids.add(s["info"]["grid_id"])
        v, c = np.unique(s["mask"], return_counts=True)
        labels.update(dict(zip(v.astype(int).tolist(), c.tolist())))
    return events, labels, grids


def random_flip(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Same random horizontal/vertical flip for the whole batch (label-preserving)."""
    if random.random() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if random.random() < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    return x, y


@torch.no_grad()
def evaluate(model, loader, device, by_event: list[int] | None = None) -> tuple[float, dict, dict]:
    """Return (mean val loss over non-ignored pixels, overall metrics, per-event metrics)."""
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


def save_predictions(model, ds: KuroSiwoDataset, device, out_dir: Path, per_event: int = 2) -> list[str]:
    """Save simple PNGs for the validation samples with most flood pixels in each event."""
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
        fig.suptitle(f"val event {s['info']['actid']}  {ds.dirs[i].name[:30]}   "
                     "white: no water  blue: permanent water  red: flood  grey: ignore (3)", fontsize=8)
        fig.tight_layout()
        p = out_dir / f"{ds.dirs[i].name}.png"
        fig.savefig(p, dpi=90)
        plt.close(fig)
        paths.append(str(p.relative_to(REPO_ROOT)))
    return paths


def fmt(m: dict) -> str:
    f = m["flood"]
    return (f"mIoU {m['mean_iou']:.3f}  flood IoU {f['iou']:.3f} F1 {f['f1']:.3f} "
            f"P {f['precision']:.3f} R {f['recall']:.3f}  water F1 {m['water_binary']['f1']:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3, help="official Adam learning rate")
    ap.add_argument("--no-flip", action="store_true", help="disable random flips")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ds, val_ds = KuroSiwoDataset(args.data / "train"), KuroSiwoDataset(args.data / "val")
    tr_ev, tr_lab, tr_grid = describe(train_ds)
    va_ev, va_lab, va_grid = describe(val_ds)
    problems = ([f"train event {a} not in official train split" for a in tr_ev if split_of(a) != "train"]
                + [f"val event {a} not in official val split" for a in va_ev if split_of(a) != "val"]
                + [f"events in both splits: {sorted(set(tr_ev) & set(va_ev))}"] * bool(set(tr_ev) & set(va_ev))
                + [f"{len(tr_grid & va_grid)} grids in both splits"] * bool(tr_grid & va_grid))
    if problems or not len(train_ds) or not len(val_ds):
        print("refusing to train:", problems or "empty split")
        return 1
    for name, ev, lab in (("train", tr_ev, tr_lab), ("val", va_ev, va_lab)):
        print(f"{name}: {sum(ev.values())} samples, {len(ev)} events {dict(sorted(ev.items()))}")
        print("   label pixels: " + ", ".join(f"{k} {LABEL_NAMES[k]} {v}" for k, v in sorted(lab.items())))

    device = pick_device(args.device)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=gen)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    val_events = [load_sample(d)["info"]["actid"] for d in val_ds.dirs]

    model = build_unet("imagenet").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    print(f"device {device} ({platform.machine()}, torch {torch.__version__}); epochs {args.epochs}, "
          f"batch {args.batch_size}, lr {args.lr} cosine, flips {not args.no_flip}, seed {args.seed}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()} | {
        "device": str(device), "encoder": "resnet18", "encoder_weights": "imagenet", "classes": 3,
        "ignore_index": IGNORE_INDEX, "loss": "CrossEntropyLoss(ignore_index=3)", "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR", "checkpoint_criterion": "lowest validation loss",
        "train_events": dict(sorted(tr_ev.items())), "val_events": dict(sorted(va_ev.items()))}
    history, best, t0 = [], None, time.time()
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        model.train()
        losses = []
        for x, y in train_loader:
            if not args.no_flip:
                x, y = random_flip(x, y)
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        val_loss, m, _ = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - te, 1), "val": m}
        history.append(row)
        improved = best is None or val_loss < best["val_loss"]
        if improved:
            best = {"epoch": epoch, "val_loss": val_loss}
            torch.save({"model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                        "config": config, "epoch": epoch, "val_loss": val_loss, "val_metrics": m}, CHECKPOINT)
        print(f"epoch {epoch:2d}  train {row['train_loss']:.4f}  val {val_loss:.4f}  {fmt(m)}  "
              f"{row['seconds']}s{'  *best' if improved else ''}", flush=True)
    train_seconds = round(time.time() - t0, 1)

    # Reload the saved best checkpoint and re-evaluate (also checks the checkpoint loads).
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    val_loss, m, per_event = evaluate(model, val_loader, device, by_event=val_events)
    print(f"\nbest checkpoint: epoch {ckpt['epoch']}, val loss {val_loss:.4f}")
    print(f"  overall: {fmt(m)}  pixel acc {m['pixel_accuracy']:.3f}")
    print(f"  valid pixels {m['valid_pixels']}, ignored pixels {m['ignored_pixels']}")
    for name, c in m["per_class"].items():
        print(f"  {name:<16} IoU {c['iou']:.3f}  F1 {c['f1']:.3f}  P {c['precision']:.3f}  R {c['recall']:.3f}"
              f"  (gt pixels {m['class_pixels'][name]})")
    for a, em in sorted(per_event.items()):
        f = em["flood"]
        print(f"  event {a}: flood IoU {f['iou']:.3f} F1 {f['f1']:.3f} P {f['precision']:.3f} R {f['recall']:.3f}"
              f"  (flood gt px {em['class_pixels']['flood']})")

    vis = save_predictions(model, val_ds, device, RUN_DIR / "predictions")
    (RUN_DIR / "history.json").write_text(json.dumps(history, indent=1))
    (RUN_DIR / "metrics.json").write_text(json.dumps(
        {"config": config, "best_epoch": ckpt["epoch"], "best_val_loss": val_loss, "val": m,
         "val_per_event": per_event, "train_seconds": train_seconds,
         "checkpoint": str(CHECKPOINT.relative_to(REPO_ROOT)), "predictions": vis}, indent=1))
    print(f"\ntraining {train_seconds} s; checkpoint {CHECKPOINT.relative_to(REPO_ROOT)}; "
          f"{len(vis)} prediction PNGs in {(RUN_DIR / 'predictions').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
