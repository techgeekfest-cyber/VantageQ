"""Evaluate the FROZEN Kuro Siwo checkpoint on the held-out official TEST subset. No training.

Loads ml/checkpoints/kurosiwo_unet_r18_best.pt (selected on validation loss in
scripts/train_kurosiwo.py) and evaluates it once on data/external/kurosiwo_experiment/test/,
which must contain only official TEST events with no event or grid overlap with train/val.
Uses the shared preprocessing, dataset, model and metrics code. See docs/TEST_EVALUATION.md.

Usage:
  python scripts/evaluate_kurosiwo_test.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.dataset import KuroSiwoDataset  # noqa: E402
from ml.eval_utils import describe, evaluate, save_predictions  # noqa: E402
from ml.kurosiwo import LABEL_NAMES, load_sample, split_of  # noqa: E402
from ml.inference import load_frozen_model  # noqa: E402

DATA = REPO_ROOT / "data" / "external" / "kurosiwo_experiment"
CHECKPOINT = REPO_ROOT / "ml" / "checkpoints" / "kurosiwo_unet_r18_best.pt"
RUN_DIR = REPO_ROOT / "ml" / "runs" / "kurosiwo_test_eval"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def na(v: float) -> str:
    return "n/a" if v is None or math.isnan(v) else f"{v:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    test_ds = KuroSiwoDataset(args.data / "test")
    te_ev, te_lab, te_grid = describe(test_ds)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    trained_on = {int(a) for a in cfg["train_events"]} | {int(a) for a in cfg["val_events"]}
    other_grids = set()
    for split in ("train", "val"):
        if (args.data / split).exists():
            other_grids |= describe(KuroSiwoDataset(args.data / split))[2]
    problems = ([f"event {a} is not an official TEST event" for a in te_ev if split_of(a) != "test"]
                + [f"events also used in train/val: {sorted(set(te_ev) & trained_on)}"] * bool(set(te_ev) & trained_on)
                + [f"{len(te_grid & other_grids)} grids also in train/val"] * bool(te_grid & other_grids))
    if problems or not len(test_ds):
        print("refusing to evaluate:", problems or "empty test set")
        return 1

    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()[:16]
    print(f"frozen checkpoint {args.checkpoint.relative_to(REPO_ROOT)} (sha256 {sha}…), "
          f"epoch {ckpt['epoch']}, selected on val loss {ckpt['val_loss']:.4f}")
    print(f"test: {len(test_ds)} samples, {len(te_ev)} events {dict(sorted(te_ev.items()))}")
    print("   label pixels: " + ", ".join(f"{k} {LABEL_NAMES[k]} {v}" for k, v in sorted(te_lab.items())))

    device = pick_device(args.device)
    model, _ = load_frozen_model(args.checkpoint, device)  # weights come from the checkpoint
    loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    events = [load_sample(d)["info"]["actid"] for d in test_ds.dirs]
    loss, m, per_event = evaluate(model, loader, device, by_event=events)

    print(f"\ndevice {device} ({platform.machine()}, torch {torch.__version__})")
    print(f"test loss {loss:.4f}  valid px {m['valid_pixels']}  ignored px {m['ignored_pixels']}")
    print(f"mean (3 classes): IoU {na(m['mean_iou'])}  F1 {na(m['mean_f1'])}  "
          f"precision {na(m['mean_precision'])}  recall {na(m['mean_recall'])}")
    for name, c in m["per_class"].items():
        print(f"  {name:<16} IoU {na(c['iou'])}  F1 {na(c['f1'])}  P {na(c['precision'])}  R {na(c['recall'])}"
              f"  (gt px {m['class_pixels'][name]}, tp {c['tp']}, fp {c['fp']}, fn {c['fn']})")
    w = m["water_binary"]
    print(f"  binary water     IoU {na(w['iou'])}  F1 {na(w['f1'])}  P {na(w['precision'])}  R {na(w['recall'])}")
    print("\nper event:")
    for a, em in sorted(per_event.items()):
        f = em["flood"]
        print(f"  {a:>8}: mIoU {na(em['mean_iou'])}  flood IoU {na(f['iou'])} F1 {na(f['f1'])} "
              f"P {na(f['precision'])} R {na(f['recall'])}  flood gt {em['class_pixels']['flood']} "
              f"tp {f['tp']} fp {f['fp']} fn {f['fn']}  | perm. water IoU {na(em['per_class']['permanent water']['iou'])}"
              f"  ignored {em['ignored_pixels']}")

    vis = save_predictions(model, test_ds, device, RUN_DIR / "predictions", split="test", repo_root=REPO_ROOT)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "metrics.json").write_text(json.dumps(
        {"checkpoint": str(args.checkpoint.relative_to(REPO_ROOT)), "checkpoint_sha256_16": sha,
         "checkpoint_epoch": ckpt["epoch"], "test_events": dict(sorted(te_ev.items())),
         "test_samples": len(test_ds), "test_loss": loss, "test": m, "test_per_event": per_event,
         "device": str(device), "predictions": vis}, indent=1, default=float))
    print(f"\n{len(vis)} prediction PNGs in {(RUN_DIR / 'predictions').relative_to(REPO_ROOT)}; "
          f"metrics in {(RUN_DIR / 'metrics.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
