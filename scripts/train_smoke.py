"""Training pipeline smoke test on the tiny Kuro Siwo subset. NOT an accuracy experiment.

Proves the end-to-end path works: dataset -> DataLoader -> official U-Net -> cross-entropy loss ->
backward -> optimizer step, for 1-3 tiny epochs. No checkpoint is saved and no metric is reported;
the resulting model must not be treated as trained.

Usage:
  python scripts/train_smoke.py                       # 2 epochs, batch 8, auto device
  python scripts/train_smoke.py --epochs 1 --device cpu --no-pretrained
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.dataset import KuroSiwoDataset  # noqa: E402
from ml.kurosiwo import IGNORE_INDEX, LABEL_NAMES, load_sample, split_of  # noqa: E402
from ml.model import build_unet  # noqa: E402

DEFAULT_DATA = REPO_ROOT / "data" / "external" / "kurosiwo_subset" / "samples"
OUT = REPO_ROOT / "ml" / "runs" / "train_smoke" / "result.json"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--epochs", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3, help="official Adam learning rate")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    ap.add_argument("--no-pretrained", action="store_true", help="random encoder init instead of ImageNet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.time()
    ds = KuroSiwoDataset(args.data)
    if len(ds) == 0:
        print(f"no samples in {args.data}; run scripts/build_kurosiwo_subset.py first")
        return 1

    events, labels = Counter(), Counter()
    for d in ds.dirs:
        s = load_sample(d)
        events[s["info"]["actid"]] += 1
        v, c = np.unique(s["mask"], return_counts=True)
        labels.update(dict(zip(v.astype(int).tolist(), c.tolist())))
    print(f"dataset: {len(ds)} samples from {len(events)} events {dict(sorted(events.items()))}")
    not_train = sorted(a for a in events if split_of(a) != "train")
    if not_train:
        print(f"refusing to train on non-train-split events {not_train}")
        return 1
    print("labels observed: " + ", ".join(f"{k} {LABEL_NAMES.get(k, 'UNEXPECTED')}: {v}" for k, v in sorted(labels.items())))

    device = pick_device(args.device)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         generator=torch.Generator().manual_seed(args.seed))
    model = build_unet(None if args.no_pretrained else "imagenet").to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"device: {device}  ({platform.machine()}, torch {torch.__version__})  "
          f"encoder weights: {'none' if args.no_pretrained else 'imagenet'}")

    result = {"samples": len(ds), "events": dict(sorted(events.items())), "labels": dict(sorted(labels.items())),
              "device": str(device), "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
              "pretrained": not args.no_pretrained, "batch_losses": [], "epoch_mean_loss": []}
    t_train = time.time()
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            grad = model.encoder.conv1.weight.grad
            backward_ok = grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0
            opt.step()
            losses.append(loss.item())
            if epoch == 0 and step == 0:
                result |= {"x_shape": list(x.shape), "x_dtype": str(x.dtype), "y_shape": list(y.shape),
                           "y_dtype": str(y.dtype), "output_shape": list(out.shape),
                           "initial_loss": loss.item(), "backward_ok_first_step": backward_ok}
                print(f"first batch: X {tuple(x.shape)} {x.dtype}  y {tuple(y.shape)} {y.dtype}  "
                      f"output {tuple(out.shape)}  loss {loss.item():.4f}  backward ok: {backward_ok}")
            if not backward_ok:
                print("backward produced missing/non-finite gradients; stopping")
                return 2
        result["batch_losses"] += losses
        result["epoch_mean_loss"].append(float(np.mean(losses)))
        print(f"epoch {epoch + 1}: {len(losses)} steps, mean loss {np.mean(losses):.4f}, last batch {losses[-1]:.4f}")

    result["final_loss"] = result["batch_losses"][-1]
    result["train_seconds"] = round(time.time() - t_train, 1)
    result["total_seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"initial loss {result['initial_loss']:.4f} -> final batch loss {result['final_loss']:.4f}; "
          f"train {result['train_seconds']} s, total {result['total_seconds']} s")
    print(f"result written to {OUT.relative_to(REPO_ROOT)} (no checkpoint saved; not a trained model)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
