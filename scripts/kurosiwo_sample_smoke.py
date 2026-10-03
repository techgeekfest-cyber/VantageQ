"""Kuro Siwo dataset smoke test: fetch a few labelled GRD samples and inspect them. No training.

Streams the first N samples of one labelled Hugging Face webdataset shard and stops reading, so
only ~1.8 MB per sample is downloaded (the shard itself is several GB). Samples are saved under
data/external/kurosiwo_smoke/ (git-ignored), inspected, and converted with the shared model-input
preprocessing (ml/preprocessing.py).

See docs/KUROSIWO_DATASET.md and docs/KUROSIWO_PREPROCESSING.md.

Usage:
  python scripts/kurosiwo_sample_smoke.py              # fetch 3 samples (max 30 MB), inspect
  python scripts/kurosiwo_sample_smoke.py --n 5
  python scripts/kurosiwo_sample_smoke.py --inspect-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.kurosiwo import (  # noqa: E402
    HF_REPO, LABEL_NAMES, SAMPLE_BYTES, SAR_FIELDS, load_sample, sample_dirs, sar_inputs, split_of,
    stream_samples,
)
from ml.preprocessing import CHANNEL_LABELS, build_model_input  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "external" / "kurosiwo_smoke"
DEFAULT_SHARD = "train_GRD/shard-00000.tar"


def inspect(d: Path) -> dict:
    s = load_sample(d)
    info = s["info"]
    valid = s["valid_mask"] == 1
    print(f"\nSample {d.name}  event {info['actid']} (split: {split_of(info['actid'])})  aoi {info['aoiid']}"
          f"  flood_date {info['flood_date']}  pflood {info['pflood']:.2f} %  pwater {info['pwater']:.2f} %")
    dates = {k: v["source_date"] for k, v in info["sources"].items()}
    print(f"  acquisitions: post MS1 {dates.get('MS1')}  pre1 SL1 {dates.get('SL1')}  pre2 SL2 {dates.get('SL2')}")
    print(f"  valid pixels: {100 * valid.mean():.2f} %")
    print(f"  {'field':<11}{'shape':<12}{'dtype':<9}{'min':>9}{'max':>9}{'NaN %':>8}{'zero %':>8}")
    for f in (*SAR_FIELDS.values(), "dem", "valid_mask", "mask"):
        a = s[f]
        print(f"  {f:<11}{str(a.shape):<12}{str(a.dtype):<9}{np.nanmin(a):>9.4g}{np.nanmax(a):>9.4g}"
              f"{100 * np.isnan(a).mean():>8.2f}{100 * (a == 0).mean():>8.2f}")

    values, counts = np.unique(s["mask"], return_counts=True)
    print("  label counts: " + ", ".join(
        f"{int(v)} {LABEL_NAMES.get(int(v), 'UNEXPECTED')}: {c} ({100 * c / s['mask'].size:.2f} %)"
        for v, c in zip(values, counts)))
    if (~valid).any():
        print(f"  label values on invalid pixels: {np.unique(s['mask'][~valid]).astype(int).tolist()}")

    x = build_model_input(sar_inputs(s))
    print(f"  model input: shape {x.shape} dtype {x.dtype}  order {list(CHANNEL_LABELS)}")
    for i, lab in enumerate(CHANNEL_LABELS):
        print(f"    ch{i} {lab:<8} min {x[i].min():7.3f}  max {x[i].max():7.3f}  mean {x[i].mean():7.3f}")
    return {"key": d.name, "actid": info["actid"], "labels": dict(zip(values.astype(int).tolist(), counts.tolist())),
            "input_shape": list(x.shape)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3, help="number of samples to fetch (default 3)")
    ap.add_argument("--shard", default=DEFAULT_SHARD, help=f"webdataset shard (default {DEFAULT_SHARD})")
    ap.add_argument("--max-mb", type=float, default=30, help="hard cap on streamed megabytes (default 30)")
    ap.add_argument("--inspect-only", action="store_true", help="only inspect already-saved samples")
    args = ap.parse_args()

    if not args.inspect_only:
        print(f"{HF_REPO}/{args.shard}: streaming only the first {args.n} samples "
              f"(≈{args.n * SAMPLE_BYTES / 1e6:.0f} MB, hard cap {args.max_mb:.0f} MB)")
        saved, nbytes = stream_samples(args.shard, OUT_DIR, args.n, max_bytes=args.max_mb * 1e6)
        print(f"streamed {nbytes / 1e6:.1f} MB, saved {len(saved)} complete samples to {OUT_DIR.relative_to(REPO_ROOT)}")

    dirs = sample_dirs(OUT_DIR)
    if not dirs:
        print("no samples found")
        return 1
    results = [inspect(d) for d in dirs]
    (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
