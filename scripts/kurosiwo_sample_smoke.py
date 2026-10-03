"""Kuro Siwo dataset smoke test: fetch a few labelled GRD samples and inspect them. No training.

Streams the first N samples of one labelled Hugging Face webdataset shard and stops reading, so
only ~1.8 MB per sample is downloaded (the shard itself is several GB). Samples are saved as
.npy files under data/external/kurosiwo_smoke/ (git-ignored). Then each sample is inspected and
converted into the 6-channel model input exactly as the official Kuro Siwo loader does.

See docs/KUROSIWO_DATASET.md and docs/KUROSIWO_PREPROCESSING.md.

Usage:
  python scripts/kurosiwo_sample_smoke.py              # fetch 3 samples (max 30 MB), inspect
  python scripts/kurosiwo_sample_smoke.py --n 5
  python scripts/kurosiwo_sample_smoke.py --inspect-only
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "external" / "kurosiwo_smoke"

HF_REPO = "orion-ai-lab/Kuro-Siwo-Webdataset"
DEFAULT_SHARD = "train_GRD/shard-00000.tar"

# Fields of one labelled GRD webdataset sample (Hugging Face dataset card).
SAR_FIELDS = ("flood_vv", "flood_vh", "sec1_vv", "sec1_vh", "sec2_vv", "sec2_vh")
FIELDS = SAR_FIELDS + ("dem", "mask", "valid_mask", "info.json")

# Official model input order: segmentation_trainer.py concatenates (post, pre_event_1, pre_event_2),
# each stacked as [vv, vh] by Dataset.concat. flood = MS1 (post), sec1 = SL1 (pre1), sec2 = SL2 (pre2).
MODEL_INPUT_ORDER = ("flood_vv", "flood_vh", "sec1_vv", "sec1_vh", "sec2_vv", "sec2_vh")
CHANNEL_LABELS = ("post VV", "post VH", "pre1 VV", "pre1 VH", "pre2 VV", "pre2 VH")

# Official configs/train/data_config.json (effective values; see docs/KUROSIWO_PREPROCESSING.md).
CLAMP = 0.15
MEAN = {"vv": 0.0953, "vh": 0.0264}
STD = {"vv": 0.0427, "vh": 0.0215}
CLASSES = {0: "no water", 1: "permanent water", 2: "flood"}
IGNORE_INDEX = 3  # used by the official loss/metrics, never assigned by the official loader
TRAIN_ACTS = {130, 470, 555, 118, 174, 324, 421, 554, 427, 518, 502, 498, 497, 496, 492, 147, 267,
              273, 275, 417, 567, 1111011, 1111004, 1111009, 1111010, 1111006, 1111005}
VAL_ACTS = {514, 559, 279, 520, 437, 1111003, 1111008}
TEST_ACTS = {321, 561, 445, 562, 411, 1111002, 277, 1111007, 205, 1111013}


# --------------------------------------------------------------------------- fetch


def shard_size(shard: str) -> int:
    url = f"https://huggingface.co/api/datasets/{HF_REPO}/paths-info/main"
    req = urllib.request.Request(url, data=json.dumps({"paths": [shard]}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)[0]["size"]


class _CountingReader(io.RawIOBase):
    """Wraps an HTTP response, counts bytes and refuses to read beyond a hard cap."""

    def __init__(self, resp, max_bytes: int):
        self.resp, self.max_bytes, self.n = resp, max_bytes, 0

    def readable(self):
        return True

    def readinto(self, buf):
        if self.n >= self.max_bytes:
            raise RuntimeError(f"byte cap of {self.max_bytes / 1e6:.0f} MB reached")
        data = self.resp.read(min(len(buf), self.max_bytes - self.n))
        self.n += len(data)
        buf[: len(data)] = data
        return len(data)


def fetch_samples(shard: str, n: int, max_mb: float) -> list[str]:
    """Stream the shard sequentially, keep the first n complete samples, then stop."""
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{shard}"
    resp = urllib.request.urlopen(url, timeout=120)
    reader = _CountingReader(resp, int(max_mb * 1e6))
    partial: dict[str, dict[str, bytes]] = {}
    done: list[str] = []
    try:
        with tarfile.open(fileobj=io.BufferedReader(reader), mode="r|") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                key, field = member.name.split(".", 1)
                field = field.removesuffix(".npy")
                partial.setdefault(key, {})[field] = tf.extractfile(member).read()
                if set(FIELDS) <= partial[key].keys():
                    _save_sample(key, partial.pop(key))
                    done.append(key)
                    if len(done) >= n:
                        break
    finally:
        resp.close()
    print(f"streamed {reader.n / 1e6:.1f} MB, saved {len(done)} complete samples to {OUT_DIR.relative_to(REPO_ROOT)}")
    return done


def _save_sample(key: str, files: dict[str, bytes]) -> None:
    d = OUT_DIR / key
    d.mkdir(parents=True, exist_ok=True)
    for field, data in files.items():
        (d / (field if field == "info.json" else f"{field}.npy")).write_bytes(data)


# --------------------------------------------------------------------------- load / model input


def load_sample(d: Path) -> dict:
    """Load one saved sample. Arrays are squeezed from (1, 224, 224) to (224, 224)."""
    s = {f: np.load(d / f"{f}.npy").squeeze(0) for f in FIELDS if f != "info.json"}
    s["info"] = json.loads((d / "info.json").read_text())
    return s


def build_model_input(sample: dict) -> np.ndarray:
    """6-channel input exactly as the official loader builds it (numpy port).

    Per channel: clamp to [0, CLAMP], NaN -> CLAMP (torch.clamp + torch.nan_to_num), then
    (x - mean) / std with the global VV/VH statistics, in MODEL_INPUT_ORDER.
    """
    channels = []
    for name in MODEL_INPUT_ORDER:
        pol = name.rsplit("_", 1)[1]
        x = np.clip(sample[name].astype("float32"), 0.0, CLAMP)
        x = np.nan_to_num(x, nan=CLAMP)
        channels.append((x - MEAN[pol]) / STD[pol])
    return np.stack(channels).astype("float32")


def split_of(actid: int) -> str:
    return "train" if actid in TRAIN_ACTS else "val" if actid in VAL_ACTS else "test" if actid in TEST_ACTS else "unknown"


# --------------------------------------------------------------------------- inspection


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
    for f in SAR_FIELDS + ("dem", "valid_mask", "mask"):
        a = s[f]
        print(f"  {f:<11}{str(a.shape):<12}{str(a.dtype):<9}{np.nanmin(a):>9.4g}{np.nanmax(a):>9.4g}"
              f"{100 * np.isnan(a).mean():>8.2f}{100 * (a == 0).mean():>8.2f}")

    values, counts = np.unique(s["mask"], return_counts=True)
    print("  label counts: " + ", ".join(
        f"{int(v)} {CLASSES.get(int(v), 'UNEXPECTED')}: {c} ({100 * c / s['mask'].size:.2f} %)"
        for v, c in zip(values, counts)))
    if (~valid).any():
        inv_vals = np.unique(s["mask"][~valid]).astype(int).tolist()
        print(f"  label values on invalid pixels: {inv_vals}")

    x = build_model_input(s)
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
        size = shard_size(args.shard)
        print(f"{HF_REPO}/{args.shard}: full shard {size / 1e9:.2f} GB; streaming only the first "
              f"{args.n} samples (≈{args.n * 1.8:.0f} MB, hard cap {args.max_mb:.0f} MB)")
        fetch_samples(args.shard, args.n, args.max_mb)

    dirs = sorted(p for p in OUT_DIR.glob("*") if p.is_dir()) if OUT_DIR.exists() else []
    if not dirs:
        print("no samples found")
        return 1
    results = [inspect(d) for d in dirs]
    (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
