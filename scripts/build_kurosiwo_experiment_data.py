"""Build the event-separated Kuro Siwo train/validation/test subsets for VantageQ experiments.

Uses shard probes (event ID at regular byte offsets of every labelled HF shard; see
ml/kurosiwo.py::probe_shard) to stream a few samples per selected event, from up to
--starts-per-event separate positions inside the event for spatial diversity.

Train, validation and test events come only from the official TRAIN / VAL / TEST splits
respectively; the script refuses anything else. Samples are de-duplicated by grid_id, because the
HF train_GRD/test_GRD folders overlap in events.

Output (git-ignored): data/external/kurosiwo_experiment/{train,val,test}/<actid>_<grid_id>/
+ manifest_<splits>.json

Usage:
  python scripts/build_kurosiwo_experiment_data.py --dry-run
  python scripts/build_kurosiwo_experiment_data.py                  # train + val
  python scripts/build_kurosiwo_experiment_data.py --splits test    # held-out test subset
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.kurosiwo import (  # noqa: E402
    SAMPLE_BYTES, TEST_SHARDS, TRAIN_SHARDS, probe_shard, split_of, stream_samples,
)

OUT_DIR = REPO_ROOT / "data" / "external" / "kurosiwo_experiment"

# Chosen for diversity across continents and climate zones (docs/TRAINING_EXPERIMENT.md).
TRAIN_EVENTS = (118, 130, 147, 273, 275, 324, 427, 470, 502, 555, 1111004, 1111005, 1111009, 1111011)
# All official validation events located by probing (514, 520, 559 were not found at 0.5 GB spacing).
VAL_EVENTS = (279, 437, 1111003, 1111008)
# Official test events located by probing (205, 411, 561, 1111002 were not found at 0.5 GB spacing).
TEST_EVENTS = (277, 321, 445, 562, 1111007, 1111013)
EVENTS = {"train": TRAIN_EVENTS, "val": VAL_EVENTS, "test": TEST_EVENTS}


def load_or_probe(step_gb: float) -> list[dict]:
    probe_file = OUT_DIR / "probes.json"
    if probe_file.exists():
        return json.loads(probe_file.read_text())
    probes, nbytes, t0 = [], 0, time.time()
    for shard in TRAIN_SHARDS + TEST_SHARDS:
        p, n = probe_shard(shard, step_gb * 1e9, workers=8)
        probes += p
        nbytes += n
    print(f"probing: {len(probes)} probes, {nbytes / 1e6:.1f} MB, {time.time() - t0:.0f} s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probe_file.write_text(json.dumps(probes, indent=1))
    return probes


def starts_for(probes: list[dict], actid: int, k: int) -> list[dict]:
    """Up to k probe positions spread over the event's probes (first, then evenly spaced)."""
    ps = [p for p in probes if p["actid"] == actid]
    if len(ps) <= k:
        return ps
    return [ps[round(i * (len(ps) - 1) / (k - 1))] for i in range(k)] if k > 1 else ps[:1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-per-event", type=int, default=18)
    ap.add_argument("--val-per-event", type=int, default=20)
    ap.add_argument("--test-per-event", type=int, default=15)
    ap.add_argument("--splits", default="train,val", help="comma-separated: train,val,test")
    ap.add_argument("--starts-per-event", type=int, default=2)
    ap.add_argument("--step-gb", type=float, default=0.5, help="probe spacing (only if no cached probes)")
    ap.add_argument("--max-mb", type=float, default=900, help="hard cap on total streamed MB")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    splits = [x.strip() for x in args.splits.split(",")]
    if not set(splits) <= set(EVENTS):
        print(f"unknown split in {splits}")
        return 1
    for split, events in EVENTS.items():
        wrong = [a for a in events if split_of(a) != split]
        if wrong:
            print(f"refusing: events {wrong} are not in the official {split} split")
            return 1

    probes = load_or_probe(args.step_gb)
    plan = []
    per_event_n = {"train": args.train_per_event, "val": args.val_per_event, "test": args.test_per_event}
    for split in splits:
        events, per_event = EVENTS[split], per_event_n[split]
        for actid in events:
            starts = starts_for(probes, actid, args.starts_per_event)
            if not starts:
                print(f"  {split} event {actid}: not found in probes, skipped")
                continue
            per_start = -(-per_event // len(starts))  # ceil
            plan += [(split, actid, s, per_start) for s in starts]
    est = sum(n for *_, n in plan) * SAMPLE_BYTES / 1e6
    for split in splits:
        ev = sorted({a for s, a, *_ in plan if s == split})
        n = sum(k for s, _, _, k in plan if s == split)
        print(f"{split}: {len(ev)} events {ev}, up to {n} samples")
    print(f"estimate ≈ {est:.0f} MB streamed (cap {args.max_mb:.0f} MB)")
    if args.dry_run:
        return 0
    if est > args.max_mb:
        print("estimate exceeds --max-mb; stopping")
        return 1

    total, seen, manifest = 0, set(), []
    for split, actid, start, per_start in plan:
        tmp = OUT_DIR / "_incoming"
        budget = min(args.max_mb * 1e6 - total, (per_start + 3) * SAMPLE_BYTES)
        saved, nbytes = stream_samples(start["shard"], tmp, per_start, start=start["offset"],
                                       max_bytes=budget, only_actid=actid)
        total += nbytes
        kept = 0
        for d in saved:
            grid = json.loads((d / "info.json").read_text())["grid_id"]
            dest = OUT_DIR / split / f"{actid}_{grid}"
            if grid in seen or dest.exists():
                shutil.rmtree(d)
                continue
            seen.add(grid)
            dest.parent.mkdir(parents=True, exist_ok=True)
            d.rename(dest)
            manifest.append({"split": split, "actid": actid, "grid_id": grid, "shard": start["shard"],
                             "offset": start["offset"]})
            kept += 1
        print(f"  {split} {actid} @ {start['shard']}:{start['offset']}: kept {kept}, {nbytes / 1e6:.1f} MB")
    shutil.rmtree(OUT_DIR / "_incoming", ignore_errors=True)
    name = "manifest.json" if splits == ["train", "val"] else f"manifest_{'_'.join(splits)}.json"
    (OUT_DIR / name).write_text(json.dumps(manifest, indent=1))
    for split in splits:
        rows = [m for m in manifest if m["split"] == split]
        print(f"{split}: {len(rows)} samples from {len({m['actid'] for m in rows})} events")
    print(f"streamed {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
