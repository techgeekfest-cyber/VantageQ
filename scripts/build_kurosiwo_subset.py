"""Build a tiny multi-event Kuro Siwo training subset by streaming, without downloading shards.

1. Probe each labelled train_GRD shard every --step-gb: a ~200 KB range read finds a tar header,
   then 512-byte header hops reach the nearest info.json, giving the event ID at that offset.
2. Keep events in the OFFICIAL train split (the HF shards also contain val/test events).
3. For up to --max-events events, stream --per-event complete samples from that event's first
   probed offset, then stop reading.

Output: data/external/kurosiwo_subset/samples/<shard>_<key>/ (git-ignored) + subset.json.

Usage:
  python scripts/build_kurosiwo_subset.py --dry-run         # probe + estimate only
  python scripts/build_kurosiwo_subset.py                   # 8 events x 8 samples (~116 MB)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.kurosiwo import SAMPLE_BYTES, TRAIN_SHARDS, probe_shard, split_of, stream_samples  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "external" / "kurosiwo_subset"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-events", type=int, default=8)
    ap.add_argument("--per-event", type=int, default=8)
    ap.add_argument("--step-gb", type=float, default=1.0, help="probe spacing within each shard")
    ap.add_argument("--max-mb", type=float, default=200, help="hard cap on total streamed MB")
    ap.add_argument("--dry-run", action="store_true", help="probe and estimate only")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probe_file = OUT_DIR / "probes.json"
    if probe_file.exists():
        probes = json.loads(probe_file.read_text())
        print(f"using cached probes ({len(probes)}) from {probe_file.relative_to(REPO_ROOT)}")
    else:
        probes, probe_bytes, t0 = [], 0, time.time()
        for shard in TRAIN_SHARDS:
            p, nbytes = probe_shard(shard, args.step_gb * 1e9)
            probes += p
            probe_bytes += nbytes
            print(f"  {shard}: events {[x['actid'] for x in p]}")
        print(f"probing fetched {probe_bytes / 1e6:.1f} MB in {time.time() - t0:.0f} s")
        probe_file.write_text(json.dumps(probes, indent=2))

    first = {}
    for p in probes:
        first.setdefault(p["actid"], p)
    by_split = Counter(split_of(a) for a in first)
    print(f"events found: {len(first)} ({dict(by_split)})")
    chosen = [p for a, p in first.items() if split_of(a) == "train"][: args.max_events]
    est_mb = len(chosen) * args.per_event * SAMPLE_BYTES / 1e6
    print(f"selected train events: {[p['actid'] for p in chosen]}")
    print(f"estimate: {len(chosen)} x {args.per_event} samples ≈ {est_mb:.0f} MB to stream (cap {args.max_mb:.0f} MB)")
    if args.dry_run:
        return 0
    if est_mb > args.max_mb:
        print("estimate exceeds --max-mb; stopping")
        return 1

    samples_dir, total, records = OUT_DIR / "samples", 0, []
    for p in chosen:
        budget = min(args.max_mb * 1e6 - total, (args.per_event + 2) * SAMPLE_BYTES)
        prefix = f"{Path(p['shard']).stem.split('-')[-1]}_"
        saved, nbytes = stream_samples(p["shard"], samples_dir, args.per_event, start=p["offset"],
                                       max_bytes=budget, prefix=prefix, only_actid=p["actid"])
        total += nbytes
        records += [{"dir": d.name, "actid": p["actid"], "shard": p["shard"]} for d in saved]
        print(f"  event {p['actid']}: {len(saved)} samples, {nbytes / 1e6:.1f} MB")
    (OUT_DIR / "subset.json").write_text(json.dumps(records, indent=2))
    print(f"streamed {total / 1e6:.1f} MB total; {len(records)} samples from "
          f"{len({r['actid'] for r in records})} events -> {samples_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
