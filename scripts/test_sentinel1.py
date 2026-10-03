"""Milestone 1 — Sentinel-1 smoke test for the Trishuli case study.

Stages:
  1. Search the CDSE STAC catalogue (public, no credentials) for Sentinel-1 GRD scenes.
  2. Keep IW / VV+VH / ascending / relative orbit 85 scenes covering the AOI and pick the first
     post-event scene plus the two latest pre-event scenes.
  3. If Sentinel Hub credentials are configured, fetch a small sub-AOI of the post-event scene as
     orthorectified linear sigma0 (VV, VH) via the Sentinel Hub Process API on CDSE.
  4. Print raster statistics and compare them with Kuro Siwo's documented input statistics.

Search, selection, grid and retrieval live in backend/vantageq/satellite/sentinel1.py.
See docs/SENTINEL1_SMOKE_TEST.md.

Usage:
  python scripts/test_sentinel1.py                 # search, then retrieve + stats if credentials
  python scripts/test_sentinel1.py --metadata-only # search only
  python scripts/test_sentinel1.py --stats FILE    # stats for an already-downloaded GeoTIFF
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from vantageq.satellite import sentinel1 as s1  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "interim" / "smoke_s1"

# Approximate Trishuli valley box from reconnaissance (docs/DATA_SOURCES.md §3): w, s, e, n.
AOI_BBOX = (85.10, 27.85, 85.42, 28.30)
# Small sub-AOI around the Trishuli river near Betrawati (~5 km x 4.4 km).
SUB_AOI_BBOX = (85.17, 27.95, 85.22, 27.99)
EVENT_DATE = date(2026, 8, 26)

# Selection criteria (docs/ARCHITECTURE.md §3).
INSTRUMENT_MODE = "IW"
POLARIZATIONS = {"VV", "VH"}
ORBIT_STATE = "ascending"
RELATIVE_ORBIT = 85
SEARCH_DAYS_BEFORE = 36
SEARCH_DAYS_AFTER = 14
N_PRE = 2

OUTPUT_CRS_EPSG = 32645  # default: UTM 45N, covers Trishuli
KURO_SIWO_CRS_EPSG = s1.KURO_SIWO_EPSG  # Kuro Siwo grid: Web Mercator, 10 map units, origin (0, 0)
RESOLUTION_M = s1.RESOLUTION

# Kuro Siwo GRD input statistics (configs/train/data_config.json), linear sigma0.
KURO_SIWO = {"VV": {"mean": 0.0953, "std": 0.0427}, "VH": {"mean": 0.0264, "std": 0.0215}}
KURO_SIWO_CLAMP = 0.15

# Shared implementation: backend/vantageq/satellite/sentinel1.py. Thin wrappers bind this
# smoke test's AOI and selection criteria.
BAND_NAMES = s1.BAND_NAMES
stac_search = s1.stac_search
bbox_coverage = s1.bbox_coverage
snap_bounds = s1.snap_bounds
get_token = s1.get_token


def scene_summary(item: dict) -> dict:
    return s1.scene_summary(item, AOI_BBOX, SUB_AOI_BBOX)


def filter_scenes(scenes: list[dict], min_coverage: float = 0.99) -> list[dict]:
    return s1.filter_scenes(scenes, relative_orbit=RELATIVE_ORBIT, orbit_state=ORBIT_STATE,
                            mode=INSTRUMENT_MODE, polarizations=POLARIZATIONS, min_coverage=min_coverage)


def select_scenes(scenes: list[dict], event: date, n_pre: int = N_PRE) -> dict:
    return s1.select_scenes(scenes, event, n_pre)


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    s1.load_dotenv(path)


def retrieve_sigma0(scene: dict, token: str, epsg: int = OUTPUT_CRS_EPSG) -> Path:
    # EPSG:3857 is snapped to the Kuro Siwo (0, 0)-origin 10-unit grid; UTM keeps the plain box.
    grid = s1.make_grid(SUB_AOI_BBOX, epsg, RESOLUTION_M, snap=(epsg == KURO_SIWO_CRS_EPSG))
    acq = scene["datetime"][:10].replace("-", "")
    suffix = "" if epsg == OUTPUT_CRS_EPSG else f"_epsg{epsg}"
    out = OUT_DIR / f"s1_sigma0_{acq}_relorb{scene['relative_orbit']}_subaoi{suffix}.tif"
    return s1.fetch_sigma0(scene, grid, token, out)


# --------------------------------------------------------------------------- statistics


def _rel(path: Path) -> str:
    """Path relative to the repo root when inside it, else absolute (for printing)."""
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def raster_stats(path: Path, histogram: bool = True) -> dict:
    """Print and return basic statistics for VV/VH bands of the smoke-test GeoTIFF."""
    import numpy as np
    import rasterio

    with rasterio.open(path) as ds:
        arr = ds.read().astype("float64")
        names = [d or f"band{i}" for i, d in enumerate(ds.descriptions, start=1)]
        info = {
            "file": _rel(path),
            "shape": list(arr.shape),
            "crs": str(ds.crs),
            "resolution": list(ds.res),
            "bounds": list(ds.bounds),
        }

    bands = dict(zip(names, arr))
    data_mask = bands.get("dataMask", np.ones(arr.shape[1:]))
    valid = (data_mask == 1) & np.isfinite(bands["VV"]) & np.isfinite(bands["VH"]) & (bands["VV"] > 0)
    info["nodata_pct"] = round(100 * (1 - valid.mean()), 2)

    print(f"\nRaster: {info['file']}")
    print(f"  shape {info['shape']}  CRS {info['crs']}  resolution {info['resolution']} m")
    print(f"  no-data: {info['nodata_pct']} %")

    pcts = [1, 5, 25, 50, 75, 95, 99]
    for name in ("VV", "VH"):
        v = bands[name][valid]
        if v.size == 0:
            print(f"  {name}: no valid pixels")
            continue
        s = {
            "min": float(v.min()),
            "max": float(v.max()),
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "std": float(v.std()),
            "percentiles": {f"p{p}": float(x) for p, x in zip(pcts, np.percentile(v, pcts))},
            "pct_above_kuro_clamp": round(100 * float((v > KURO_SIWO_CLAMP).mean()), 2),
            "mean_clamped": float(np.clip(v, 0, KURO_SIWO_CLAMP).mean()),
            "median_db": float(10 * np.log10(np.median(v))),
        }
        info[name] = s
        k = KURO_SIWO[name]
        print(
            f"  {name}: min {s['min']:.4f}  max {s['max']:.4f}  mean {s['mean']:.4f}  median {s['median']:.4f}"
            f"  std {s['std']:.4f}  (median {s['median_db']:.1f} dB)"
        )
        print("       " + "  ".join(f"{p} {x:.4f}" for p, x in s["percentiles"].items()))
        print(
            f"       > {KURO_SIWO_CLAMP} (Kuro Siwo clamp): {s['pct_above_kuro_clamp']} %   "
            f"mean after clamp {s['mean_clamped']:.4f}   Kuro Siwo mean/std {k['mean']}/{k['std']}"
        )

    stats_path = path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(info, indent=2))
    print(f"  stats written to {_rel(stats_path)}")

    if histogram:
        _histogram(bands, valid, path.with_suffix(".hist.png"))
    return info


def _histogram(bands: dict, valid, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    bins = np.logspace(-4, 0.5, 80)
    for ax, name in zip(axes, ("VV", "VH")):
        ax.hist(bands[name][valid], bins=bins, color="0.4")
        ax.axvline(KURO_SIWO[name]["mean"], color="tab:blue", label="Kuro Siwo mean")
        ax.axvline(KURO_SIWO_CLAMP, color="tab:red", linestyle="--", label="Kuro Siwo clamp 0.15")
        ax.set_xscale("log")
        ax.set_title(f"{name} linear sigma0 (sub-AOI)")
        ax.set_xlabel("sigma0 (linear)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  histogram written to {_rel(out)}")


# --------------------------------------------------------------------------- main


def _error_text(err: urllib.error.HTTPError) -> str:
    """Server error message, truncated. Never includes request credentials."""
    return err.read().decode(errors="replace")[:500]


def _print_scene(label: str, s: dict) -> None:
    print(
        f"  {label:<5} {s['datetime'][:19]}  {s['platform']}  {s['orbit_state']}  relorb {s['relative_orbit']}"
        f"  abs {s['absolute_orbit']}  AOI {s['aoi_coverage']:.0%}  sub-AOI {s['sub_aoi_coverage']:.0%}\n"
        f"        {s['id']}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata-only", action="store_true", help="search only, no raster retrieval")
    ap.add_argument("--stats", type=Path, help="only compute statistics for an existing GeoTIFF")
    ap.add_argument(
        "--epsg", type=int, default=OUTPUT_CRS_EPSG, choices=[OUTPUT_CRS_EPSG, KURO_SIWO_CRS_EPSG],
        help=f"output CRS: {OUTPUT_CRS_EPSG} (UTM 45N, default) or {KURO_SIWO_CRS_EPSG} (Kuro Siwo grid)",
    )
    args = ap.parse_args()

    if args.stats:
        raster_stats(args.stats.resolve())
        return 0

    start = datetime.combine(EVENT_DATE - timedelta(days=SEARCH_DAYS_BEFORE), datetime.min.time(), timezone.utc)
    end = datetime.combine(EVENT_DATE + timedelta(days=SEARCH_DAYS_AFTER), datetime.min.time(), timezone.utc)
    print(f"CDSE STAC search: {s1.STAC_COLLECTION}, bbox {AOI_BBOX}, {start:%Y-%m-%d} .. {end:%Y-%m-%d}")
    scenes = [scene_summary(i) for i in stac_search(AOI_BBOX, start, end)]
    print(f"  {len(scenes)} items returned")
    for s in sorted(scenes, key=lambda s: s["datetime"]):
        print(
            f"    {s['datetime'][:19]}  {s['platform']}  {s['mode']}  {'+'.join(s['polarizations'])}"
            f"  {s['orbit_state']:<10} relorb {s['relative_orbit']:<3}  AOI {s['aoi_coverage']:.0%}"
        )

    matched = filter_scenes(scenes)
    print(
        f"\nFiltered ({INSTRUMENT_MODE}, VV+VH, {ORBIT_STATE}, relorb {RELATIVE_ORBIT}, AOI covered): "
        f"{len(matched)} scenes"
    )
    for s in matched:
        _print_scene("", s)

    sel = select_scenes(matched, EVENT_DATE)
    print(f"\nSelection around event date {EVENT_DATE}:")
    for s in sel["pre"]:
        _print_scene("pre", s)
    if sel["post"]:
        _print_scene("post", sel["post"])
    for s in sel["skipped_event_day"]:
        _print_scene("skip", s)
    if not sel["post"] or len(sel["pre"]) < N_PRE:
        print("No complete same-track pre/post set found. Stopping.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "selection.json").write_text(json.dumps(sel, indent=2))
    print(f"  selection written to {_rel(OUT_DIR / 'selection.json')}")

    if args.metadata_only:
        return 0

    load_dotenv()
    client_id, client_secret = os.environ.get("SH_CLIENT_ID"), os.environ.get("SH_CLIENT_SECRET")
    if not (client_id and client_secret):
        print(
            "\nRaster retrieval skipped: SH_CLIENT_ID / SH_CLIENT_SECRET are not set.\n"
            "  Create a Sentinel Hub OAuth client in the CDSE dashboard and put both values in .env\n"
            "  (see .env.example and docs/SENTINEL1_SMOKE_TEST.md). Stopping at the metadata stage."
        )
        return 2

    print(f"\nRequesting sub-AOI {SUB_AOI_BBOX} of post-event scene (EPSG:{args.epsg}) via Sentinel Hub Process API ...")
    try:
        token = get_token(client_id, client_secret)
    except urllib.error.HTTPError as err:
        print(f"  authentication failed (CDSE token endpoint): HTTP {err.code}: {_error_text(err)}")
        print("  Check SH_CLIENT_ID / SH_CLIENT_SECRET in .env and the OAuth client in the CDSE dashboard.")
        return 3
    print("  authentication OK (token obtained)")
    try:
        path = retrieve_sigma0(sel["post"], token, args.epsg)
    except urllib.error.HTTPError as err:
        print(f"  Process API request failed: HTTP {err.code}: {_error_text(err)}")
        return 4
    print(f"  saved {_rel(path)}")
    raster_stats(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
