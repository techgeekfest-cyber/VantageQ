"""Trishuli inference experiment with the FROZEN Kuro Siwo U-Net. No labels, no accuracy.

1. Find the Track 85 ascending scenes (CDSE STAC): post 2026-08-28, pre1 2026-08-16, pre2 2026-08-04.
2. Fetch each as orthorectified linear sigma0 (SIGMA0_ELLIPSOID, COPERNICUS_30, Lee 7x7) on ONE
   shared EPSG:3857 10-unit grid (cached under data/interim/trishuli/).
3. Check all three rasters share CRS, transform, size and bounds.
4. Build the 6-channel input with ml/preprocessing.py, run tiled inference with the frozen
   checkpoint, write class / probability GeoTIFFs and quicklooks to data/processed/trishuli/.

Uses only permitted inputs (Sentinel-1). Not validated against any reference. See
docs/TRISHULI_INFERENCE.md.

Usage:
  python scripts/infer_trishuli.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from ml.inference import TILE, load_frozen_model, predict_tiled, sar_inputs_from_dates  # noqa: E402
from ml.kurosiwo import LABELS  # noqa: E402
from ml.preprocessing import CLAMP, build_model_input  # noqa: E402
from vantageq.satellite import sentinel1 as s1  # noqa: E402

AOI_BBOX = (85.10, 27.85, 85.42, 28.30)  # Trishuli valley box used for scene coverage
SUB_AOI_BBOX = (85.17, 27.95, 85.22, 27.99)  # inference sub-AOI (WGS84 w, s, e, n)
EVENT_DATE = date(2026, 8, 26)
RELATIVE_ORBIT, ORBIT_STATE = 85, "ascending"
EXPECTED_DATES = {"post": "2026-08-28", "pre1": "2026-08-16", "pre2": "2026-08-04"}

CHECKPOINT = REPO_ROOT / "ml" / "checkpoints" / "kurosiwo_unet_r18_best.pt"
IN_DIR = REPO_ROOT / "data" / "interim" / "trishuli"
OUT_DIR = REPO_ROOT / "data" / "processed" / "trishuli"
CLASS_NODATA = 255


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def rel(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT))


def select_track85() -> dict[str, dict]:
    start = datetime.combine(EVENT_DATE - timedelta(days=36), datetime.min.time(), timezone.utc)
    end = datetime.combine(EVENT_DATE + timedelta(days=14), datetime.min.time(), timezone.utc)
    scenes = [s1.scene_summary(i, AOI_BBOX, SUB_AOI_BBOX) for i in s1.stac_search(AOI_BBOX, start, end)]
    sel = s1.select_scenes(s1.filter_scenes(scenes, relative_orbit=RELATIVE_ORBIT, orbit_state=ORBIT_STATE),
                           EVENT_DATE, n_pre=2)
    if sel["post"] is None or len(sel["pre"]) < 2:
        raise SystemExit("no complete Track 85 post/pre1/pre2 set found")
    return {"post": sel["post"], "pre1": sel["pre"][-1], "pre2": sel["pre"][-2]}


def raw_stats(vv: np.ndarray, vh: np.ndarray, valid: np.ndarray) -> dict:
    out = {"nodata_pct": round(100 * float((~valid).mean()), 3)}
    for name, a in (("VV", vv), ("VH", vh)):
        v = a[valid]
        p = np.percentile(v, [1, 50, 99])
        out[name] = {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()), "p1": float(p[0]),
                     "median": float(p[1]), "p99": float(p[2]), "pct_above_clamp": round(100 * float((v > CLAMP).mean()), 2)}
    return out


def write_tif(path: Path, data: np.ndarray, profile: dict, dtype: str, nodata, description: str) -> None:
    prof = profile | {"count": 1, "dtype": dtype, "nodata": nodata, "compress": "deflate"}
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(data.astype(dtype), 1)
        ds.set_band_description(1, description)


def quicklooks(bands: dict, cls: np.ndarray, probs: np.ndarray, valid: np.ndarray, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    def db(a):
        return 10 * np.log10(np.clip(a, 1e-4, None))

    def masked(a):
        return np.where(valid, a, np.nan)

    flood_p, water_p = masked(probs[2]), masked(probs[1] + probs[2])
    paths = []

    fig, ax = plt.subplots(1, 5, figsize=(22, 4.8))
    ax[0].imshow(db(bands["pre1"]["VV"]), cmap="gray", vmin=-20, vmax=5)
    ax[0].set_title("pre1 VV dB (2026-08-16)")
    ax[1].imshow(db(bands["post"]["VV"]), cmap="gray", vmin=-20, vmax=5)
    ax[1].set_title("post VV dB (2026-08-28)")
    ax[2].imshow(db(bands["post"]["VV"]), cmap="gray", vmin=-20, vmax=5)
    overlay = np.ma.masked_where(~np.isin(cls, [1, 2]), cls)
    ax[2].imshow(overlay, cmap=ListedColormap(["#2c7fb8", "#d7301f"]), vmin=1, vmax=2, interpolation="nearest")
    ax[2].set_title("predicted: blue permanent water, red flood")
    im = ax[3].imshow(flood_p, cmap="magma", vmin=0, vmax=1)
    ax[3].set_title("flood probability")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    im = ax[4].imshow(water_p, cmap="viridis", vmin=0, vmax=1)
    ax[4].set_title("water probability (perm. + flood)")
    fig.colorbar(im, ax=ax[4], fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.suptitle("Trishuli sub-AOI 85.17–85.22 E, 27.95–27.99 N · frozen Kuro Siwo U-Net · "
                 "inference experiment, NOT validated", fontsize=10)
    fig.tight_layout()
    paths.append(out_dir / "quicklook_prediction.png")
    fig.savefig(paths[-1], dpi=90)
    plt.close(fig)

    pre_mean_vv = (bands["pre1"]["VV"] + bands["pre2"]["VV"]) / 2
    ratio_db = masked(db(bands["post"]["VV"]) - db(pre_mean_vv))
    pre_change = masked(db(bands["pre1"]["VV"]) - db(bands["pre2"]["VV"]))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    im = ax[0].imshow(ratio_db, cmap="RdBu", vmin=-6, vmax=6)
    ax[0].set_title("post − mean(pre1, pre2) VV [dB]\n(red = darker after: possible new water)")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    im = ax[1].imshow(pre_change, cmap="RdBu", vmin=-6, vmax=6)
    ax[1].set_title("pre1 − pre2 VV [dB]\n(reference: change with no event)")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].imshow(np.where(valid, cls, 3), cmap=ListedColormap(["#f2f2f2", "#2c7fb8", "#d7301f", "#9e9e9e"]),
                 vmin=0, vmax=3, interpolation="nearest")
    ax[2].set_title("predicted class (white none, blue perm., red flood)")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    paths.append(out_dir / "quicklook_change.png")
    fig.savefig(paths[-1], dpi=90)
    plt.close(fig)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--overlap", type=int, default=64, help="tile overlap in pixels (tile = 224)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    # 1. scenes
    scenes = select_track85()
    for role, sc in scenes.items():
        print(f"{role:<5} {sc['datetime'][:19]}  {sc['orbit_state']} relorb {sc['relative_orbit']}  {sc['id']}")
        if sc["datetime"][:10] != EXPECTED_DATES[role]:
            print(f"  unexpected {role} date (expected {EXPECTED_DATES[role]}); stopping")
            return 1

    # 2. shared grid + fetch (cached)
    grid = s1.make_grid(SUB_AOI_BBOX, s1.KURO_SIWO_EPSG, s1.RESOLUTION, snap=True)
    print(f"grid: EPSG:{grid.epsg} bounds {grid.bounds} size {grid.width}x{grid.height} res {grid.res}")
    paths = {role: IN_DIR / f"s1_{role}_{sc['datetime'][:10].replace('-', '')}_relorb85_epsg3857.tif"
             for role, sc in scenes.items()}
    missing = [r for r, p in paths.items() if not p.exists()]
    if missing:
        s1.load_dotenv(REPO_ROOT / ".env")
        cid, secret = os.environ.get("SH_CLIENT_ID"), os.environ.get("SH_CLIENT_SECRET")
        if not (cid and secret):
            print("SH_CLIENT_ID / SH_CLIENT_SECRET not set (.env); cannot fetch", missing)
            return 2
        token = s1.get_token(cid, secret)
        for role in missing:
            s1.fetch_sigma0(scenes[role], grid, token, paths[role])
            print(f"  fetched {role} -> {rel(paths[role])}")
    else:
        print("  using cached rasters in", rel(IN_DIR))

    # 3. identical grid
    problems = s1.check_same_grid(list(paths.values()), expected=grid)
    if problems:
        print("grid mismatch:", *problems, sep="\n  ")
        return 3
    print("grid check: all three rasters share CRS, transform, size and bounds")

    bands, valid = {}, None
    for role, p in paths.items():
        with rasterio.open(p) as ds:
            vv, vh, dm = ds.read().astype("float32")
            profile = ds.profile
        bands[role] = {"VV": vv, "VH": vh}
        v = (dm == 1) & np.isfinite(vv) & np.isfinite(vh)
        valid = v if valid is None else valid & v
    stats = {}
    for role in paths:
        st = raw_stats(bands[role]["VV"], bands[role]["VH"], valid)
        with rasterio.open(paths[role]) as ds:
            st["nodata_pct_own_datamask"] = round(100 * float((ds.read(3) != 1).mean()), 3)
        stats[role] = st
        print(f"  {role:<5} VV mean {st['VV']['mean']:.4f} median {st['VV']['median']:.4f} p99 {st['VV']['p99']:.3f} "
              f">0.15 {st['VV']['pct_above_clamp']}% | VH mean {st['VH']['mean']:.4f} median {st['VH']['median']:.4f} "
              f">0.15 {st['VH']['pct_above_clamp']}% | own no-data {st['nodata_pct_own_datamask']}%")

    # 4. model input + inference
    x = build_model_input(sar_inputs_from_dates(bands["post"], bands["pre1"], bands["pre2"]))
    print(f"model input: {x.shape} {x.dtype}  (all-date valid pixels {100 * valid.mean():.2f}%)")
    device = pick_device(args.device)
    model, ckpt = load_frozen_model(args.checkpoint, device)
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()[:16]
    probs = predict_tiled(model, x, device, tile=TILE, overlap=args.overlap)
    # Seam/tiling diagnostic: a second pass with a different tiling should give near-identical output.
    probs_alt = predict_tiled(model, x, device, tile=TILE, overlap=112, margin=80)
    seam_diff = float(np.abs(probs[2] - probs_alt[2])[valid].mean())

    cls = probs.argmax(0).astype("uint8")
    cls_out = np.where(valid, cls, CLASS_NODATA)
    n_valid = int(valid.sum())
    pct = {LABELS[c]: round(100 * float((cls[valid] == c).sum()) / n_valid, 3) for c in LABELS}
    fp = probs[2][valid]
    flood_px = int((cls[valid] == 2).sum())
    lat = (SUB_AOI_BBOX[1] + SUB_AOI_BBOX[3]) / 2
    px_area_m2 = (grid.res[0] * math.cos(math.radians(lat))) * (grid.res[1] * math.cos(math.radians(lat)))
    flood_summary = {
        "mean": float(fp.mean()), "median": float(np.median(fp)), "p90": float(np.percentile(fp, 90)),
        "p99": float(np.percentile(fp, 99)), "max": float(fp.max()),
        "pct_gt_0.5": round(100 * float((fp > 0.5).mean()), 3), "pct_gt_0.9": round(100 * float((fp > 0.9).mean()), 3),
        "flood_pixels": flood_px, "flood_pct": pct["flood"],
        "flood_area_km2_approx": round(flood_px * px_area_m2 / 1e6, 3),
    }
    print(f"device {device}; checkpoint {rel(args.checkpoint)} sha256 {sha}… epoch {ckpt['epoch']}")
    print("predicted class % of valid pixels:", pct)
    print("flood probability:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in flood_summary.items()})
    print(f"tiling diagnostic: mean |Δ flood prob| between overlap 64 and 112 tilings = {seam_diff:.5f}")

    # 5. outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nan = np.float32("nan")
    outputs = {
        "class_map": OUT_DIR / "trishuli_20260828_class.tif",
        "prob_flood": OUT_DIR / "trishuli_20260828_prob_flood.tif",
        "prob_permanent_water": OUT_DIR / "trishuli_20260828_prob_permanent_water.tif",
        "prob_water": OUT_DIR / "trishuli_20260828_prob_water.tif",
    }
    write_tif(outputs["class_map"], cls_out, profile, "uint8", CLASS_NODATA,
              "0 no water, 1 permanent water, 2 flood, 255 no data")
    write_tif(outputs["prob_flood"], np.where(valid, probs[2], nan), profile, "float32", nan, "P(flood)")
    write_tif(outputs["prob_permanent_water"], np.where(valid, probs[1], nan), profile, "float32", nan,
              "P(permanent water)")
    write_tif(outputs["prob_water"], np.where(valid, probs[1] + probs[2], nan), profile, "float32", nan,
              "P(permanent water) + P(flood)")
    problems = s1.check_same_grid(list(outputs.values()), expected=grid)
    print("output grid check:", "OK" if not problems else problems)
    looks = quicklooks(bands, cls, probs, valid, OUT_DIR)

    summary = {
        "note": "Inference experiment with a frozen Kuro Siwo model; not validated against any reference.",
        "scenes": {r: {k: sc[k] for k in ("id", "datetime", "platform", "orbit_state", "relative_orbit",
                                          "absolute_orbit")} for r, sc in scenes.items()},
        "sub_aoi_wgs84": SUB_AOI_BBOX, "grid": {"epsg": grid.epsg, "bounds": grid.bounds, "width": grid.width,
                                                "height": grid.height, "res": grid.res,
                                                "transform": list(grid.transform)[:6]},
        "inputs": {r: rel(p) for r, p in paths.items()}, "raw_stats": stats,
        "valid_pixels": n_valid, "valid_pct": round(100 * float(valid.mean()), 3),
        "model_input_shape": list(x.shape), "checkpoint": rel(args.checkpoint), "checkpoint_sha256_16": sha,
        "checkpoint_epoch": ckpt["epoch"], "device": str(device), "tile": TILE, "overlap": args.overlap,
        "class_pct": pct, "flood_probability": flood_summary, "tiling_diagnostic_mean_abs_diff": seam_diff,
        "outputs": {k: rel(v) for k, v in outputs.items()}, "quicklooks": [rel(p) for p in looks],
    }
    (OUT_DIR / "trishuli_20260828_summary.json").write_text(json.dumps(summary, indent=1))
    print("outputs:", *[rel(p) for p in outputs.values()], sep="\n  ")
    print("quicklooks:", *[rel(p) for p in looks], sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
