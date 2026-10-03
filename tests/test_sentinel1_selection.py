"""Scene filtering/selection logic of scripts/test_sentinel1.py on synthetic metadata."""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import test_sentinel1 as s1  # noqa: E402


def scene(dt, orbit_state="ascending", relorb=85, pols=("VV", "VH"), mode="IW", cov=1.0, sid=None):
    return {
        "id": sid or f"S1_{dt}_{relorb}",
        "datetime": dt,
        "mode": mode,
        "polarizations": list(pols),
        "orbit_state": orbit_state,
        "relative_orbit": relorb,
        "aoi_coverage": cov,
    }


def test_filter_keeps_only_matching_track_mode_pols_and_coverage():
    scenes = [
        scene("2026-08-16T12:21:41Z"),
        scene("2026-08-24T00:18:44Z", orbit_state="descending", relorb=19),
        scene("2026-08-19T12:21:41Z", relorb=12),
        scene("2026-08-20T12:21:41Z", pols=("VV",)),
        scene("2026-08-21T12:21:41Z", mode="EW"),
        scene("2026-08-22T12:21:41Z", cov=0.3),
    ]
    kept = s1.filter_scenes(scenes)
    assert [s["datetime"] for s in kept] == ["2026-08-16T12:21:41Z"]


def test_filter_deduplicates_same_acquisition_time():
    scenes = [scene("2026-08-16T12:21:41Z", sid="a"), scene("2026-08-16T12:21:41Z", sid="b")]
    assert len(s1.filter_scenes(scenes)) == 1


def test_select_first_post_and_two_latest_pre():
    scenes = [scene(d) for d in (
        "2026-07-23T12:21:40Z", "2026-08-04T12:21:40Z", "2026-08-16T12:21:41Z",
        "2026-08-28T12:21:41Z", "2026-09-09T12:21:41Z",
    )]
    sel = s1.select_scenes(scenes, date(2026, 8, 26))
    assert sel["post"]["datetime"].startswith("2026-08-28")
    assert [s["datetime"][:10] for s in sel["pre"]] == ["2026-08-04", "2026-08-16"]


def test_event_day_acquisition_is_skipped_as_ambiguous():
    scenes = [scene(d) for d in ("2026-08-14T12:00:00Z", "2026-08-26T12:00:00Z", "2026-09-07T12:00:00Z")]
    sel = s1.select_scenes(scenes, date(2026, 8, 26))
    assert sel["post"]["datetime"].startswith("2026-09-07")
    assert [s["datetime"][:10] for s in sel["skipped_event_day"]] == ["2026-08-26"]


def test_select_without_post_scene():
    sel = s1.select_scenes([scene("2026-08-16T12:21:41Z")], date(2026, 8, 26))
    assert sel["post"] is None


def test_bbox_coverage_full_and_partial():
    square = {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}
    assert s1.bbox_coverage(square, (1, 1, 9, 9)) == 1.0
    assert 0.4 < s1.bbox_coverage(square, (5, 1, 15, 9)) < 0.6


def test_raster_stats_on_synthetic_geotiff(tmp_path):
    """Exercises the statistics routine on a clearly synthetic 3-band raster (not real data)."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(0)
    vv = rng.uniform(0.01, 0.2, (20, 20)).astype("float32")
    vh = (vv / 4).astype("float32")
    data_mask = np.ones((20, 20), "float32")
    data_mask[:, :2] = 0  # 10 % no-data
    path = tmp_path / "synthetic.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=20, height=20, count=3, dtype="float32",
        crs="EPSG:32645", transform=from_origin(300000, 3100000, 10, 10),
    ) as ds:
        ds.write(np.stack([vv, vh, data_mask]))
        for i, name in enumerate(s1.BAND_NAMES, start=1):
            ds.set_band_description(i, name)

    info = s1.raster_stats(path, histogram=False)
    assert info["shape"] == [3, 20, 20]
    assert info["resolution"] == [10.0, 10.0]
    assert info["nodata_pct"] == 10.0
    expected = vv[:, 2:]
    assert abs(info["VV"]["mean"] - float(expected.mean())) < 1e-6
    assert 0 < info["VV"]["pct_above_kuro_clamp"] < 100


def test_snap_bounds_expands_to_10_unit_grid():
    assert s1.snap_bounds((9481231.4, 3242001.0, 9486799.9, 3247050.0)) == (9481230, 3242000, 9486800, 3247050)
