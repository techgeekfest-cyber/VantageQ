"""Shared grid, identical-grid validation, per-date channel assembly and tiled inference (synthetic data)."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from ml.preprocessing import MEAN, STD, build_model_input  # noqa: E402
from vantageq.satellite import sentinel1 as s1  # noqa: E402

SUB_AOI = (85.17, 27.95, 85.22, 27.99)


def write_raster(path, transform, width=20, height=10, epsg=3857):
    import rasterio

    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=3, dtype="float32",
                       crs=f"EPSG:{epsg}", transform=transform) as ds:
        ds.write(np.ones((3, height, width), "float32"))
    return path


def test_make_grid_is_snapped_to_10_unit_grid_with_exact_pixel_size():
    g = s1.make_grid(SUB_AOI, 3857, 10, snap=True)
    assert g.epsg == 3857 and all(v % 10 == 0 for v in g.bounds)
    assert g.res == (10.0, 10.0)
    assert (g.width, g.height) == (557, 505)  # the grid used by the Trishuli inference run


def test_process_request_uses_shared_grid_and_kuro_siwo_processing():
    g = s1.make_grid(SUB_AOI)
    scene = {"start_datetime": "2026-08-28T12:21:41Z", "end_datetime": "2026-08-28T12:22:06Z",
             "orbit_state": "ascending"}
    req = s1.build_process_request(scene, g)
    assert req["input"]["bounds"]["bbox"] == list(g.bounds)
    assert req["input"]["bounds"]["properties"]["crs"].endswith("/3857")
    assert (req["output"]["width"], req["output"]["height"]) == (g.width, g.height)
    proc = req["input"]["data"][0]["processing"]
    assert proc["backCoeff"] == "SIGMA0_ELLIPSOID" and proc["orthorectify"] is True
    assert proc["demInstance"] == "COPERNICUS_30" and proc["speckleFilter"]["windowSizeX"] == 7


def test_check_same_grid_accepts_identical_and_flags_differences(tmp_path):
    from rasterio.transform import from_origin

    t = from_origin(9481080, 3247720, 10, 10)
    a = write_raster(tmp_path / "a.tif", t)
    b = write_raster(tmp_path / "b.tif", t)
    assert s1.check_same_grid([a, b]) == []
    shifted = write_raster(tmp_path / "shifted.tif", from_origin(9481090, 3247720, 10, 10))
    assert any("transform" in p for p in s1.check_same_grid([a, shifted]))
    resized = write_raster(tmp_path / "resized.tif", t, width=21)
    assert any("shape" in p for p in s1.check_same_grid([a, resized]))
    other_crs = write_raster(tmp_path / "utm.tif", t, epsg=32645)
    assert any("crs" in p for p in s1.check_same_grid([a, other_crs]))
    expected = s1.Grid(3857, (9481080.0, 3247620.0, 9481280.0, 3247720.0), 20, 10)
    assert s1.check_same_grid([a, b], expected=expected) == []
    assert s1.check_same_grid([a], expected=s1.Grid(3857, expected.bounds, 40, 20))


def test_select_scenes_orders_pre1_as_most_recent():
    scenes = [{"datetime": d} for d in ("2026-08-04T12:00:00Z", "2026-08-16T12:00:00Z", "2026-08-28T12:00:00Z")]
    sel = s1.select_scenes(scenes, date(2026, 8, 26))
    assert sel["pre"][-1]["datetime"].startswith("2026-08-16")  # pre1
    assert sel["pre"][-2]["datetime"].startswith("2026-08-04")  # pre2


def test_sar_inputs_from_dates_gives_official_channel_order():
    from ml.inference import sar_inputs_from_dates

    val = {("post", "VV"): 0.10, ("post", "VH"): 0.02, ("pre1", "VV"): 0.08, ("pre1", "VH"): 0.03,
           ("pre2", "VV"): 0.06, ("pre2", "VH"): 0.04}
    dates = {d: {p: np.full((5, 7), val[(d, p)], "float32") for p in ("VV", "VH")} for d in ("post", "pre1", "pre2")}
    x = build_model_input(sar_inputs_from_dates(dates["post"], dates["pre1"], dates["pre2"]))
    assert x.shape == (6, 5, 7)
    for i, (d, p) in enumerate([("post", "VV"), ("post", "VH"), ("pre1", "VV"), ("pre1", "VH"),
                                ("pre2", "VV"), ("pre2", "VH")]):
        assert np.allclose(x[i], (val[(d, p)] - MEAN[p.lower()]) / STD[p.lower()], atol=1e-6)


class PixelwiseModel:
    """1x1 convolution: output depends only on the pixel itself, so tiling must not change it."""

    def __init__(self):
        import torch

        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(6, 3, kernel_size=1)

    def __call__(self, x):
        return self.conv(x)

    def eval(self):
        return self


@pytest.mark.parametrize("shape", [(505, 557), (100, 150), (224, 224), (230, 451)])
def test_tiled_inference_matches_untiled_for_pixelwise_model(shape):
    torch = pytest.importorskip("torch")
    from ml.inference import predict_tiled

    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, *shape)).astype("float32")
    model = PixelwiseModel()
    probs = predict_tiled(model, x, torch.device("cpu"), overlap=64)
    assert probs.shape == (3, *shape)
    assert np.allclose(probs.sum(0), 1, atol=1e-5)
    with torch.no_grad():
        direct = torch.softmax(model(torch.from_numpy(x)[None]), 1)[0].numpy()
    assert np.allclose(probs, direct, atol=1e-5)  # no seams, edges handled by padding


def test_tiled_inference_rejects_invalid_overlap():
    torch = pytest.importorskip("torch")
    from ml.inference import predict_tiled

    with pytest.raises(ValueError):
        predict_tiled(PixelwiseModel(), np.zeros((6, 50, 50), "float32"), torch.device("cpu"), overlap=224)
