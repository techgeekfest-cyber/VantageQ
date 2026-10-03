"""Model-input construction of scripts/kurosiwo_sample_smoke.py on synthetic samples (no real data)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import kurosiwo_sample_smoke as ks  # noqa: E402


def synthetic_sample():
    # Distinct constant value per SAR field, all below the clamp, so channel order is identifiable.
    values = {"flood_vv": 0.10, "flood_vh": 0.02, "sec1_vv": 0.08, "sec1_vh": 0.03,
              "sec2_vv": 0.06, "sec2_vh": 0.04}
    return {k: np.full((224, 224), v, "float32") for k, v in values.items()}, values


def test_channel_order_matches_official_post_pre1_pre2():
    sample, values = synthetic_sample()
    x = ks.build_model_input(sample)
    assert x.shape == (6, 224, 224) and x.dtype == np.float32
    for i, name in enumerate(["flood_vv", "flood_vh", "sec1_vv", "sec1_vh", "sec2_vv", "sec2_vh"]):
        pol = name[-2:]
        expected = (values[name] - ks.MEAN[pol]) / ks.STD[pol]
        assert np.allclose(x[i], expected, atol=1e-6), name


def test_clamp_and_nan_handling():
    sample, _ = synthetic_sample()
    sample["flood_vv"][0, 0] = 5.0      # bright -> clamped to 0.15
    sample["flood_vv"][0, 1] = np.nan   # NaN -> 0.15
    sample["flood_vv"][0, 2] = -1.0     # negative -> clamped to 0
    x = ks.build_model_input(sample)
    top = (ks.CLAMP - ks.MEAN["vv"]) / ks.STD["vv"]
    assert np.isclose(x[0, 0, 0], top) and np.isclose(x[0, 0, 1], top)
    assert np.isclose(x[0, 0, 2], (0 - ks.MEAN["vv"]) / ks.STD["vv"])
    assert np.isfinite(x).all()


def test_split_lookup():
    assert ks.split_of(470) == "train"
    assert ks.split_of(514) == "val"
    assert ks.split_of(1111007) == "test"
    assert ks.split_of(999) == "unknown"
