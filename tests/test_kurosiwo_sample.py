"""Shared ml/ preprocessing, Kuro Siwo dataset and U-Net on clearly synthetic data (no real data)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import kurosiwo  # noqa: E402
from ml.preprocessing import CHANNELS, CLAMP, MEAN, STD, build_model_input  # noqa: E402

VALUES = {"post_vv": 0.10, "post_vh": 0.02, "pre1_vv": 0.08, "pre1_vh": 0.03, "pre2_vv": 0.06, "pre2_vh": 0.04}


def sar():
    # Distinct constant per band, all below the clamp, so channel order is identifiable.
    return {k: np.full((224, 224), v, "float32") for k, v in VALUES.items()}


def test_channel_order_is_post_pre1_pre2_vv_vh():
    assert CHANNELS == ("post_vv", "post_vh", "pre1_vv", "pre1_vh", "pre2_vv", "pre2_vh")
    x = build_model_input(sar())
    for i, name in enumerate(CHANNELS):
        pol = name[-2:]
        assert np.allclose(x[i], (VALUES[name] - MEAN[pol]) / STD[pol], atol=1e-6), name


def test_normalization_constants_and_clamping():
    assert CLAMP == 0.15 and MEAN == {"vv": 0.0953, "vh": 0.0264} and STD == {"vv": 0.0427, "vh": 0.0215}
    bands = sar()
    bands["post_vv"][0, :3] = [5.0, np.nan, -1.0]  # bright -> 0.15, NaN -> 0.15, negative -> 0
    bands["post_vh"][0, 0] = 5.0
    x = build_model_input(bands)
    assert np.isclose(x[0, 0, 0], (0.15 - 0.0953) / 0.0427)
    assert np.isclose(x[0, 0, 1], (0.15 - 0.0953) / 0.0427)
    assert np.isclose(x[0, 0, 2], (0.0 - 0.0953) / 0.0427)
    assert np.isclose(x[1, 0, 0], (0.15 - 0.0264) / 0.0215)
    assert np.isfinite(x).all()


def test_model_input_shape_and_dtype():
    x = build_model_input(sar())
    assert x.shape == (6, 224, 224) and x.dtype == np.float32


def test_sar_inputs_maps_webdataset_fields():
    assert kurosiwo.SAR_FIELDS == {"post_vv": "flood_vv", "post_vh": "flood_vh", "pre1_vv": "sec1_vv",
                                   "pre1_vh": "sec1_vh", "pre2_vv": "sec2_vv", "pre2_vh": "sec2_vh"}


def test_split_lookup():
    assert kurosiwo.split_of(470) == "train"
    assert kurosiwo.split_of(1111003) == "val"
    assert kurosiwo.split_of(1111013) == "test"
    assert kurosiwo.split_of(999) == "unknown"


def test_tar_header_parser_accepts_valid_and_rejects_garbage():
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        info = tarfile.TarInfo("000001.info.json")
        info.size = 7
        tf.addfile(info, io.BytesIO(b'{"a":1}'))
    assert kurosiwo._parse_tar_header(buf.getvalue()[:512]) == ("000001.info.json", 7)
    assert kurosiwo._parse_tar_header(b"\0" * 512) is None


def write_synthetic_sample(d: Path, actid=470):
    d.mkdir(parents=True)
    for field, wd in kurosiwo.SAR_FIELDS.items():
        np.save(d / f"{wd}.npy", np.full((1, 224, 224), VALUES[field], "float32"))
    np.save(d / "dem.npy", np.zeros((1, 224, 224), "float32"))
    mask = np.zeros((1, 224, 224), "float32")
    mask[0, :10] = 2
    mask[0, 10:20] = 1
    mask[0, -1] = 3
    np.save(d / "mask.npy", mask)
    np.save(d / "valid_mask.npy", np.ones((1, 224, 224), "float32"))
    (d / "info.json").write_text(json.dumps({"actid": actid}))


def test_dataset_loads_sample_with_expected_shapes(tmp_path):
    torch = pytest.importorskip("torch")
    from ml.dataset import KuroSiwoDataset

    write_synthetic_sample(tmp_path / "s0")
    write_synthetic_sample(tmp_path / "s1", actid=497)
    ds = KuroSiwoDataset(tmp_path)
    assert len(ds) == 2
    x, y = ds[0]
    assert x.shape == (6, 224, 224) and x.dtype == torch.float32
    assert y.shape == (224, 224) and y.dtype == torch.int64
    assert set(torch.unique(y).tolist()) == {0, 1, 2, 3}
    assert torch.allclose(x[0], torch.full((224, 224), (0.10 - 0.0953) / 0.0427), atol=1e-6)


def test_unet_forward_and_backward_on_cpu():
    torch = pytest.importorskip("torch")
    from ml.model import build_unet

    model = build_unet(encoder_weights=None)  # no network access in tests
    x = torch.randn(2, 6, 224, 224)
    y = torch.randint(0, 3, (2, 224, 224))
    y[0, 0, 0] = kurosiwo.IGNORE_INDEX
    out = model(x)
    assert out.shape == (2, 3, 224, 224)
    loss = torch.nn.CrossEntropyLoss(ignore_index=kurosiwo.IGNORE_INDEX)(out, y)
    loss.backward()
    assert torch.isfinite(loss) and model.encoder.conv1.weight.grad is not None


def test_num_classes_excludes_ignore_index_and_ignored_pixels_do_not_affect_loss():
    torch = pytest.importorskip("torch")

    assert kurosiwo.NUM_CLASSES == 3 and kurosiwo.IGNORE_INDEX == 3
    assert kurosiwo.IGNORE_INDEX not in range(kurosiwo.NUM_CLASSES)
    assert set(kurosiwo.LABELS) == set(range(kurosiwo.NUM_CLASSES))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=kurosiwo.IGNORE_INDEX)
    logits = torch.randn(1, kurosiwo.NUM_CLASSES, 4, 4)
    y = torch.randint(0, 3, (1, 4, 4))
    y_ignored = y.clone()
    y_ignored[0, 0, :] = kurosiwo.IGNORE_INDEX
    # Loss with ignored pixels equals the loss over only the remaining pixels.
    expected = torch.nn.functional.cross_entropy(logits[..., 1:, :], y[..., 1:, :])
    assert torch.isclose(loss_fn(logits, y_ignored), expected)
