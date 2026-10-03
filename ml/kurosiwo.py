"""Kuro Siwo labelled GRD webdataset access: official splits, labels, shard probing, streaming, I/O.

Samples are streamed from Hugging Face (orion-ai-lab/Kuro-Siwo-Webdataset) and saved as one
directory per sample with .npy files + info.json. See docs/KUROSIWO_DATASET.md.
"""

from __future__ import annotations

import io
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

HF_REPO = "orion-ai-lab/Kuro-Siwo-Webdataset"
HF_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/"
TRAIN_SHARDS = tuple(f"train_GRD/shard-{i:05d}.tar" for i in range(5))
SAMPLE_BYTES = 1.82e6  # approximate size of one labelled GRD sample in the tar stream

# Official splits by event (activation) ID, configs/train/data_config.json. The Hugging Face
# train/test shards do NOT follow these (e.g. val event 1111003 is in train_GRD), so always filter.
TRAIN_ACTS = frozenset({130, 470, 555, 118, 174, 324, 421, 554, 427, 518, 502, 498, 497, 496, 492, 147,
                        267, 273, 275, 417, 567, 1111011, 1111004, 1111009, 1111010, 1111006, 1111005})
VAL_ACTS = frozenset({514, 559, 279, 520, 437, 1111003, 1111008})
TEST_ACTS = frozenset({321, 561, 445, 562, 411, 1111002, 277, 1111007, 205, 1111013})

LABELS = {0: "no water", 1: "permanent water", 2: "flood"}
# Value 3 also occurs in the data (not on the dataset card). In the 57-sample subset, every invalid
# pixel (valid_mask == 0) had label 3, and some valid pixels did too; why those valid pixels are 3 is
# not established. 3 is not a model class: the official loss ignores it (ignore_index=3).
IGNORE_INDEX = 3
LABEL_NAMES = {**LABELS, IGNORE_INDEX: "ignore"}
NUM_CLASSES = 3

# Webdataset field -> canonical model-input name (ml/preprocessing.py).
SAR_FIELDS = {"post_vv": "flood_vv", "post_vh": "flood_vh", "pre1_vv": "sec1_vv",
              "pre1_vh": "sec1_vh", "pre2_vv": "sec2_vv", "pre2_vh": "sec2_vh"}
SAMPLE_FIELDS = (*SAR_FIELDS.values(), "dem", "mask", "valid_mask", "info.json")


def split_of(actid: int) -> str:
    return ("train" if actid in TRAIN_ACTS else "val" if actid in VAL_ACTS
            else "test" if actid in TEST_ACTS else "unknown")


# --------------------------------------------------------------------------- sample I/O


def load_sample(d: Path) -> dict:
    """Load one saved sample: arrays squeezed to (224, 224) plus parsed info.json under 'info'."""
    s = {f: np.load(d / f"{f}.npy").squeeze(0) for f in SAMPLE_FIELDS if f != "info.json"}
    s["info"] = json.loads((d / "info.json").read_text())
    return s


def sar_inputs(sample: dict) -> dict[str, np.ndarray]:
    """Map a loaded sample to the canonical keys expected by ml.preprocessing.build_model_input."""
    return {name: sample[field] for name, field in SAR_FIELDS.items()}


def sample_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if (p / "info.json").exists()) if root.exists() else []


# --------------------------------------------------------------------------- HTTP helpers


class _Http:
    """Range-request helper that resolves the Hugging Face redirect once and counts bytes."""

    def __init__(self, shard: str):
        req = urllib.request.Request(HF_RESOLVE + shard, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            self.url = resp.geturl()  # signed CDN URL, valid for the session
            self.size = int(resp.headers["Content-Range"].split("/")[1])
        self.bytes = 0

    def get(self, start: int, length: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + length - 1}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        self.bytes += len(data)
        return data


def _parse_tar_header(block: bytes) -> tuple[str, int] | None:
    if len(block) < 512 or block[257:262] != b"ustar":
        return None
    checksum = int(block[148:156].rstrip(b"\0 ") or b"0", 8)
    if checksum != sum(block[:148]) + 8 * 32 + sum(block[156:512]):
        return None
    return block[:100].rstrip(b"\0").decode(), int(block[124:136].rstrip(b"\0 ") or b"0", 8)


def _probe(http: _Http, offset: int, max_hops: int = 14) -> dict | None:
    """Event metadata of the first sample whose info.json follows `offset` (a few small reads)."""
    offset -= offset % 512
    window = http.get(offset, 202 * 1024)  # > one member (~201 KB), so it contains a header
    for i in range(0, len(window) - 511, 512):
        header = _parse_tar_header(window[i:i + 512])
        if header:
            break
    else:
        return None
    pos = offset + i
    for _ in range(max_hops):
        name, size = header
        if name.endswith(".info.json"):
            info = json.loads(http.get(pos + 512, size))
            return {"offset": pos, "key": name.split(".")[0], "actid": info["actid"],
                    "aoiid": info["aoiid"], "pflood": info["pflood"]}
        pos += 512 + (size + 511) // 512 * 512
        header = _parse_tar_header(http.get(pos, 512))
        if not header:
            return None
    return None


def probe_shard(shard: str, step_bytes: float = 1e9) -> tuple[list[dict], int]:
    """Probe a shard every `step_bytes`; return (probes, bytes fetched). Only headers + info.json."""
    http = _Http(shard)
    probes = []
    for off in range(0, http.size, int(step_bytes)):
        p = _probe(http, off)
        if p:
            probes.append({"shard": shard, **p})
    return probes, http.bytes


# --------------------------------------------------------------------------- streaming


class ByteCapReached(RuntimeError):
    """Raised when a stream reaches its byte cap."""


class _CountingReader(io.RawIOBase):
    """Wraps an HTTP response, counts bytes and refuses to read beyond a hard cap."""

    def __init__(self, resp, max_bytes: int):
        self.resp, self.max_bytes, self.n = resp, max_bytes, 0

    def readable(self):
        return True

    def readinto(self, buf):
        if self.n >= self.max_bytes:
            raise ByteCapReached(f"byte cap of {self.max_bytes / 1e6:.0f} MB reached")
        data = self.resp.read(min(len(buf), self.max_bytes - self.n))
        self.n += len(data)
        buf[: len(data)] = data
        return len(data)


def stream_samples(shard: str, out_dir: Path, n: int, start: int = 0, max_bytes: float = 30e6,
                   prefix: str = "", only_actid: int | None = None) -> tuple[list[Path], int]:
    """Stream a shard from byte `start` (a tar header boundary), save the first n complete samples.

    A sample cut off by `start` is incomplete and skipped. Returns (saved dirs, bytes streamed).
    """
    req = urllib.request.Request(HF_RESOLVE + shard, headers={"Range": f"bytes={start}-"})
    resp = urllib.request.urlopen(req, timeout=120)
    reader = _CountingReader(resp, int(max_bytes))
    partial: dict[str, dict[str, bytes]] = {}
    saved: list[Path] = []
    try:
        with tarfile.open(fileobj=io.BufferedReader(reader), mode="r|") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                key, field = member.name.split(".", 1)
                partial.setdefault(key, {})[field.removesuffix(".npy")] = tf.extractfile(member).read()
                if set(SAMPLE_FIELDS) <= partial[key].keys():
                    files = partial.pop(key)
                    if only_actid is not None and json.loads(files["info.json"])["actid"] != only_actid:
                        continue
                    saved.append(_save(out_dir / f"{prefix}{key}", files))
                    if len(saved) >= n:
                        break
    except (ByteCapReached, tarfile.ReadError) as err:  # cap reached / stream cut: keep what was saved
        print(f"  stopped early ({err}); {len(saved)} samples saved")
    finally:
        resp.close()
    return saved, reader.n


def _save(d: Path, files: dict[str, bytes]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for field, data in files.items():
        (d / (field if field == "info.json" else f"{field}.npy")).write_bytes(data)
    return d
