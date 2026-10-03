"""Sentinel-1 GRD discovery and retrieval on the Copernicus Data Space Ecosystem (CDSE).

- Catalogue search: CDSE STAC API (public, no credentials).
- Scene selection: same relative orbit and orbit direction, first post-event scene plus the latest
  pre-event scenes.
- Retrieval: Sentinel Hub Process API on CDSE, orthorectified linear sigma0 (SIGMA0_ELLIPSOID,
  COPERNICUS_30 DEM, Lee 7x7), matching the Kuro Siwo input representation
  (docs/KUROSIWO_PREPROCESSING.md).
- Grid: one shared `Grid` (CRS, snapped bounds, width, height) per AOI, so every date is returned
  on exactly the same pixel grid; `check_same_grid` verifies it.
"""

from __future__ import annotations

import json
import math
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"
STAC_COLLECTION = "sentinel-1-grd"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

KURO_SIWO_EPSG = 3857  # Kuro Siwo grid: Web Mercator, 10 map units, origin (0, 0)
RESOLUTION = 10

EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV", "VH", "dataMask"]}],
    output: {bands: 3, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(s) {
  return [s.VV, s.VH, s.dataMask];
}
"""
# shadowMask is not requested: Sentinel Hub only provides it with GAMMA0_TERRAIN (radiometric
# terrain correction), and we need SIGMA0_ELLIPSOID to stay comparable with Kuro Siwo.
BAND_NAMES = ("VV", "VH", "dataMask")


# --------------------------------------------------------------------------- catalogue


def stac_search(bbox, start: datetime, end: datetime) -> list[dict]:
    """Return all STAC items intersecting bbox (WGS84) in [start, end], following pagination."""
    params = {
        "collections": STAC_COLLECTION,
        "bbox": ",".join(str(v) for v in bbox),
        "datetime": f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}",
        "limit": "100",
    }
    url = f"{STAC_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    items: list[dict] = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as resp:
            page = json.load(resp)
        items.extend(page.get("features", []))
        url = next((link["href"] for link in page.get("links", []) if link.get("rel") == "next"), None)
    return items


def _point_in_ring(x: float, y: float, ring: list) -> bool:
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def bbox_coverage(geometry: dict, bbox, n: int = 11) -> float:
    """Approximate fraction of bbox covered by a GeoJSON (Multi)Polygon, using an n x n grid."""
    if geometry["type"] == "Polygon":
        rings = [geometry["coordinates"][0]]
    else:
        rings = [poly[0] for poly in geometry["coordinates"]]
    w, s, e, nn = bbox
    pts = [(w + (e - w) * i / (n - 1), s + (nn - s) * j / (n - 1)) for i in range(n) for j in range(n)]
    return sum(any(_point_in_ring(x, y, r) for r in rings) for x, y in pts) / len(pts)


def scene_summary(item: dict, aoi_bbox, sub_bbox=None) -> dict:
    p = item["properties"]
    return {
        "id": item["id"],
        "datetime": p["datetime"],
        "platform": p.get("platform"),
        "mode": p.get("sar:instrument_mode"),
        "polarizations": p.get("sar:polarizations", []),
        "orbit_state": p.get("sat:orbit_state"),
        "relative_orbit": p.get("sat:relative_orbit"),
        "absolute_orbit": p.get("sat:absolute_orbit"),
        "product_type": p.get("product:type"),
        "start_datetime": p.get("start_datetime", p["datetime"]),
        "end_datetime": p.get("end_datetime", p["datetime"]),
        "aoi_coverage": bbox_coverage(item["geometry"], aoi_bbox),
        "sub_aoi_coverage": bbox_coverage(item["geometry"], sub_bbox) if sub_bbox else None,
    }


def filter_scenes(scenes: list[dict], *, relative_orbit: int, orbit_state: str, mode: str = "IW",
                  polarizations=frozenset({"VV", "VH"}), min_coverage: float = 0.99) -> list[dict]:
    """Keep scenes matching mode, polarisations, orbit state, relative orbit and AOI coverage."""
    kept = [
        s for s in scenes
        if s["mode"] == mode
        and set(polarizations).issubset(s["polarizations"])
        and s["orbit_state"] == orbit_state
        and s["relative_orbit"] == relative_orbit
        and s["aoi_coverage"] >= min_coverage
    ]
    unique = {s["datetime"]: s for s in kept}  # one entry per acquisition time
    return [unique[k] for k in sorted(unique)]


def select_scenes(scenes: list[dict], event: date, n_pre: int = 2) -> dict:
    """First scene strictly after the event day and the n_pre latest strictly before it.

    `pre` is ordered oldest -> newest, so pre[-1] is pre1 (most recent) and pre[-2] is pre2.
    Scenes on the event day itself are ambiguous (event time unknown) and skipped.
    """
    def day(s):
        return datetime.fromisoformat(s["datetime"].replace("Z", "+00:00")).date()

    ordered = sorted(scenes, key=lambda s: s["datetime"])
    post = next((s for s in ordered if day(s) > event), None)
    pre = [s for s in ordered if day(s) < event][-n_pre:]
    same_day = [s for s in ordered if day(s) == event]
    return {"post": post, "pre": pre, "skipped_event_day": same_day}


# --------------------------------------------------------------------------- grid


def snap_bounds(bounds, step: float = RESOLUTION) -> tuple:
    """Expand bounds outward to multiples of step, i.e. a grid with origin (0, 0)."""
    w, s, e, n = bounds
    return (math.floor(w / step) * step, math.floor(s / step) * step,
            math.ceil(e / step) * step, math.ceil(n / step) * step)


@dataclass(frozen=True)
class Grid:
    """Output pixel grid shared by all dates of one AOI."""

    epsg: int
    bounds: tuple  # (west, south, east, north) in the grid CRS
    width: int
    height: int

    @property
    def res(self) -> tuple[float, float]:
        w, s, e, n = self.bounds
        return (e - w) / self.width, (n - s) / self.height

    @property
    def transform(self):
        from rasterio.transform import from_bounds

        return from_bounds(*self.bounds, self.width, self.height)


def make_grid(bbox_wgs84, epsg: int = KURO_SIWO_EPSG, res: float = RESOLUTION, snap: bool = True) -> Grid:
    """Grid covering a WGS84 bbox. With snap, bounds are whole `res` pixels on a (0, 0)-origin grid
    (Kuro Siwo / SNAP alignToStandardGrid), so the pixel size is exactly `res`."""
    from rasterio.warp import transform_bounds

    bounds = transform_bounds("EPSG:4326", f"EPSG:{epsg}", *bbox_wgs84)
    if snap:
        bounds = snap_bounds(bounds, res)
    w, s, e, n = bounds
    return Grid(epsg, tuple(float(v) for v in bounds), round((e - w) / res), round((n - s) / res))


def check_same_grid(paths, expected: Grid | None = None, tol: float = 1e-6) -> list[str]:
    """Return a list of problems if the rasters do not share CRS, transform, size and bounds."""
    import rasterio

    problems, ref = [], None
    for p in paths:
        with rasterio.open(p) as ds:
            g = {"crs": ds.crs.to_epsg(), "transform": tuple(ds.transform)[:6], "shape": (ds.height, ds.width),
                 "bounds": tuple(ds.bounds)}
        if expected is not None:
            exp = {"crs": expected.epsg, "transform": tuple(expected.transform)[:6],
                   "shape": (expected.height, expected.width), "bounds": expected.bounds}
            problems += _diff(Path(p).name, g, exp, "expected grid", tol)
        if ref is None:
            ref = (Path(p).name, g)
        else:
            problems += _diff(Path(p).name, g, ref[1], ref[0], tol)
    return problems


def _diff(name: str, g: dict, ref: dict, ref_name: str, tol: float) -> list[str]:
    out = []
    for key in ("crs", "shape"):
        if g[key] != ref[key]:
            out.append(f"{name}: {key} {g[key]} != {ref[key]} ({ref_name})")
    for key in ("transform", "bounds"):
        if any(abs(a - b) > tol for a, b in zip(g[key], ref[key])):
            out.append(f"{name}: {key} {g[key]} != {ref[key]} ({ref_name})")
    return out


# --------------------------------------------------------------------------- retrieval


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_token(client_id: str, client_secret: str) -> str:
    data = urllib.parse.urlencode(
        {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
    ).encode()
    with urllib.request.urlopen(urllib.request.Request(TOKEN_URL, data=data), timeout=60) as resp:
        return json.load(resp)["access_token"]


def build_process_request(scene: dict, grid: Grid) -> dict:
    """Process API request for one acquisition: orthorectified linear sigma0 on exactly `grid`."""
    start = datetime.fromisoformat(scene["start_datetime"].replace("Z", "+00:00")) - timedelta(minutes=1)
    end = datetime.fromisoformat(scene["end_datetime"].replace("Z", "+00:00")) + timedelta(minutes=1)
    return {
        "input": {
            "bounds": {
                "bbox": list(grid.bounds),
                "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{grid.epsg}"},
            },
            "data": [
                {
                    "type": "sentinel-1-grd",
                    "dataFilter": {
                        "timeRange": {"from": f"{start:%Y-%m-%dT%H:%M:%SZ}", "to": f"{end:%Y-%m-%dT%H:%M:%SZ}"},
                        "acquisitionMode": "IW",
                        "polarization": "DV",
                        "orbitDirection": scene["orbit_state"].upper(),
                        "resolution": "HIGH",
                    },
                    "processing": {
                        "backCoeff": "SIGMA0_ELLIPSOID",
                        "orthorectify": True,
                        "demInstance": "COPERNICUS_30",
                        "speckleFilter": {"type": "LEE", "windowSizeX": 7, "windowSizeY": 7},
                        "upsampling": "BILINEAR",
                    },
                }
            ],
        },
        "output": {
            "width": grid.width,
            "height": grid.height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }


def fetch_sigma0(scene: dict, grid: Grid, token: str, out_path: Path) -> Path:
    """Download one acquisition as a 3-band GeoTIFF (VV, VH, dataMask) on `grid`."""
    import rasterio

    req = urllib.request.Request(
        PROCESS_URL,
        data=json.dumps(build_process_request(scene, grid)).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "image/tiff"},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=180) as resp:
        out_path.write_bytes(resp.read())
    with rasterio.open(out_path, "r+") as ds:
        for i, name in enumerate(BAND_NAMES[: ds.count], start=1):
            ds.set_band_description(i, name)
    return out_path
