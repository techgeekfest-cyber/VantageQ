"""One-page Trishuli situation report built only from structured pipeline outputs.

Single source of truth:
- data/processed/trishuli/trishuli_20260828_summary.json            (scenes, flood class %, model)
- data/processed/trishuli/infrastructure/infrastructure_summary.json (impact numbers, rules, OSM snapshot)
- data/processed/trishuli/infrastructure/connectivity_summary.json   (settlement statuses, network)
- data/processed/trishuli/infrastructure/*.geojson                   (map geometry)
- frontend/public/demo/trishuli/manifest.json  (event date, AOI name; its recorded source hashes must
  match the files above, so the report and the dashboard always show the same run)
- docs/DATA_SOURCES.md §7                      (attribution wording, parsed, not retyped)

Every number in the report is read from these files. Rendering: matplotlib (A4 PDF + SVG map
for the HTML version), deterministic for the same inputs.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import math
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

CUT_OFF_DEFINITION = ("Potentially cut off = disconnected in the simplified road graph after removing "
                      "flood-affected edges.")
CUT_OFF_DISCLAIMER = "This is a graph-based indicator, not confirmed real-world isolation."
MISSING_MESSAGE = "Trishuli analysis outputs not found. Run the analysis pipeline first."

STATUS_LABELS = {  # same wording as the dashboard (frontend/src/lib/format.ts)
    "potentially_cut_off": "Potentially cut off (simplified graph indicator)",
    "connected": "Connected in simplified road graph",
    "not_on_main_network_before": "Already separate from main road network (before flood)",
    "no_road_within_snap_distance": "No mapped road within snap distance",
}
LIMITATIONS = [
    "Trishuli flood prediction is unvalidated against ground truth in this project.",
    "Terrain-related SAR false positives are possible.",
    "Bridge intersection does not establish bridge damage or impassability.",
    "Potentially cut off is a simplified graph indicator, not confirmed isolation.",
    "Footpaths are excluded from the connectivity graph.",
    "The road graph is limited to the analysis AOI.",
]
ATTRIBUTION_SOURCES = ("Sentinel-1", "Copernicus DEM", "Kuro Siwo dataset", "OpenStreetMap")
FORBIDDEN = re.compile(r"\b(damaged|destroyed)\b", re.IGNORECASE)
# "confirmed (real-world) isolation" may only appear negated ("not confirmed ... isolation").
CONFIRMED_ISOLATION = re.compile(r"(\bnot\s+)?\bconfirmed\s+(real-world\s+)?isolation\b", re.IGNORECASE)


class ReportDataError(RuntimeError):
    pass


@dataclass
class Paths:
    root: Path

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed" / "trishuli"

    @property
    def infra(self) -> Path:
        return self.processed / "infrastructure"

    @property
    def manifest(self) -> Path:
        return self.root / "frontend" / "public" / "demo" / "trishuli" / "manifest.json"

    @property
    def data_sources_md(self) -> Path:
        return self.root / "docs" / "DATA_SOURCES.md"

    def required(self) -> dict[str, Path]:
        layers = ["flood_extent", "roads", "affected_roads", "affected_bridges", "affected_buildings", "settlements"]
        return {
            "inference_summary": self.processed / "trishuli_20260828_summary.json",
            "infrastructure_summary": self.infra / "infrastructure_summary.json",
            "connectivity_summary": self.infra / "connectivity_summary.json",
            **{name: self.infra / f"{name}.geojson" for name in layers},
            "manifest": self.manifest,
            "data_sources_md": self.data_sources_md,
        }


@dataclass
class ReportData:
    inference: dict
    infrastructure: dict
    connectivity: dict
    manifest: dict
    layers: dict[str, dict]
    attribution: list[str]
    files: dict[str, Path] = field(default_factory=dict)


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_attribution(md: str) -> list[str]:
    """Attribution lines from the table in docs/DATA_SOURCES.md §7 for the sources used here."""
    sec = md.split("## 7. Licensing and attribution summary", 1)
    if len(sec) < 2:
        raise ReportDataError("docs/DATA_SOURCES.md has no '## 7. Licensing and attribution summary' table")
    lines = []
    for row in sec[1].split("\n## ", 1)[0].splitlines():
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0].startswith(ATTRIBUTION_SOURCES):
            if "[VERIFY" in row:
                raise ReportDataError(f"unverified attribution in docs/DATA_SOURCES.md: {cells[0]}")
            text = cells[2].replace("**", "").replace('"', "").strip()
            lines.append(f"{cells[0]} ({cells[1]}): {text}")
    if len(lines) < len(ATTRIBUTION_SOURCES):
        raise ReportDataError("attribution table in docs/DATA_SOURCES.md is incomplete")
    return lines


def load(root: Path) -> ReportData:
    paths = Paths(root)
    files = paths.required()
    missing = [str(p.relative_to(root)) for p in files.values() if not p.exists()]
    if missing:
        raise ReportDataError(f"{MISSING_MESSAGE} Missing: {', '.join(missing)}")
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    manifest = read(files["manifest"])
    stale = [rel for rel, digest in manifest.get("sources_sha256", {}).items()
             if (root / rel).exists() and _sha256(root / rel) != digest]
    if stale:
        raise ReportDataError("Dashboard snapshot does not match the current analysis outputs "
                              f"({', '.join(stale)}). Run scripts/build_demo_snapshot.py first.")
    layers = {k: read(files[k]) for k in ("flood_extent", "roads", "affected_roads", "affected_bridges",
                                          "affected_buildings", "settlements")}
    data = ReportData(read(files["inference_summary"]), read(files["infrastructure_summary"]),
                      read(files["connectivity_summary"]), manifest, layers,
                      parse_attribution(files["data_sources_md"].read_text(encoding="utf-8")), files)
    _validate(data)
    return data


def _validate(d: ReportData) -> None:
    need = [("infrastructure.flood_area_km2_utm", d.infrastructure.get("flood_area_km2_utm")),
            ("infrastructure.roads.segments_affected", d.infrastructure.get("roads", {}).get("segments_affected")),
            ("infrastructure.roads.affected_length_km", d.infrastructure.get("roads", {}).get("affected_length_km")),
            ("infrastructure.bridges.affected", d.infrastructure.get("bridges", {}).get("affected")),
            ("infrastructure.buildings.affected", d.infrastructure.get("buildings", {}).get("affected")),
            ("inference.class_pct.flood", d.inference.get("class_pct", {}).get("flood"))]
    bad = [k for k, v in need if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v)]
    if not isinstance(d.connectivity.get("settlements"), list):
        bad.append("connectivity.settlements")
    if bad:
        raise ReportDataError(f"Required fields missing in analysis outputs: {', '.join(bad)}")


# --------------------------------------------------------------------------- content


def _date(iso: str) -> str:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%d %B %Y")


def _n(v: float, digits: int = 0) -> str:
    return f"{v:,.{digits}f}"


def build_content(d: ReportData) -> dict:
    """All report text, derived from the loaded outputs. Renderers only lay this out."""
    s, c, inf, m = d.infrastructure, d.connectivity, d.inference, d.manifest
    sc = inf["scenes"]
    net = c["network"]
    settlements = c["settlements"]
    cut = [x for x in settlements if x["status"] == "potentially_cut_off"]
    osm_date = s["osm_snapshot"]
    return {
        "title": "VANTAGEQ",
        "tagline": "Satellite-Powered Flood Intelligence for Disaster Response",
        "doc_type": "SITUATION REPORT · research prototype",
        "event": f"Trishuli Valley Flood Event · {_date(m['event_date'] + 'T00:00:00Z')}",
        "aoi": (f"AOI: {m['aoi_name']} ({inf['sub_aoi_wgs84'][0]}–{inf['sub_aoi_wgs84'][2]}° E, "
                f"{inf['sub_aoi_wgs84'][1]}–{inf['sub_aoi_wgs84'][3]}° N)"),
        "acquisition": (f"Post-event Sentinel-1: {_date(sc['post']['datetime'])}   ·   Pre-event Sentinel-1: "
                        f"{_date(sc['pre1']['datetime'])}, {_date(sc['pre2']['datetime'])}   ·   "
                        f"relative orbit {sc['post']['relative_orbit']} {sc['post']['orbit_state']}"),
        "banner": ("Indicators derived from an unvalidated model prediction and pre-event OpenStreetMap. "
                   "Not confirmed damage or isolation."),
        "flood": {
            "value": f"{_n(s['flood_area_km2_utm'], 3)} km²",
            "label": "Predicted flood extent",
            "detail": (f"{_n(s['flood_area_km2_utm'] * 100, 1)} ha · {_n(inf['class_pct']['flood'], 2)} % of the "
                       f"analysed AOI pixels predicted as flood"),
            "note": "Model prediction (class “flood”); not a confirmed flood area.",
        },
        "kpis": [
            (_n(s["roads"]["segments_affected"]), "Road ways potentially affected",
             f"of {_n(s['roads']['segments_total'])} mapped road ways"),
            (f"{_n(s['roads']['affected_length_km'] * 1000)} m", "Road length potentially affected",
             f"inside predicted flood; of {_n(s['roads']['length_km_total'], 1)} km mapped"),
            (_n(s["bridges"]["affected"]), "Bridges intersecting predicted flood",
             f"of {_n(s['bridges']['total'])} mapped bridges"),
            (_n(s["buildings"]["affected"]), "Buildings potentially affected",
             f"of {_n(s['buildings']['total'])} mapped buildings"),
        ],
        "connectivity_summary": (
            f"{_n(s['settlements']['analyzed'])} settlements analysed; {_n(len(cut))} potentially cut off"
            + (f": {', '.join(x['name'] or str(x['osm_id']) for x in cut)}." if cut else ".")),
        "network": (f"{_n(net['blocked_edges'])} road-graph edges blocked by predicted flood; main road network "
                    f"{_n(net['main_network_km_before'], 1)} km → {_n(net['main_network_km_after'], 1)} km "
                    f"(vehicle roads in AOI)."),
        "settlements": [
            {"name": x["name"] or str(x["osm_id"]), "place": x["place"], "status": x["status"],
             "status_label": STATUS_LABELS.get(x["status"], x["status"].replace("_", " ")),
             "access": (f"reachable road network {_n(x['reachable_road_km_before'], 2)} km → "
                        f"{_n(x['reachable_road_km_after'], 2)} km; nearest road node "
                        f"{_n(x['snap_distance_m'])} m")}
            for x in sorted(settlements, key=lambda x: (x["status"] != "potentially_cut_off", x["name"] or ""))],
        "definition": CUT_OFF_DEFINITION,
        "disclaimer": CUT_OFF_DISCLAIMER,
        "methodology_chain": ["Sentinel-1 GRD", "Kuro Siwo-trained U-Net", "predicted flood extent",
                              "historical OSM infrastructure", "spatial impact analysis",
                              "simplified road-network connectivity"],
        "methodology": [
            f"Historical OSM: {osm_date} (Overpass historical/attic snapshot, pre-event).",
            "Model: Kuro Siwo-trained U-Net; 6-channel Sentinel-1 temporal input (VV/VH: post, pre1, pre2).",
            "Prototype impact rules: " + "; ".join([
                "roads ≥ 10 m inside predicted flood", "bridges intersect predicted flood",
                "buildings > 0.01 m² overlap"]) + ". Not calibrated damage thresholds.",
        ],
        "limitations": LIMITATIONS,
        "attribution": d.attribution,
        "provenance": (f"Generated from VantageQ pipeline outputs; checkpoint sha256 "
                       f"{inf['checkpoint_sha256_16']}…; OSM snapshot {osm_date}."),
    }


def all_text(content: dict) -> str:
    out = []

    def walk(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            [walk(x) for x in v.values()]
        elif isinstance(v, (list, tuple)):
            [walk(x) for x in v]
    walk(content)
    return "\n".join(out)


def check_wording(content: dict) -> None:
    text = all_text(content)
    hit = FORBIDDEN.search(text)
    if hit:
        raise ReportDataError(f"forbidden wording in report: '{hit.group(0)}'")
    for m in CONFIRMED_ISOLATION.finditer(text):
        if not m.group(1):
            raise ReportDataError(f"forbidden wording in report: '{m.group(0)}' (only allowed negated)")


# --------------------------------------------------------------------------- map


def _rings(geom: dict):
    t, c = geom["type"], geom["coordinates"]
    if t == "Polygon":
        return [c[0]]
    if t == "MultiPolygon":
        return [p[0] for p in c]
    return []


def _lines(geom: dict):
    t, c = geom["type"], geom["coordinates"]
    return [c] if t == "LineString" else c if t == "MultiLineString" else []


def draw_map(ax, d: ReportData) -> None:
    """Static map of the AOI from the derived GeoJSON (no basemap tiles; offline and deterministic)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    w, s, e, n = d.inference["sub_aoi_wgs84"]
    L = d.layers
    for f in L["roads"]["features"]:
        major = f["properties"].get("highway") in ("primary", "secondary", "trunk")
        for ln in _lines(f["geometry"]):
            xs, ys = zip(*ln)
            ax.plot(xs, ys, color="#9a9a9a", lw=1.1 if major else 0.45, zorder=1)
    for f in L["flood_extent"]["features"]:
        for ring in _rings(f["geometry"]):
            xs, ys = zip(*ring)
            ax.fill(xs, ys, color="#2b8cbe", alpha=0.75, lw=0.3, ec="#08519c", zorder=2)
    for f in L["affected_roads"]["features"]:
        for ln in _lines(f["geometry"]):
            xs, ys = zip(*ln)
            ax.plot(xs, ys, color="#e34a33", lw=1.3, ls=(0, (3, 2)), zorder=3)
    for f in L["affected_buildings"]["features"]:
        ring = _rings(f["geometry"])[0]
        ax.plot(sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring), "o", ms=2.6,
                color="#fd8d3c", mec="#7f2704", mew=0.3, zorder=4)
    for f in L["affected_bridges"]["features"]:
        pts = [p for ln in _lines(f["geometry"]) for p in ln]
        ax.plot(sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts), "D", ms=5,
                color="#7a0177", mec="white", mew=0.7, zorder=5)
    tone = {"potentially_cut_off": "#cb181d", "connected": "#238b45"}
    for f in sorted(L["settlements"]["features"], key=lambda f: f["properties"].get("name") or ""):
        p = f["properties"]
        x, y = f["geometry"]["coordinates"]
        cut = p["status"] == "potentially_cut_off"
        ax.plot(x, y, "*" if cut else "o", ms=11 if cut else 6, color=tone.get(p["status"], "#6a51a3"),
                mec="black", mew=0.5, zorder=6)
        ax.annotate(p.get("name") or str(p["osm_id"]), (x, y), xytext=(5, 3), textcoords="offset points",
                    fontsize=6.5, fontweight="bold", zorder=7,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "#888", "lw": 0.4, "alpha": 0.9})
    ax.plot([w, e, e, w, w], [s, s, n, n, s], color="#333", lw=0.6, ls="--", zorder=1)
    lat = (s + n) / 2
    ax.set_aspect(1 / math.cos(math.radians(lat)))
    pad = 0.0015
    ax.set_xlim(w - pad, e + pad)
    ax.set_ylim(s - pad, n + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#bbbbbb")
    km_deg = 1 / (111.32 * math.cos(math.radians(lat)))  # 1 km in degrees of longitude
    x0, y0 = e - 0.004 - km_deg, s + 0.0022
    ax.plot([x0, x0 + km_deg], [y0, y0], color="black", lw=1.5, zorder=8)
    ax.text(x0 + km_deg / 2, y0 + 0.0006, "1 km", ha="center", fontsize=6, zorder=8)
    ax.legend(handles=[
        Patch(color="#2b8cbe", alpha=0.75, label="Predicted flood extent"),
        Line2D([], [], color="#e34a33", lw=1.3, ls=(0, (3, 2)), label="Road way potentially affected"),
        Line2D([], [], marker="D", color="#7a0177", ls="", ms=5, label="Bridge intersecting predicted flood"),
        Line2D([], [], marker="o", color="#fd8d3c", ls="", ms=3.5, label="Building potentially affected"),
        Line2D([], [], marker="*", color="#cb181d", ls="", ms=9, label="Settlement potentially cut off"),
        Line2D([], [], marker="o", color="#6a51a3", ls="", ms=5, label="Other settlement"),
        Line2D([], [], color="#9a9a9a", lw=1, label="Road (OSM 2026-08-25)")],
        loc="upper right", fontsize=5.6, framealpha=0.92, borderpad=0.4, handlelength=1.6)


# --------------------------------------------------------------------------- PDF (A4, one page)

A4 = (8.27, 11.69)
NAVY = "#12355b"


def _wrap(text: str, width: int) -> str:
    return textwrap.fill(text, width=width, break_long_words=False)


def render_figure(d: ReportData, content: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig = plt.figure(figsize=A4)
    T = fig.text
    L, R = 0.06, 0.94

    def section(y, title):
        T(L, y, title, fontsize=9.5, fontweight="bold", color=NAVY, va="top")
        fig.add_artist(Rectangle((L, y - 0.017), R - L, 0.0012, color=NAVY, lw=0, transform=fig.transFigure))

    # Header
    fig.add_artist(Rectangle((0, 0.915), 1, 0.085, color=NAVY, lw=0, transform=fig.transFigure))
    T(L, 0.975, content["title"], fontsize=20, fontweight="bold", color="white", va="top")
    T(L, 0.94, content["tagline"], fontsize=9, color="white", va="top")
    T(R, 0.972, content["doc_type"], fontsize=7.5, color="white", ha="right", va="top")
    T(L, 0.903, content["event"], fontsize=12, fontweight="bold", va="top")
    T(L, 0.881, content["acquisition"], fontsize=7.3, va="top")
    T(L, 0.866, content["aoi"], fontsize=7.3, color="#444", va="top")
    fig.add_artist(Rectangle((L, 0.832), R - L, 0.024, color="#fff4d6", ec="#e8cf8a", lw=0.6,
                             transform=fig.transFigure))
    T(0.5, 0.844, content["banner"], fontsize=7.3, color="#6b5200", ha="center", va="center")

    # Section 1 + map
    section(0.818, "1  FLOOD OVERVIEW")
    ax = fig.add_axes((L, 0.478, 0.53, 0.318))
    draw_map(ax, d)
    xr = 0.62
    f = content["flood"]
    T(xr, 0.788, f["value"], fontsize=20, fontweight="bold", color=NAVY, va="top")
    T(xr, 0.756, f["label"], fontsize=9, fontweight="bold", va="top")
    T(xr, 0.740, _wrap(f["detail"], 44), fontsize=7.2, color="#333", va="top", linespacing=1.3)
    T(xr, 0.713, _wrap(f["note"], 46), fontsize=6.8, color="#666", style="italic", va="top", linespacing=1.3)

    # Section 2: KPI cards (2 x 2) to the right of the map
    T(xr, 0.680, "2  POTENTIAL INFRASTRUCTURE IMPACT", fontsize=8.6, fontweight="bold", color=NAVY, va="top")
    fig.add_artist(Rectangle((xr, 0.663), R - xr, 0.0012, color=NAVY, lw=0, transform=fig.transFigure))
    cw, ch, gap = (R - xr - 0.012) / 2, 0.092, 0.012
    for i, (val, label, detail) in enumerate(content["kpis"]):
        cx, cy = xr + (i % 2) * (cw + gap), 0.560 - (i // 2) * (ch + gap)
        fig.add_artist(FancyBboxPatch((cx, cy), cw, ch, boxstyle="round,pad=0,rounding_size=0.006",
                                      fc="#f4f6f9", ec="#cfd6df", lw=0.6, transform=fig.transFigure))
        T(cx + 0.01, cy + ch - 0.01, val, fontsize=15, fontweight="bold", color=NAVY, va="top")
        T(cx + 0.01, cy + ch - 0.041, _wrap(label, 20), fontsize=7, fontweight="bold", va="top", linespacing=1.2)
        T(cx + 0.01, cy + 0.008, _wrap(detail, 25), fontsize=6, color="#555", va="bottom", linespacing=1.2)

    # Section 3: connectivity
    section(0.458, "3  CONNECTIVITY (SIMPLIFIED ROAD GRAPH)")
    T(L, 0.432, content["connectivity_summary"], fontsize=8.3, fontweight="bold", va="top")
    T(L, 0.416, content["network"], fontsize=7, color="#333", va="top")
    y = 0.397
    colors = {"potentially_cut_off": "#b3141b", "connected": "#1d6b3a"}
    for row in content["settlements"]:
        col = colors.get(row["status"], "#4b3f8a")
        fig.add_artist(Rectangle((L, y - 0.022), 0.004, 0.023, color=col, lw=0, transform=fig.transFigure))
        T(L + 0.012, y, f"{row['name']} ({row['place']})", fontsize=7.8, fontweight="bold", va="top")
        T(L + 0.2, y, row["status_label"], fontsize=7.4, fontweight="bold", color=col, va="top")
        T(L + 0.012, y - 0.0125, row["access"], fontsize=6.8, color="#444", va="top")
        y -= 0.029
    T(L, y - 0.002, content["definition"], fontsize=7.2, va="top")
    T(L, y - 0.016, content["disclaimer"], fontsize=7.2, fontweight="bold", va="top")

    # Sections 4 + 5 side by side
    top = y - 0.038
    mid = 0.515
    T(L, top, "4  METHODOLOGY", fontsize=9.5, fontweight="bold", color=NAVY, va="top")
    fig.add_artist(Rectangle((L, top - 0.017), mid - L - 0.02, 0.0012, color=NAVY, lw=0, transform=fig.transFigure))
    chain = _wrap(" → ".join(content["methodology_chain"]), 58)
    yy = top - 0.028
    T(L, yy, chain, fontsize=7.2, fontweight="bold", va="top", linespacing=1.35)
    yy -= 0.0128 * (chain.count("\n") + 1) + 0.006
    for line in content["methodology"]:
        txt = _wrap(line, 62)
        T(L, yy - 0.004, txt, fontsize=6.6, color="#333", va="top", linespacing=1.25)
        yy -= 0.0105 * (txt.count("\n") + 1) + 0.005

    T(mid, top, "5  LIMITATIONS", fontsize=9.5, fontweight="bold", color=NAVY, va="top")
    fig.add_artist(Rectangle((mid, top - 0.017), R - mid, 0.0012, color=NAVY, lw=0, transform=fig.transFigure))
    yy = top - 0.028
    for line in content["limitations"]:
        txt = _wrap(line, 60)
        T(mid, yy, "•", fontsize=7.2, va="top")
        T(mid + 0.015, yy, txt, fontsize=7.2, va="top", linespacing=1.25)
        yy -= 0.0125 * (txt.count("\n") + 1) + 0.005

    # Attribution footer
    fig.add_artist(Rectangle((L, 0.091), R - L, 0.0008, color="#999", lw=0, transform=fig.transFigure))
    yy = 0.086
    T(L, yy, "Data attribution", fontsize=6.6, fontweight="bold", va="top")
    yy -= 0.011
    for line in content["attribution"]:
        txt = _wrap(line, 160)
        T(L, yy, txt, fontsize=5.8, color="#333", va="top", linespacing=1.2)
        yy -= 0.0085 * (txt.count("\n") + 1) + 0.002
    T(L, 0.012, content["provenance"], fontsize=5.6, color="#777", va="bottom")
    return fig


def layout_problems(fig) -> list[str]:
    """Text outside the page or text boxes overlapping each other (map interior excluded)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    boxes = [(t.get_text(), t.get_window_extent(r)) for t in fig.texts if t.get_text().strip()]
    probs = [f"outside page: {s[:40]!r}" for s, b in boxes if b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H]
    probs += [f"outside side margins: {s[:40]!r}" for s, b in boxes if b.x0 < 0.04 * W or b.x1 > 0.96 * W]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.x0 < b.x1 - 0.5 and b.x0 < a.x1 - 0.5 and a.y0 < b.y1 - 0.5 and b.y0 < a.y1 - 0.5:
                probs.append(f"overlap: {boxes[i][0][:30]!r} / {boxes[j][0][:30]!r}")
    for ax in fig.axes:  # nothing outside the map axes may overlap it
        ab = ax.get_window_extent(r)
        for s, b in boxes:
            if ab.x0 < b.x1 and b.x0 < ab.x1 and ab.y0 < b.y1 and b.y0 < ab.y1:
                probs.append(f"text over map: {s[:40]!r}")
    return probs


def write_pdf(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None,
                                              "Creator": "VantageQ scripts/generate_situation_report.py",
                                              "Title": "VantageQ Trishuli Situation Report"})


def pdf_page_count(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page\b(?!s)", path.read_bytes()))


# --------------------------------------------------------------------------- HTML


def render_html(d: ReportData, content: dict) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "vantageq", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    draw_map(ax, d)
    fig.tight_layout(pad=0.2)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    svg = buf.getvalue()
    svg = svg[svg.index("<svg"):]
    e = html.escape
    kpis = "".join(f'<div class="kpi"><b>{e(v)}</b><span>{e(lbl)}</span><small>{e(dt)}</small></div>'
                   for v, lbl, dt in content["kpis"])
    rows = "".join(f'<li class="{e(r["status"])}"><b>{e(r["name"])}</b> ({e(r["place"])}) · '
                   f'<em>{e(r["status_label"])}</em><br><small>{e(r["access"])}</small></li>'
                   for r in content["settlements"])
    f = content["flood"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>VantageQ Trishuli Situation Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:0 auto;color:#1f2933;font-size:13px}}
header{{background:{NAVY};color:#fff;padding:12px 18px}} header h1{{margin:0;font-size:24px}}
.banner{{background:#fff4d6;border:1px solid #e8cf8a;padding:4px 10px;color:#6b5200;font-size:12px}}
h2{{color:{NAVY};font-size:14px;border-bottom:1px solid {NAVY};margin:14px 0 6px}}
.row{{display:flex;gap:16px}} .row>div{{flex:1}} svg{{width:100%;height:auto}}
svg text{{font-family:system-ui,sans-serif !important}}
.kpis{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.kpi{{background:#f4f6f9;border:1px solid #cfd6df;border-radius:6px;padding:6px 8px}}
.kpi b{{display:block;font-size:20px;color:{NAVY}}} .kpi span{{display:block;font-weight:600}}
.kpi small,small{{color:#555}} ul{{padding-left:18px}} li.potentially_cut_off em{{color:#b3141b}}
footer{{font-size:11px;color:#555;border-top:1px solid #999;margin-top:14px;padding-top:6px}}
</style></head><body>
<header><h1>{e(content['title'])}</h1><div>{e(content['tagline'])}</div></header>
<p><b>{e(content['event'])}</b><br>{e(content['acquisition'])}<br><small>{e(content['aoi'])}</small></p>
<div class="banner">{e(content['banner'])}</div>
<div class="row"><div><h2>1 Flood overview</h2>{svg}</div>
<div><p class="kpi"><b>{e(f['value'])}</b><span>{e(f['label'])}</span><small>{e(f['detail'])}<br>{e(f['note'])}</small></p>
<h2>2 Potential infrastructure impact</h2><div class="kpis">{kpis}</div></div></div>
<h2>3 Connectivity (simplified road graph)</h2>
<p><b>{e(content['connectivity_summary'])}</b><br><small>{e(content['network'])}</small></p>
<ul>{rows}</ul><p>{e(content['definition'])}<br><b>{e(content['disclaimer'])}</b></p>
<div class="row"><div><h2>4 Methodology</h2><p>{' → '.join(e(x) for x in content['methodology_chain'])}</p>
<ul>{''.join(f'<li>{e(x)}</li>' for x in content['methodology'])}</ul></div>
<div><h2>5 Limitations</h2><ul>{''.join(f'<li>{e(x)}</li>' for x in content['limitations'])}</ul></div></div>
<footer><b>Data attribution</b><br>{'<br>'.join(e(x) for x in content['attribution'])}<br>{e(content['provenance'])}</footer>
</body></html>
"""


def generate(root: Path, out_dir: Path) -> dict:
    """Load, validate, render PDF (+ HTML). Raises ReportDataError on missing/invalid data or layout."""
    import matplotlib.pyplot as plt

    d = load(root)
    content = build_content(d)
    check_wording(content)
    fig = render_figure(d, content)
    problems = layout_problems(fig)
    if problems:
        plt.close(fig)
        raise ReportDataError("report layout problems: " + "; ".join(problems))
    pdf = out_dir / "VantageQ_Trishuli_Situation_Report.pdf"
    write_pdf(fig, pdf)
    plt.close(fig)
    pages = pdf_page_count(pdf)
    if pages != 1:
        raise ReportDataError(f"report has {pages} pages, expected 1")
    html_path = out_dir / "VantageQ_Trishuli_Situation_Report.html"
    html_path.write_text(render_html(d, content), encoding="utf-8")
    return {"pdf": pdf, "html": html_path, "pages": pages, "content": content,
            "files": {k: str(v.relative_to(root)) for k, v in d.files.items()}}
