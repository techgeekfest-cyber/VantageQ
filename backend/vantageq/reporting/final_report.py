"""Final hackathon technical report (≤ 6 A4 pages), generated from repository outputs and docs.

Numbers come from:
- pipeline outputs via situation_report.load() (Trishuli scenes, flood, impact, connectivity),
- ml/runs/kurosiwo_experiment/metrics.json (training/validation) and
  ml/runs/kurosiwo_test_eval/metrics.json (held-out test, per event, confusion matrices),
- a few values documented only in docs/*.md, extracted by pattern so they cannot drift (doc_value).
Reuses situation_report for loading, the infrastructure map, attribution, wording and layout checks.
Deterministic for the same inputs.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

from . import situation_report as sr

PAGE = (8.27, 11.69)
L, R, TOP, BOT = 0.08, 0.92, 0.95, 0.06
PT = 1 / 72 / PAGE[1]  # one point as a fraction of page height
CHAR_W = 0.56          # average DejaVu Sans glyph width in em (wrapping estimate; verified by layout check)
NAVY, INK, MUTED = "#12355b", "#1f2933", "#555555"
MAX_PAGES = 6
TITLE = "VantageQ — Satellite-Powered Flood Intelligence for Disaster Response"


# --------------------------------------------------------------------------- data


def doc_value(root: Path, doc: str, pattern: str) -> str:
    """First regex group of `pattern` in docs/<doc>; fails if the documented value is missing."""
    text = (root / "docs" / doc).read_text(encoding="utf-8")
    m = re.search(pattern, text)
    if not m:
        raise sr.ReportDataError(f"value not found in docs/{doc}: {pattern}")
    return m.group(1)


def event_regions(root: Path) -> dict[str, str]:
    """Event id -> region from the test-event table in docs/TEST_EVALUATION.md."""
    text = (root / "docs" / "TEST_EVALUATION.md").read_text(encoding="utf-8")
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"^\| (\d+) \| ([^|]+) \| \d+ \|", text, re.M)}


def load_all(root: Path) -> dict:
    d = sr.load(root)  # validates outputs, manifest hashes and attribution
    paths = {"train": root / "ml/runs/kurosiwo_experiment/metrics.json",
             "test": root / "ml/runs/kurosiwo_test_eval/metrics.json",
             "class_map": root / "data/processed/trishuli/trishuli_20260828_class.tif",
             "dashboard": root / "docs/figures/dashboard_zoom_to_impacts.jpg",
             **{role: root / rel for role, rel in d.inference["inputs"].items()}}  # post/pre1/pre2 sigma0
    missing = [str(p.relative_to(root)) for p in paths.values() if not p.exists()]
    if missing:
        raise sr.ReportDataError(f"Model/experiment outputs not found. Missing: {', '.join(missing)}")
    return {"d": d, "train": json.loads(paths["train"].read_text()), "test": json.loads(paths["test"].read_text()),
            "paths": paths, "regions": event_regions(root), "root": root}


def f3(v: float) -> str:
    return f"{v:.3f}"


def build_content(a: dict) -> dict:
    """All report text. Every number is formatted from loaded data or extracted from docs."""
    d, tr, te, root = a["d"], a["train"], a["test"], a["root"]
    s, c, inf = d.infrastructure, d.connectivity, d.inference
    sc = inf["scenes"]
    n_tr, n_va, n_te = (sum(x.values()) for x in (tr["config"]["train_events"], tr["config"]["val_events"],
                                                    te["test_events"]))
    e_tr, e_va, e_te = (len(x) for x in (tr["config"]["train_events"], tr["config"]["val_events"], te["test_events"]))
    vf, tf = tr["val"]["flood"], te["test"]["flood"]
    cm = te["test"]["confusion_matrix"]
    flood_gt = sum(cm[2])
    to_perm = cm[2][1] / flood_gt
    per_event = []
    for ev, m in sorted(te["test_per_event"].items(), key=lambda kv: -kv[1]["flood"]["iou"]):
        row = m["confusion_matrix"][2]
        per_event.append([ev, a["regions"].get(ev, "?"), str(te["test_events"][ev]), f3(m["flood"]["iou"]),
                          f3(m["flood"]["precision"]), f3(m["flood"]["recall"]), f"{100 * row[1] / sum(row):.1f} %"])
    largest = max(te["test_per_event"].items(), key=lambda kv: sum(kv[1]["confusion_matrix"][2]))
    cut = [x for x in c["settlements"] if x["status"] == "potentially_cut_off"]
    paper_f1 = doc_value(root, "TRAINING_EXPERIMENT.md", r"Kuro Siwo reports flood F1 ([\d.]+) %")
    dark6 = doc_value(root, "TRISHULI_INFERENCE.md", r"Pixels > 6 dB darker than the pre-event mean: ([\d.]+) %")
    dark6_ref = doc_value(root, "TRISHULI_INFERENCE.md", r"\(vs ([\d.]+) % between the two\s+pre-event dates\)")
    dark6_hit = doc_value(root, "TRISHULI_INFERENCE.md", r"([\d.]+) % of them are predicted flood")
    dark3_hit = doc_value(root, "TRISHULI_INFERENCE.md", r"only ([\d.]+) % are predicted flood")
    day = lambda k: sc[k]["datetime"][:10]  # noqa: E731
    km2 = f"{s['flood_area_km2_utm']:.3f} km²"
    pct = f"{inf['class_pct']['flood']:.2f} %"
    road_m = f"{round(s['roads']['affected_length_km'] * 1000):,} m"
    cut_names = " and ".join(x["name"] for x in cut)

    return {
        "title": TITLE,
        "subtitle": ["IIT Mandi Multimodal AI Hackathon 2026", "Track B — Mapping Flood Damage from Space",
                     "Technical report · research prototype"],
        "summary": [
            "End-to-end prototype: Sentinel-1 scene selection → 6-channel U-Net flood segmentation → historical-OSM "
            "infrastructure overlay → simplified road connectivity → dashboard and one-page situation report.",
            f"Model (Kuro Siwo, event-separated): held-out test flood IoU {f3(tf['iou'])}, F1 {f3(tf['f1'])}, "
            f"precision {f3(tf['precision'])}, recall {f3(tf['recall'])} on {n_te} samples from {e_te} unseen events.",
            f"Trishuli, August 2026 (unvalidated inference): predicted flood extent {km2} ({pct} of analysed AOI "
            f"pixels); {s['roads']['segments_affected']} road ways and {s['buildings']['affected']} buildings potentially "
            f"affected, {s['bridges']['affected']} bridges intersecting predicted flood, {len(cut)} of "
            f"{s['settlements']['analyzed']} settlements potentially cut off in a simplified road graph.",
        ],
        "problem": [
            "During a flood, responders need to know where water is, which roads, bridges and buildings may be "
            "affected, and which communities may have lost road access. In Himalayan valleys such as the Trishuli, "
            "floods coincide with monsoon cloud, so optical imagery is often unusable for weeks.",
            "Sentinel-1 synthetic aperture radar (SAR) images through cloud, day and night, at about 10 m, and calm "
            "open water appears dark in its backscatter. Comparing a post-event image with pre-event images from the "
            "same orbit geometry highlights new water. VantageQ links that signal to pre-event OpenStreetMap "
            "infrastructure and a simple connectivity analysis, and presents results as indicators, not confirmed damage.",
        ],
        "arch_steps": ["AOI + event date", "Sentinel-1 scene\nselection (same track)",
                       "Kuro Siwo-compatible\npreprocessing", "6-channel U-Net\nflood segmentation",
                       "Predicted flood\nextent", "Historical OSM\ninfrastructure overlay",
                       "Simplified road\nconnectivity", "Dashboard +\nsituation report"],
        "arch_caption": ("Figure 1. VantageQ workflow. Inputs: Sentinel-1 GRD via the Copernicus Data Space "
                         "Ecosystem (orthorectified with Copernicus DEM inside Sentinel Hub), a U-Net trained on "
                         "Kuro Siwo, and a pre-event OSM snapshot. Outputs are files read by the static dashboard "
                         "and the report generator."),
        "data_rows": [
            ["Sentinel-1 GRD (IW, VV+VH)", "Model input", f"post {day('post')}, pre1 {day('pre1')}, pre2 {day('pre2')}; "
             f"relative orbit {sc['post']['relative_orbit']} {sc['post']['orbit_state']}"],
            ["Kuro Siwo (labelled GRD)", "Training/evaluation", f"{n_tr} train / {n_va} val / {n_te} test samples; "
             "official event splits"],
            ["OpenStreetMap (historical)", "Infrastructure", f"Overpass attic snapshot {s['osm_snapshot']} (pre-event)"],
            ["Copernicus DEM GLO-30", "Orthorectification", "used inside Sentinel Hub (COPERNICUS_30); not distributed"],
        ],
        "compliance": ("No Copernicus EMS product (including EMSR927), UNOSAT or other published damage/reference map, "
                       "and no post-event OSM edit, was used as a production input, for training, thresholds or model "
                       "selection. EMSR927 is reserved for a separate, not-yet-built evaluation-only comparison."),
        "model": [
            "U-Net with a ResNet-18 encoder (ImageNet initialisation), the official Kuro Siwo baseline configuration.",
            "Input: 6 channels [post VV, post VH, pre1 VV, pre1 VH, pre2 VV, pre2 VH], linear σ⁰ clamped to [0, 0.15] "
            "and normalised with fixed Kuro Siwo statistics (VV (x − 0.0953)/0.0427, VH (x − 0.0264)/0.0215); "
            "one shared preprocessing function for training and Trishuli inference.",
            "Output: 3 classes (no water, permanent water, flood); label 3 is excluded from loss and metrics "
            "(ignore_index=3).",
            f"Training: {n_tr} samples from {e_tr} official train events; Adam (lr 1e-3), cosine schedule, flips, "
            f"20 epochs; checkpoint chosen on validation loss ({n_va} samples, {e_va} validation events, best epoch "
            f"{tr['best_epoch']}). The checkpoint was then frozen.",
        ],
        "metric_rows": [
            ["Validation (model selection)", f"{n_va} / {e_va}", f3(vf["iou"]), f3(vf["f1"]), f3(vf["precision"]),
             f3(vf["recall"])],
            ["Held-out test (evaluated once)", f"{n_te} / {e_te}", f3(tf["iou"]), f3(tf["f1"]), f3(tf["precision"]),
             f3(tf["recall"])],
        ],
        "per_event": per_event,
        "model_notes": [
            "These are Kuro Siwo dataset results on unseen events, not Trishuli accuracy. Kuro Siwo contains no "
            "steep Himalayan terrain.",
            f"Flood precision is high but recall is low. {100 * to_perm:.1f} % of test flood pixels were predicted "
            f"as permanent water, mostly in event {largest[0]}, {a['regions'].get(largest[0], '')}, which holds "
            f"{100 * sum(largest[1]['confusion_matrix'][2]) / flood_gt:.0f} % of test flood pixels.",
            f"Validation scores are optimistic (same set used for selection). Not comparable with the Kuro Siwo paper "
            f"(flood F1 {paper_f1} % with full data, ResNet-50).",
        ],
        "trishuli": [
            f"Event: Trishuli Valley flood, {sr._date(d.manifest['event_date'] + 'T00:00:00Z')}; sub-AOI "
            f"{inf['sub_aoi_wgs84'][0]}–{inf['sub_aoi_wgs84'][2]}° E, {inf['sub_aoi_wgs84'][1]}–"
            f"{inf['sub_aoi_wgs84'][3]}° N near Betrawati.",
            f"Sentinel-1 post {day('post')}, pre1 {day('pre1')}, pre2 {day('pre2')}, all relative orbit "
            f"{sc['post']['relative_orbit']} {sc['post']['orbit_state']} (same viewing geometry).",
            f"All dates on one shared EPSG:{inf['grid']['epsg']} grid ({inf['grid']['width']} × {inf['grid']['height']} "
            f"pixels of 10 map units), linear σ⁰ (SIGMA0_ELLIPSOID), Lee 7×7, as in Kuro Siwo; tiled inference "
            f"(224 px, overlap averaging; tiling difference {inf['tiling_diagnostic_mean_abs_diff']:.4f}).",
            f"Result: predicted flood extent {km2}, {pct} of the analysed AOI pixels. This is an unvalidated model "
            f"prediction, not a measured flood area; no Trishuli ground truth was used.",
        ],
        "quicklook_caption": ("Figure 2. Trishuli sub-AOI: pre-event (pre1) and post-event VV backscatter, change "
                              "post − mean(pre1, pre2) (red = darker after), and the predicted classes on the post image "
                              "(red flood, blue permanent water). Model prediction, not validated."),
        "impact_rows": [
            ["Road ways potentially affected", str(s["roads"]["segments_affected"]),
             f"of {s['roads']['segments_total']}; {road_m} inside predicted flood"],
            ["Bridges intersecting predicted flood", str(s["bridges"]["affected"]), f"of {s['bridges']['total']} mapped"],
            ["Buildings potentially affected", str(s["buildings"]["affected"]), f"of {s['buildings']['total']:,} mapped"],
            ["Settlements potentially cut off", str(len(cut)), f"of {s['settlements']['analyzed']} analysed: {cut_names}"],
        ],
        "rules": [
            "Road way: ≥ 10 m of its length inside the predicted flood (prototype rule, not calibrated).",
            "Bridge: intersects the predicted flood. Overlap is not evidence of damage; bridges cross water by design.",
            "Building: > 0.01 m² of footprint inside the predicted flood.",
            f"Potentially cut off: {sr.CUT_OFF_DEFINITION[len('Potentially cut off = '):]} {sr.CUT_OFF_DISCLAIMER}",
        ],
        "settlement_rows": [[x["name"], sr.STATUS_LABELS[x["status"]],
                             f"{x['reachable_road_km_before']:.2f} → {x['reachable_road_km_after']:.2f} km"]
                            for x in sorted(c["settlements"], key=lambda x: (x["status"] != "potentially_cut_off", x["name"]))],
        "connectivity_notes": [
            f"Graph: vehicle roads in the AOI ({c['network']['blocked_edges']} edges blocked by predicted flood; "
            f"main network {c['network']['main_network_km_before']:.1f} → {c['network']['main_network_km_after']:.1f} km). "
            "Footpaths are excluded, and the graph is limited to the AOI.",
            f"{cut_names} lose the highway link to the main network inside the AOI. The Bhainse result is "
            "especially uncertain: it sits on the AOI edge and the highway continues beyond it.",
        ],
        "map_caption": ("Figure 3. Infrastructure and connectivity indicators from pre-event OSM (2026-08-25) and the "
                        "predicted flood. Indicators only; not confirmed damage or isolation."),
        "dashboard": [
            "Static Next.js + MapLibre dashboard (no backend or database): interactive map with predicted flood, "
            "potentially affected road sections, bridges intersecting predicted flood, potentially affected buildings "
            "and settlements; popups, KPI cards, a settlement panel, methodology and limitations, and a "
            "“Zoom to impacts” control.",
            "It reads a committed, deterministic demo snapshot (~100 KB) built from the pipeline outputs. Live "
            "OpenStreetMap tiles serve as basemap for orientation only; all analytical layers come from the snapshot.",
        ],
        "dashboard_caption": ("Figure 4. Dashboard after “Zoom to impacts”: impact layers on the map, KPI cards and the "
                              "settlement panel with the cut-off definition."),
        "discussion": [
            f"On held-out Kuro Siwo events the model finds flood reliably when it predicts it (precision "
            f"{f3(tf['precision'])}) but misses much of it (recall {f3(tf['recall'])}); performance varies strongly "
            f"by event (flood IoU {min(m['flood']['iou'] for m in te['test_per_event'].values()):.3f}–"
            f"{max(m['flood']['iou'] for m in te['test_per_event'].values()):.3f}).",
            "Terrain-driven false positives appeared on validation (dark sloped terrain), and flood/permanent-water "
            "confusion is the largest error mode on test.",
            f"At Trishuli, the predicted flood coincides with strong post-event darkening: pixels > 6 dB darker than "
            f"the pre-event mean cover {dark6} % of the AOI (vs {dark6_ref} % between the two pre-event dates), and "
            f"{dark6_hit} % of them are predicted flood. But only {dark3_hit} % of pixels > 3 dB darker are predicted "
            "flood, and much of a dark trail along the valley is missed.",
            "The Trishuli output is therefore an unvalidated prototype indicator. VantageQ does not claim to have "
            "solved Himalayan flood mapping.",
        ],
        "limitations": [
            "Trishuli flood map not validated against ground truth.",
            "Terrain effects (layover, shadow, dark slopes) can cause SAR false positives; no slope/shadow mask yet.",
            "Limited mountainous diversity in the training data (no Himalayan events in Kuro Siwo).",
            "Simplified connectivity: vehicle roads only, footpaths excluded, graph limited to the AOI.",
            "Bridge intersection with predicted flood is not bridge damage or impassability.",
            "Dashboard basemap depends on live OSM tiles (internet required).",
        ],
        "future": [
            "More labelled mountainous flood data for training and testing.",
            "DEM- and shadow-aware filtering of predictions.",
            "Broader OSM connectivity (footpaths, larger graph extent).",
            "Optional downstream flood-path tracing from the Copernicus DEM (planned bonus).",
            "Independent validation of the Trishuli result.",
        ],
        "conclusion": ("VantageQ is a working end-to-end prototype that turns Sentinel-1 radar imagery into a "
                       "predicted flood extent and connects it to infrastructure and road-connectivity indicators, "
                       "delivered through a dashboard and a one-page situation report generated from the same "
                       "structured outputs. Its model results on held-out Kuro Siwo events are modest and its Trishuli "
                       "output is unvalidated; the value lies in a transparent, reproducible chain from satellite data "
                       "to responder-oriented indicators, with every number traceable and every limitation stated."),
        "attribution": d.attribution,
        "disclosure": ("Developed with an AI coding assistant (Claude Code, Claude Opus 5.5). The authors directed the "
                       "work and are responsible for validating all code, results and claims."),
    }


# --------------------------------------------------------------------------- layout


class Flow:
    """Minimal top-to-bottom text flow over A4 matplotlib figures."""

    def __init__(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
        self.plt = plt
        self.figs = []
        self.new_page()

    def new_page(self):
        self.fig = self.plt.figure(figsize=PAGE)
        self.figs.append(self.fig)
        self.y = TOP

    def need(self, h: float):
        if self.y - h < BOT:
            self.new_page()

    @staticmethod
    def wrap(text: str, size: float, width_frac: float, bold: bool = False) -> str:
        chars = max(8, int(width_frac * PAGE[0] * 72 / (size * (CHAR_W * 1.12 if bold else CHAR_W))))
        return "\n".join(textwrap.fill(p, chars, break_long_words=False) for p in text.split("\n"))

    def text(self, s, size=8.6, x=L, width=None, after=0.006, keep=0.0, **kw):
        width = width if width is not None else R - x
        t = self.wrap(s, size, width, bold=kw.get("fontweight") == "bold")
        h = (t.count("\n") + 1) * size * 1.36 * PT
        self.need(h + keep)
        self.fig.text(x, self.y, t, fontsize=size, va="top", linespacing=1.36, color=kw.pop("color", INK), **kw)
        self.y -= h + after

    def heading(self, s):
        self.need(0.11)
        self.y -= 0.006
        self.text(s, size=11.5, fontweight="bold", color=NAVY, after=0.004)
        from matplotlib.patches import Rectangle

        self.fig.add_artist(Rectangle((L, self.y), R - L, 0.0012, color=NAVY, lw=0, transform=self.fig.transFigure))
        self.y -= 0.009

    def bullets(self, items, size=8.6):
        for it in items:
            t = self.wrap(it, size, R - L - 0.02)
            self.need((t.count("\n") + 1) * size * 1.36 * PT)
            self.fig.text(L + 0.004, self.y, "•", fontsize=size, va="top", color=INK)
            self.text(it, size=size, x=L + 0.02, after=0.004)

    def table(self, header, rows, cols, size=7.6):
        """cols: column starts as fractions of the text width; cells wrap to their column."""
        from matplotlib.patches import Rectangle

        xs = [L + c * (R - L) for c in cols]
        widths = [b - a - 0.012 for a, b in zip(xs, xs[1:] + [R + 0.012])]
        cells = [[self.wrap(str(v), size, w, bold=(i == 0)) for v, w in zip(row, widths)]
                 for i, row in enumerate([header] + rows)]
        heights = [(max(c.count("\n") for c in row) + 1) * size * 1.3 * PT + 0.007 for row in cells]
        self.need(sum(heights) + 0.004)
        for i, (row, rh) in enumerate(zip(cells, heights)):
            if i % 2 == 0 and i:
                self.fig.add_artist(Rectangle((L, self.y - rh + 0.002), R - L, rh, color="#f2f4f7", lw=0,
                                              transform=self.fig.transFigure))
            for x, cell in zip(xs, row):
                self.fig.text(x, self.y - 0.0035, cell, fontsize=size, va="top", color=INK, linespacing=1.3,
                              fontweight="bold" if i == 0 else "normal")
            self.y -= rh
            if i == 0:
                self.fig.add_artist(Rectangle((L, self.y + 0.002), R - L, 0.0008, color=MUTED, lw=0,
                                              transform=self.fig.transFigure))
        self.y -= 0.008

    def figure(self, h, draw, caption):
        cap = self.wrap(caption, 7.4, R - L)
        ch = (cap.count("\n") + 1) * 7.4 * 1.36 * PT
        self.need(h + ch + 0.012)
        ax = self.fig.add_axes((L, self.y - h, R - L, h))
        draw(ax)
        self.y -= h + 0.006
        self.text(caption, size=7.4, color=MUTED, style="italic", after=0.01)


def draw_architecture(ax, steps):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, h = 0.205, 0.32
    xs = [0.0, 0.265, 0.53, 0.795]
    pos = [(xs[i], 0.6) for i in range(4)] + [(xs[3 - i], 0.08) for i in range(4)]
    fill = ["#e8eef6"] * 3 + [NAVY] + ["#dceefa"] + ["#fdf0e6"] * 2 + ["#e9f5ec"]
    for i, ((x, y), label) in enumerate(zip(pos, steps)):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.03", fc=fill[i],
                                    ec=NAVY, lw=0.8))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.6,
                color="white" if fill[i] == NAVY else INK, fontweight="bold" if i in (3, 4) else "normal")
    for i in range(7):
        (x0, y0), (x1, y1) = pos[i], pos[i + 1]
        if y0 == y1:
            a, b = ((x0 + w, y0 + h / 2), (x1, y1 + h / 2)) if x1 > x0 else ((x0, y0 + h / 2), (x1 + w, y1 + h / 2))
        else:
            a, b = (x0 + w / 2, y0), (x1 + w / 2, y1 + h)
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=NAVY, lw=1))


def draw_trishuli(paths: dict):
    """Four panels from the pipeline rasters: pre1 VV, post VV, change, predicted classes."""
    def draw(ax):
        import numpy as np
        import rasterio
        from matplotlib.colors import ListedColormap

        vv = {r: rasterio.open(paths[r]).read(1).astype("float64") for r in ("post", "pre1", "pre2")}
        cls = rasterio.open(paths["class_map"]).read(1)
        db = lambda x: 10 * np.log10(np.clip(x, 1e-4, None))  # noqa: E731
        change = db(vv["post"]) - db((vv["pre1"] + vv["pre2"]) / 2)
        overlay = np.ma.masked_where(~np.isin(cls, [1, 2]), cls)
        ax.axis("off")
        panels = [("Pre-event VV (pre1), dB", db(vv["pre1"]), {"cmap": "gray", "vmin": -20, "vmax": 5}),
                  ("Post-event VV, dB", db(vv["post"]), {"cmap": "gray", "vmin": -20, "vmax": 5}),
                  ("Change post − pre, dB", change, {"cmap": "RdBu", "vmin": -6, "vmax": 6}),
                  ("Predicted classes", db(vv["post"]), {"cmap": "gray", "vmin": -20, "vmax": 5})]
        for i, (title, img, kw) in enumerate(panels):
            sub = ax.inset_axes((i * 0.25 + 0.004, 0.0, 0.242, 0.9))
            sub.imshow(img, interpolation="nearest", **kw)
            if i == 3:
                sub.imshow(overlay, cmap=ListedColormap(["#2c7fb8", "#d7301f"]), vmin=1, vmax=2, interpolation="nearest")
            sub.set_title(title, fontsize=7, pad=2)
            sub.set_xticks([])
            sub.set_yticks([])
    return draw


def draw_image(path: Path):
    def draw(ax):
        import matplotlib.image as mpimg

        ax.imshow(mpimg.imread(path))
        ax.axis("off")
    return draw


def render(a: dict, content: dict) -> list:
    k = content
    f = Flow()
    fig = f.fig
    from matplotlib.patches import Rectangle

    fig.add_artist(Rectangle((0, 0.885), 1, 0.115, color=NAVY, lw=0, transform=fig.transFigure))
    fig.text(L, 0.975, "VantageQ", fontsize=24, fontweight="bold", color="white", va="top")
    fig.text(L, 0.938, "Satellite-Powered Flood Intelligence for Disaster Response", fontsize=11.5, color="white", va="top")
    fig.text(L, 0.914, k["subtitle"][0] + " · " + k["subtitle"][1], fontsize=8.4, color="#d6e2f0", va="top")
    fig.text(R, 0.975, k["subtitle"][2], fontsize=7.6, color="#d6e2f0", va="top", ha="right")
    f.y = 0.868
    f.text("Summary", size=10, fontweight="bold", color=NAVY, after=0.003)
    f.bullets(k["summary"])

    f.heading("1  Problem and motivation")
    for p in k["problem"]:
        f.text(p)
    f.heading("2  System overview")
    f.figure(0.15, lambda ax: draw_architecture(ax, k["arch_steps"]), k["arch_caption"])

    f.heading("3  Data and compliance")
    f.table(["Source", "Role", "Detail"], k["data_rows"], [0.0, 0.3, 0.5])
    f.text(k["compliance"], fontweight="bold", size=8.4)
    att_h = sum((f.wrap(x, 7.2, R - L).count("\n") + 1) * 7.2 * 1.36 * PT + 0.002 for x in k["attribution"])
    f.need(att_h + 0.03)
    f.text("Attribution (docs/DATA_SOURCES.md §7):", size=8, fontweight="bold", after=0.002)
    for line in k["attribution"]:
        f.text(line, size=7.2, color="#333", after=0.002)
    f.y -= 0.004

    f.heading("4  AI model and Kuro Siwo evaluation")
    f.bullets(k["model"])
    f.table(["Kuro Siwo split", "Samples / events", "Flood IoU", "Flood F1", "Precision", "Recall"],
            k["metric_rows"], [0.0, 0.33, 0.5, 0.62, 0.74, 0.86])
    f.text("Held-out test, flood by event (sorted by IoU):", size=8, fontweight="bold", after=0.002)
    f.table(["Event", "Region", "Samples", "Flood IoU", "Precision", "Recall", "Flood GT → perm. water"],
            k["per_event"], [0.0, 0.11, 0.36, 0.47, 0.58, 0.69, 0.79], size=7.2)
    f.bullets(k["model_notes"])

    f.heading("5  Trishuli case study (unvalidated inference)")
    f.bullets(k["trishuli"])
    f.figure(0.15, draw_trishuli(a["paths"]), k["quicklook_caption"])

    f.heading("6  Infrastructure impact and connectivity")
    f.table(["Indicator", "Count", "Context"], k["impact_rows"], [0.0, 0.42, 0.52])
    f.bullets(k["rules"])
    f.table(["Settlement", "Status (simplified road graph)", "Reachable road before → after"],
            k["settlement_rows"], [0.0, 0.2, 0.68])
    f.bullets(k["connectivity_notes"])
    f.figure(0.275, lambda ax: sr.draw_map(ax, a["d"]), k["map_caption"])

    f.heading("7  Dashboard")
    f.bullets(k["dashboard"])
    f.figure(0.29, draw_image(a["paths"]["dashboard"]), k["dashboard_caption"])

    f.heading("8  Results and discussion")
    f.bullets(k["discussion"])
    f.heading("9  Limitations and future work")
    f.text("Limitations", size=8.6, fontweight="bold", after=0.002)
    f.bullets(k["limitations"])
    f.text("Future work", size=8.6, fontweight="bold", after=0.002)
    f.bullets(k["future"])
    f.heading("10  Conclusion")
    f.text(k["conclusion"])
    f.y -= 0.004
    f.text(k["disclosure"], size=7, color=MUTED, style="italic")

    n = len(f.figs)
    for i, pg in enumerate(f.figs, start=1):
        pg.text(L, 0.028, "VantageQ · technical report · IIT Mandi Multimodal AI Hackathon 2026", fontsize=6.8,
                color=MUTED, va="bottom")
        pg.text(R, 0.028, f"{i} / {n}", fontsize=6.8, color=MUTED, ha="right", va="bottom")
    return f.figs


def generate(root: Path, out_dir: Path) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    a = load_all(root)
    content = build_content(a)
    sr.check_wording(content)
    figs = render(a, content)
    try:
        problems = [f"page {i}: {p}" for i, fig in enumerate(figs, 1) for p in sr.layout_problems(fig)]
        if problems:
            raise sr.ReportDataError("layout problems: " + "; ".join(problems))
        if len(figs) > MAX_PAGES:
            raise sr.ReportDataError(f"report has {len(figs)} pages, maximum is {MAX_PAGES}")
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf = out_dir / "VantageQ_Final_Hackathon_Report.pdf"
        with PdfPages(pdf, metadata={"Title": TITLE, "Creator": "VantageQ scripts/generate_final_report.py",
                                     "Producer": None, "CreationDate": None, "ModDate": None}) as pp:
            for fig in figs:
                pp.savefig(fig)
    finally:
        for fig in figs:
            plt.close(fig)
    return {"pdf": pdf, "pages": sr.pdf_page_count(pdf), "content": content}
