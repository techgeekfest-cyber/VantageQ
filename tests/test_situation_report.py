"""Situation report: values from JSON, wording, missing/stale data, one page, determinism.

Uses a synthetic output tree with distinctive numbers (the real outputs are git-ignored); one test
also checks the real outputs when they exist.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from vantageq.reporting import situation_report as sr  # noqa: E402

AOI = [85.17, 27.95, 85.22, 27.99]


def fc(features):
    return {"type": "FeatureCollection", "features": features}


def line(coords, **props):
    return {"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords}, "properties": props}


def write_fixture(root: Path, cut_off=("Alpha", "Beta"), flood_km2=0.4321) -> Path:
    proc, infra = root / "data/processed/trishuli", root / "data/processed/trishuli/infrastructure"
    infra.mkdir(parents=True)
    scenes = {r: {"datetime": d, "relative_orbit": 85, "orbit_state": "ascending", "id": f"S1_{r}"}
              for r, d in (("post", "2026-08-28T12:21:41Z"), ("pre1", "2026-08-16T12:21:41Z"),
                           ("pre2", "2026-08-04T12:21:40Z"))}
    (proc / "trishuli_20260828_summary.json").write_text(json.dumps({
        "scenes": scenes, "sub_aoi_wgs84": AOI, "class_pct": {"no water": 98.0, "permanent water": 0.77, "flood": 1.23},
        "checkpoint_sha256_16": "abcdef0123456789"}))
    (infra / "infrastructure_summary.json").write_text(json.dumps({
        "osm_snapshot": "2026-08-25T00:00:00Z", "flood_area_km2_utm": flood_km2,
        "roads": {"segments_total": 211, "segments_affected": 7, "length_km_total": 98.7, "affected_length_km": 1.234},
        "bridges": {"total": 19, "affected": 5}, "buildings": {"total": 5432, "affected": 87},
        "settlements": {"analyzed": 3, "potentially_cut_off": len(cut_off)}}))
    sett = [{"osm_id": i, "name": nm, "place": "village", "status": "potentially_cut_off", "snap_distance_m": 12.0,
             "reachable_road_km_before": 44.4, "reachable_road_km_after": 1.1} for i, nm in enumerate(cut_off)]
    sett.append({"osm_id": 99, "name": "Gamma", "place": "hamlet", "status": "connected", "snap_distance_m": 50.0,
                 "reachable_road_km_before": 44.4, "reachable_road_km_after": 44.4})
    (infra / "connectivity_summary.json").write_text(json.dumps({
        "settlements": sett, "network": {"blocked_edges": 4, "main_network_km_before": 44.4, "main_network_km_after": 40.0}}))
    poly = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
        [85.18, 27.96], [85.185, 27.96], [85.185, 27.965], [85.18, 27.965], [85.18, 27.96]]]}}
    (infra / "flood_extent.geojson").write_text(json.dumps(fc([poly])))
    (infra / "roads.geojson").write_text(json.dumps(fc([line([[85.17, 27.96], [85.22, 27.97]], highway="primary")])))
    (infra / "affected_roads.geojson").write_text(json.dumps(fc([line([[85.18, 27.96], [85.19, 27.97]])])))
    (infra / "affected_bridges.geojson").write_text(json.dumps(fc([line([[85.181, 27.961], [85.182, 27.962]])])))
    (infra / "affected_buildings.geojson").write_text(json.dumps(fc([poly])))
    pts = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [85.19 + i * 0.01, 27.97]},
            "properties": {"name": s["name"], "status": s["status"], "osm_id": s["osm_id"]}} for i, s in enumerate(sett)]
    (infra / "settlements.geojson").write_text(json.dumps(fc(pts)))
    (root / "docs").mkdir()
    shutil.copy(ROOT / "docs" / "DATA_SOURCES.md", root / "docs" / "DATA_SOURCES.md")
    man = root / "frontend/public/demo/trishuli"
    man.mkdir(parents=True)
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(proc.rglob("*")) if p.is_file()}
    (man / "manifest.json").write_text(json.dumps({"event_date": "2026-08-26", "aoi_name": "Test valley",
                                                   "sources_sha256": hashes}))
    return root


@pytest.fixture
def fixture_root(tmp_path):
    return write_fixture(tmp_path / "repo")


def test_numbers_come_from_json(fixture_root, tmp_path):
    res = sr.generate(fixture_root, tmp_path / "out")
    c = res["content"]
    assert c["flood"]["value"] == "0.432 km²"
    assert "1.23 % of the analysed AOI pixels" in c["flood"]["detail"]
    kpis = {label: (value, detail) for value, label, detail in c["kpis"]}
    assert kpis["Road ways potentially affected"] == ("7", "of 211 mapped road ways")
    assert kpis["Road length potentially affected"][0] == "1,234 m"
    assert kpis["Bridges intersecting predicted flood"] == ("5", "of 19 mapped bridges")
    assert kpis["Buildings potentially affected"] == ("87", "of 5,432 mapped buildings")
    assert "4 road-graph edges blocked" in c["network"] and "44.4 km → 40.0 km" in c["network"]
    html = res["html"].read_text()
    for s in ("0.432 km²", "1,234 m", "5,432", "87", "28 August 2026", "16 August 2026", "04 August 2026",
              "2026-08-25T00:00:00Z", "Overpass historical/attic snapshot"):
        assert s in html, s


def test_settlements_and_statuses_come_from_connectivity_json(tmp_path):
    root = write_fixture(tmp_path / "r", cut_off=("Alpha", "Beta"))
    c = sr.build_content(sr.load(root))
    assert c["connectivity_summary"] == "3 settlements analysed; 2 potentially cut off: Alpha, Beta."
    rows = {r["name"]: r for r in c["settlements"]}
    assert rows["Alpha"]["status_label"] == sr.STATUS_LABELS["potentially_cut_off"]
    assert rows["Gamma"]["status_label"] == sr.STATUS_LABELS["connected"]
    assert "44.40 km → 1.10 km" in rows["Beta"]["access"]
    assert [r["name"] for r in c["settlements"]][:2] == ["Alpha", "Beta"]  # cut-off first
    assert c["definition"] == ("Potentially cut off = disconnected in the simplified road graph after removing "
                               "flood-affected edges.")
    assert c["disclaimer"] == "This is a graph-based indicator, not confirmed real-world isolation."


def test_no_forbidden_wording_and_guard_works(fixture_root):
    c = sr.build_content(sr.load(fixture_root))
    text = sr.all_text(c).lower()
    assert "damaged" not in text and "destroyed" not in text
    assert "predicted flood extent" in text and "potentially affected" in text
    sr.check_wording(c)
    with pytest.raises(sr.ReportDataError):
        sr.check_wording(c | {"extra": "3 damaged bridges"})


def test_attribution_is_parsed_from_data_sources(fixture_root):
    att = sr.load(fixture_root).attribution
    joined = " ".join(att)
    assert "Contains modified Copernicus Sentinel data 2026" in joined
    assert "© OpenStreetMap contributors" in joined and "Bountos" in joined
    assert "[VERIFY" not in joined and "**" not in joined


def test_missing_data_fails_clearly(fixture_root, tmp_path):
    (fixture_root / "data/processed/trishuli/infrastructure/connectivity_summary.json").unlink()
    with pytest.raises(sr.ReportDataError, match="Trishuli analysis outputs not found. Run the analysis pipeline first."):
        sr.generate(fixture_root, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_stale_dashboard_snapshot_is_rejected(fixture_root):
    p = fixture_root / "data/processed/trishuli/infrastructure/infrastructure_summary.json"
    d = json.loads(p.read_text()) | {"flood_area_km2_utm": 9.9}
    p.write_text(json.dumps(d))
    with pytest.raises(sr.ReportDataError, match="build_demo_snapshot"):
        sr.load(fixture_root)


def test_missing_required_number_is_rejected(fixture_root):
    p = fixture_root / "data/processed/trishuli/infrastructure/infrastructure_summary.json"
    d = json.loads(p.read_text())
    del d["buildings"]["affected"]
    p.write_text(json.dumps(d))
    man = fixture_root / "frontend/public/demo/trishuli/manifest.json"  # keep the hash check satisfied
    m = json.loads(man.read_text())
    m["sources_sha256"][str(p.relative_to(fixture_root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    man.write_text(json.dumps(m))
    with pytest.raises(sr.ReportDataError, match="infrastructure.buildings.affected"):
        sr.load(fixture_root)


def test_one_page_clean_layout_and_deterministic(fixture_root, tmp_path):
    a = sr.generate(fixture_root, tmp_path / "a")
    b = sr.generate(fixture_root, tmp_path / "b")
    assert a["pages"] == 1
    assert a["pdf"].read_bytes() == b["pdf"].read_bytes()
    assert a["html"].read_bytes() == b["html"].read_bytes()
    import matplotlib.pyplot as plt

    d = sr.load(fixture_root)
    fig = sr.render_figure(d, sr.build_content(d))
    assert sr.layout_problems(fig) == []
    plt.close(fig)


@pytest.mark.skipif(not (ROOT / "data/processed/trishuli/infrastructure/infrastructure_summary.json").exists(),
                    reason="real Trishuli outputs not present (git-ignored)")
def test_real_outputs_produce_one_page_report_with_json_values(tmp_path):
    res = sr.generate(ROOT, tmp_path)
    s = json.loads((ROOT / "data/processed/trishuli/infrastructure/infrastructure_summary.json").read_text())
    c = res["content"]
    assert res["pages"] == 1
    assert c["flood"]["value"] == f"{s['flood_area_km2_utm']:,.3f} km²"
    assert {v for v, *_ in c["kpis"]} >= {str(s["roads"]["segments_affected"]), str(s["bridges"]["affected"]),
                                          str(s["buildings"]["affected"])}


DEM_NOTICE = ("produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
              "2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved")


def test_copernicus_dem_attribution_uses_verified_article_6b_notice(fixture_root):
    att = sr.load(fixture_root).attribution
    dem = next(a for a in att if a.startswith("Copernicus DEM"))
    assert DEM_NOTICE in dem and "https://doi.org/10.5270/ESA-c5d3d65" in dem


def test_unverified_attribution_is_refused(fixture_root):
    md = fixture_root / "docs" / "DATA_SOURCES.md"
    md.write_text(md.read_text().replace("Contains modified Copernicus Sentinel data 2026",
                                         "Contains modified Copernicus Sentinel data 2026 **[VERIFY wording]**"))
    with pytest.raises(sr.ReportDataError, match="unverified attribution"):
        sr.load(fixture_root)


def test_confirmed_isolation_only_allowed_negated_and_unvalidated_stated(fixture_root):
    c = sr.build_content(sr.load(fixture_root))
    sr.check_wording(c)  # contains only negated forms
    text = sr.all_text(c)
    assert "not confirmed real-world isolation" in text and "unvalidated" in text
    assert "Trishuli flood prediction is unvalidated against ground truth in this project." in c["limitations"]
    for phrase in ("predicted flood extent", "potentially affected", "bridges intersecting predicted flood",
                   "potentially cut off"):
        assert phrase in text.lower(), phrase
    with pytest.raises(sr.ReportDataError):
        sr.check_wording(c | {"extra": "Betrawati: confirmed isolation"})
