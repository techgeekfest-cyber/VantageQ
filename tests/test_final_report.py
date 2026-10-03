"""Final technical report: traceable numbers, wording, page limit, layout and determinism."""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from vantageq.reporting import final_report as fr  # noqa: E402
from vantageq.reporting import situation_report as sr  # noqa: E402

HAVE_OUTPUTS = all((ROOT / p).exists() for p in (
    "ml/runs/kurosiwo_experiment/metrics.json", "ml/runs/kurosiwo_test_eval/metrics.json",
    "data/processed/trishuli/infrastructure/infrastructure_summary.json",
    "data/processed/trishuli/trishuli_20260828_class.tif"))
needs_outputs = pytest.mark.skipif(not HAVE_OUTPUTS, reason="pipeline/experiment outputs not present (git-ignored)")


def test_doc_values_are_extracted_not_typed_and_fail_when_missing(tmp_path):
    assert fr.doc_value(ROOT, "TRISHULI_INFERENCE.md", r"([\d.]+) % of them are predicted flood") == "64.5"
    assert fr.doc_value(ROOT, "TRAINING_EXPERIMENT.md", r"Kuro Siwo reports flood F1 ([\d.]+) %") == "80.1"
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "X.md").write_text("nothing here")
    with pytest.raises(sr.ReportDataError, match="value not found"):
        fr.doc_value(tmp_path, "X.md", r"(\d+) % of them")


def test_test_event_regions_come_from_test_evaluation_doc():
    regions = fr.event_regions(ROOT)
    assert regions["562"].startswith("Australia") and regions["277"] == "Greece" and len(regions) == 6


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    if not HAVE_OUTPUTS:
        pytest.skip("pipeline/experiment outputs not present (git-ignored)")
    out = tmp_path_factory.mktemp("final")
    return fr.generate(ROOT, out)


@needs_outputs
def test_page_limit_and_determinism(report, tmp_path):
    assert 1 <= report["pages"] <= fr.MAX_PAGES
    again = fr.generate(ROOT, tmp_path)
    assert again["pdf"].read_bytes() == report["pdf"].read_bytes()


@needs_outputs
def test_model_numbers_match_metrics_json(report):
    c = report["content"]
    tr = json.loads((ROOT / "ml/runs/kurosiwo_experiment/metrics.json").read_text())
    te = json.loads((ROOT / "ml/runs/kurosiwo_test_eval/metrics.json").read_text())
    test_row = c["metric_rows"][1]
    f = te["test"]["flood"]
    assert test_row[1] == f"{sum(te['test_events'].values())} / {len(te['test_events'])}"
    assert test_row[2:] == [f"{f[k]:.3f}" for k in ("iou", "f1", "precision", "recall")]
    assert c["metric_rows"][0][2] == f"{tr['val']['flood']['iou']:.3f}"
    model = " ".join(c["model"])
    assert f"{sum(tr['config']['train_events'].values())} samples from {len(tr['config']['train_events'])}" in model
    assert {r[0] for r in c["per_event"]} == set(te["test_per_event"])


@needs_outputs
def test_trishuli_and_impact_numbers_match_outputs(report):
    c = report["content"]
    s = json.loads((ROOT / "data/processed/trishuli/infrastructure/infrastructure_summary.json").read_text())
    con = json.loads((ROOT / "data/processed/trishuli/infrastructure/connectivity_summary.json").read_text())
    text = sr.all_text(c)
    assert f"{s['flood_area_km2_utm']:.3f} km²" in " ".join(c["trishuli"])
    rows = {r[0]: r for r in c["impact_rows"]}
    assert rows["Road ways potentially affected"][1] == str(s["roads"]["segments_affected"])
    assert rows["Bridges intersecting predicted flood"][1] == str(s["bridges"]["affected"])
    assert rows["Buildings potentially affected"][1] == str(s["buildings"]["affected"])
    assert f"{round(s['roads']['affected_length_km'] * 1000)} m inside predicted flood" in text
    assert [r[0] for r in c["settlement_rows"]] == sorted(
        [x["name"] for x in con["settlements"]], key=lambda n: (
            next(x["status"] for x in con["settlements"] if x["name"] == n) != "potentially_cut_off", n))


@needs_outputs
def test_trishuli_section_makes_no_accuracy_claim(report):
    tri = " ".join(report["content"]["trishuli"])
    assert "unvalidated" in tri and "not a measured flood area" in tri
    assert not re.search(r"\b(IoU|F1|precision|recall|accuracy)\b", tri, re.I)


@needs_outputs
def test_compliance_attribution_and_wording(report):
    c = report["content"]
    assert "EMSR927" in c["compliance"] and "No Copernicus EMS product" in c["compliance"]
    assert c["attribution"] == sr.parse_attribution((ROOT / "docs" / "DATA_SOURCES.md").read_text())
    sr.check_wording(c)
    assert "not Trishuli accuracy" in " ".join(c["model_notes"])
