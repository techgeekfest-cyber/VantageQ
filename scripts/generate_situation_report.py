"""Generate the one-page Trishuli situation report (PDF + HTML) from the structured pipeline outputs.

All numbers come from the analysis JSON files; see backend/vantageq/reporting/situation_report.py.
Output: docs/output/VantageQ_Trishuli_Situation_Report.pdf (+ .html). Deterministic for the same inputs.

Usage:
  python scripts/generate_situation_report.py
  python scripts/generate_situation_report.py --preview preview.png   # also save a PNG preview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from vantageq.reporting import situation_report as sr  # noqa: E402

OUT_DIR = REPO_ROOT / "docs" / "output"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--preview", type=Path, help="optional PNG preview of the page")
    args = ap.parse_args()
    try:
        res = sr.generate(REPO_ROOT, args.out_dir)
    except sr.ReportDataError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    if args.preview:
        import matplotlib.pyplot as plt

        d = sr.load(REPO_ROOT)
        fig = sr.render_figure(d, sr.build_content(d))
        fig.savefig(args.preview, dpi=110)
        plt.close(fig)
        print("preview:", args.preview)
    print("data files used:", *res["files"].values(), sep="\n  ")
    print(f"PDF:  {res['pdf'].relative_to(REPO_ROOT) if res['pdf'].is_relative_to(REPO_ROOT) else res['pdf']} "
          f"({res['pages']} page)")
    print(f"HTML: {res['html'].relative_to(REPO_ROOT) if res['html'].is_relative_to(REPO_ROOT) else res['html']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
