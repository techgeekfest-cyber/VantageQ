"""Generate the final hackathon technical report (≤ 6 A4 pages) from repository outputs and docs.

See backend/vantageq/reporting/final_report.py. Output: docs/output/VantageQ_Final_Hackathon_Report.pdf.
Needs the (git-ignored) pipeline and experiment outputs; deterministic for the same inputs.

Usage:
  python scripts/generate_final_report.py
  python scripts/generate_final_report.py --preview-dir /tmp/pages   # also PNG page previews
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from vantageq.reporting import final_report as fr  # noqa: E402
from vantageq.reporting.situation_report import ReportDataError  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "output")
    ap.add_argument("--preview-dir", type=Path, help="optional directory for PNG page previews")
    args = ap.parse_args()
    try:
        res = fr.generate(REPO_ROOT, args.out_dir)
    except ReportDataError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    if args.preview_dir:
        import matplotlib.pyplot as plt

        a = fr.load_all(REPO_ROOT)
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        for i, fig in enumerate(fr.render(a, fr.build_content(a)), start=1):
            fig.savefig(args.preview_dir / f"page_{i}.png", dpi=80)
            plt.close(fig)
        print("previews:", args.preview_dir)
    pdf = res["pdf"]
    print(f"PDF: {pdf.relative_to(REPO_ROOT) if pdf.is_relative_to(REPO_ROOT) else pdf} ({res['pages']} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
