from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.model_report import ModelPerformanceInvestigator


if __name__ == "__main__":
    investigator = ModelPerformanceInvestigator()
    for mode in ["returns", "cov"]:
        try:
            plots_dir, reports_dir = investigator.save_report(mode)
            print(f"[{mode}] plots: {plots_dir}")
            print(f"[{mode}] reports: {reports_dir}")
        except FileNotFoundError as exc:
            print(f"[{mode}] skipped: {exc}")
