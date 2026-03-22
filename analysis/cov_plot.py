from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.data_report import MarketDataInvestigator


if __name__ == "__main__":
    location = MarketDataInvestigator().save_report()
    print(f"Saved covariance and market-data plots to {location.plots_dir}")
