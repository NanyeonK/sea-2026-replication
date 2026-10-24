#!/usr/bin/env python3
"""Format Appendix A SHAP summaries for each XGBoost refit and their mean."""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401
from _paths import DERIVED_DATA_DIR, TABLES_DIR, require  # noqa: E402

REFIT_SENSITIVITY_FILE = DERIVED_DATA_DIR / "window_sensitivity.csv"
require(REFIT_SENSITIVITY_FILE)

DISPLAY_COLUMNS = {
    "District": "location_gu",
    "Contract Month": "contract_ym",
    "Build Year": "build_year",
    "Area (m²)": "area_numeric",
    "Total Parking": "total_parking",
    "Time Trend": "time_trend",
}


def main() -> int:
    s = pd.read_csv(REFIT_SENSITIVITY_FILE)
    out = pd.DataFrame({"Refit": s["refit"].astype(str)})
    for label, col in DISPLAY_COLUMNS.items():
        out[label] = s[col].round(4)
    out["Time share, mean %"] = s["district_time_pct_mean"].round(1)
    out["Time share, range %"] = (
        s["district_time_pct_min"].round(1).astype(str)
        + "-"
        + s["district_time_pct_max"].round(1).astype(str)
    )
    out.to_csv(TABLES_DIR / "TableA_window_sensitivity.csv", index=False)

    refits = s[s["refit"] != "mean"]
    mean_row = s[s["refit"] == "mean"].iloc[0]
    ok = True
    for col in DISPLAY_COLUMNS.values():
        got, exp = float(mean_row[col]), float(refits[col].mean())

        m = abs(got - exp) < 5e-6
        ok = ok and m
        print(f"mean({col:16s}) got={got:.6f} exp={exp:.6f}  {'PASS' if m else 'FAIL'}")
    lowest = set(refits.nsmallest(2, "contract_ym")["refit"].astype(str))
    m = lowest == {"60", "72"}
    ok = ok and m
    print(f"two lowest contract_ym refits = {sorted(lowest)}  {'PASS' if m else 'FAIL'}")

    print()
    print(out.to_string(index=False))
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
