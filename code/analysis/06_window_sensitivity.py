#!/usr/bin/env python3
"""Appendix table — SHAP attribution, refit by refit.

The SHAP figures average absolute attribution over all seven XGBoost refits. This table shows the
seven separately, so the averaging is inspectable rather than asserted.

What it shows: attribution to time is 0.18-0.24 in refits 0 through 48 and collapses to 0.035 and
0.014 in refits 60 and 72. Those two train on 2021 onward, after the Seoul price run-up ended, when
a monotone time index stops carrying predictive weight. The figures report the mean over all seven
because the forecasting advantage they decompose is measured over 84 evaluation windows, and each
refit produced twelve of them.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401  (starts the run log; see docs/ and the README's Run logs section)
from _paths import DERIVED, TABLES, require  # noqa: E402

SRC = DERIVED / "window_sensitivity.csv"
require(SRC)

# Column label -> source column. Labels match code/analysis/figure_config.py's VARIABLE_DISPLAY so
# the appendix and the figures name the same variable the same way.
COLS = {
    "District": "location_gu",
    "Contract Month": "contract_ym",
    "Build Year": "build_year",
    "Area (m²)": "area_numeric",
    "Total Parking": "total_parking",
    "Time Trend": "time_trend",
}


def main() -> int:
    s = pd.read_csv(SRC)
    out = pd.DataFrame({"Refit": s["refit"].astype(str)})
    for label, col in COLS.items():
        out[label] = s[col].round(4)
    out["Time share, mean %"] = s["district_time_pct_mean"].round(1)
    out["Time share, range %"] = (s["district_time_pct_min"].round(1).astype(str)
                                  + "-" + s["district_time_pct_max"].round(1).astype(str))
    out.to_csv(TABLES / "TableA_window_sensitivity.csv", index=False)

    # Self-check: the mean row must equal the mean of the seven refit rows, and the two refits the
    # text singles out must be the two smallest on contract_ym. Both are claims the appendix makes.
    refits = s[s["refit"] != "mean"]
    mean_row = s[s["refit"] == "mean"].iloc[0]
    ok = True
    for col in COLS.values():
        got, exp = float(mean_row[col]), float(refits[col].mean())
        # 5e-6: the two paths sum float32 SHAP in different orders, so they agree to about 1e-6.
        # That is four orders below the precision this table prints.
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
