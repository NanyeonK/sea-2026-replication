#!/usr/bin/env python3
"""Generate Table 3 from pairwise HAC-adjusted Diebold-Mariano results."""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401
from _paths import DERIVED_DATA_DIR, TABLES_DIR, require  # noqa: E402
from _labels import standardize_model_labels  # noqa: E402

HORIZONS = [1, 3, 6, 12]
require(
    DERIVED_DATA_DIR / "table3_all_pairwise_dm_hac_full.csv",
    DERIVED_DATA_DIR / "table3_all_pairwise_dm_hac_subsample.csv",
)


def build(panel: str, csv: str) -> pd.DataFrame:
    df = pd.read_csv(DERIVED_DATA_DIR / csv)

    for c in ("Model_1", "Model_2"):
        df = standardize_model_labels(df, c)
    pairs = df[(df.Model_1 == "XGBoost") | (df.Model_2 == "XGBoost")]
    others = [m for m in pd.unique(pairs[["Model_1", "Model_2"]].values.ravel()) if m != "XGBoost"]
    rows = []
    for m in others:
        row = {"": f"vs {m}"}
        for h in HORIZONS:
            t = pairs[(pairs.Horizon == h) & ((pairs.Model_1 == m) | (pairs.Model_2 == m))]
            if t.empty:
                row[f"h={h}"] = "—"
                continue
            dm = abs(float(t.iloc[0].DM_Statistic))
            sig = str(t.iloc[0].Significance).replace("nan", "")
            row[f"h={h}"] = f"{dm:.2f}{sig}"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / f"Table3_{panel}_DM.csv", index=False)
    return out


def main() -> int:
    full = build("Full", "table3_all_pairwise_dm_hac_full.csv")
    sub = build("Subsample", "table3_all_pairwise_dm_hac_subsample.csv")

    chk = {
        "Full vs H4 h=1": (full[full[""] == "vs H4"]["h=1"].iloc[0], "14.87"),
        "Sub vs H4 h=1": (sub[sub[""] == "vs H4"]["h=1"].iloc[0], "9.45"),
    }
    ok = True
    for k, (got, exp) in chk.items():
        m = got.startswith(exp)
        ok = ok and m
        print(f"{k:18s} got={got:12s} exp~{exp}  {'PASS' if m else 'FAIL'}")
    print("\nFull panel:")
    print(full.to_string(index=False))
    print("\nSubsample panel:")
    print(sub.to_string(index=False))
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
