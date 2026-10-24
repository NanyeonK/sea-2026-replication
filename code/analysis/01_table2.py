#!/usr/bin/env python3
"""Generate Table 2 and selected DM and SHAP summaries from prepared inputs.

Table 2 averages metrics across rolling windows at full precision and applies
half-up rounding once for display.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401
from _paths import DERIVED_DATA_DIR, TABLES_DIR, require  # noqa: E402
from _labels import standardize_model_labels  # noqa: E402

HORIZONS = [1, 3, 6, 12]

HEDONIC_MODELS = ["H1", "H2", "H3g", "H3c", "H4", "H5"]


def half_up(x: float, places: int) -> float:
    """Round a full-precision value to the requested places using ROUND_HALF_UP."""
    q = Decimal(1).scaleb(-places)
    # Convert NumPy scalars before constructing Decimal.
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


PERFORMANCE_FILES_BY_SAMPLE = {
    "Full": "full_window_level_results.csv",
    "Amenity": "subsample_window_level_results.csv",
}


def _aggregate(sample_name: str) -> pd.DataFrame:
    """Average R2, RMSE and MAE across rolling windows by model and horizon."""
    w = pd.read_csv(DERIVED_DATA_DIR / PERFORMANCE_FILES_BY_SAMPLE[sample_name])
    out = (
        w.groupby(["model", "horizon"])[["oos_r2", "rmse", "mae"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "model": "Model",
                "horizon": "Horizon",
                "oos_r2": "R2",
                "rmse": "RMSE",
                "mae": "MAE",
            }
        )
    )
    out = standardize_model_labels(out, "Model", dedupe_on=["Horizon"])
    return out[out["Horizon"].isin(HORIZONS)]


def _oos_table(sample_name: str) -> pd.DataFrame:
    """Return full-precision R2 values with models as rows and horizons as columns."""
    wide = _aggregate(sample_name).pivot(index="Model", columns="Horizon", values="R2")
    return wide.reindex(columns=HORIZONS)


def _manuscript_layout(sample_name: str) -> pd.DataFrame:
    """Format model-by-horizon R2, RMSE and MAE for Table 2."""
    out = _aggregate(sample_name).sort_values(["Model", "Horizon"]).reset_index(drop=True)
    for c in ["R2", "RMSE", "MAE"]:
        out[c] = out[c].map(lambda v: half_up(v, 3))
    return out


def build_table2() -> dict[str, float]:
    """Write Table 2 and XGBoost performance gaps for both samples."""
    checks: dict[str, float] = {}
    for sample_name in ("Full", "Amenity"):
        _manuscript_layout(sample_name).to_csv(
            TABLES_DIR / f"Table2_{sample_name}_R2_RMSE_MAE.csv", index=False
        )
        wide = _oos_table(sample_name)
        wide.to_csv(TABLES_DIR / f"Table2_{sample_name}.csv")
        xgb = wide.loc["XGBoost"]

        hed = wide.loc[[m for m in HEDONIC_MODELS if m in wide.index]]
        best = hed.max()
        gap = (xgb - best) * 100.0
        gap_tbl = pd.DataFrame(
            {"XGBoost": xgb, "BestHedonic": best, "BestHedonicModel": hed.idxmax(), "Gap_pp": gap}
        )
        gap_tbl.to_csv(TABLES_DIR / f"Table2_{sample_name}_gap.csv")

        # Reference checks use the percentages displayed in the article.
        xgb_pct = half_up(xgb[1] * 100, 1)
        best_pct = half_up(best[1] * 100, 1)
        checks[f"{sample_name}_XGB_h1"] = xgb_pct
        checks[f"{sample_name}_gap_h1"] = half_up(xgb_pct - best_pct, 1)
    return checks


def build_dm() -> dict[str, float]:
    """Write HAC-adjusted DM comparisons of XGBoost and H4 for the amenity sample."""
    dm = pd.read_csv(DERIVED_DATA_DIR / "dm_tests_hac.csv")
    sub = dm[
        ((dm.Model_1 == "XGBoost") & (dm.Model_2 == "H4"))
        | ((dm.Model_1 == "H4") & (dm.Model_2 == "XGBoost"))
    ].copy()
    sub = sub.sort_values("Horizon")[
        ["Horizon", "DM_Statistic", "P_Value", "Significance", "Better_Model"]
    ]
    sub.insert(0, "Sample", "Amenity subsample")
    sub.to_csv(TABLES_DIR / "Table3_DM_XGB_vs_H4.csv", index=False)
    return {"DM_h1": float(sub[sub.Horizon == 1].DM_Statistic.iloc[0])}


def build_shap() -> dict[str, float]:
    """Write the district SHAP category decomposition used in Figure 6."""
    shap = pd.read_csv(DERIVED_DATA_DIR / "shap_district_decomp.csv")
    shap.to_csv(TABLES_DIR / "Figure6_SHAP_district_decomp.csv", index=False)
    gangnam = shap[shap.gu == "강남구"].iloc[0]
    return {"Gangnam_location_pct": round(float(gangnam.location_pct), 1)}


def main() -> int:
    require(
        DERIVED_DATA_DIR / "full_window_level_results.csv",
        DERIVED_DATA_DIR / "subsample_window_level_results.csv",
        DERIVED_DATA_DIR / "dm_tests_hac.csv",
        DERIVED_DATA_DIR / "shap_district_decomp.csv",
    )
    checks = {**build_table2(), **build_dm(), **build_shap()}

    # Reference values for the study data vintage.
    expected = {
        "Full_XGB_h1": 82.0,
        "Full_gap_h1": 28.2,
        "Amenity_XGB_h1": 86.4,
        "Amenity_gap_h1": 19.7,
        "DM_h1": 9.4532,
        "Gangnam_location_pct": 58.0,
    }
    print(f"{'check':24s} {'got':>10s} {'expected':>10s}  result")
    ok = True
    for k, exp in expected.items():
        got = checks[k]

        match = abs(got - exp) < (0.01 if k == "DM_h1" else 0.051)
        ok = ok and match
        print(f"{k:24s} {got:>10.4f} {exp:>10.4f}  {'PASS' if match else 'FAIL'}")
    print("\nEXHIBITS written to:", TABLES_DIR)
    print("OVERALL:", "PASS — selected reference values match" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
