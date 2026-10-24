#!/usr/bin/env python3
"""
Replication: build manuscript exhibits from canonical engine outputs.

Reads ONLY the canonical outputs produced by
  the rolling-window estimation engine
and regenerates the paper's headline exhibits: Table 2 (Full + Amenity), the
Diebold-Mariano comparison against the best hedonic specification, and the
SHAP category decomposition by district.

Table 2 is aggregated from the per-window results at full precision and rounded once; see
`half_up` for why that matters. Six headline numbers are self-checked against the manuscript's
printed values. Exit 0 = PASS.

Usage:
    python3 code/analysis/01_table2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401  (starts the run log; see docs/ and the README's Run logs section)
from _paths import DERIVED, TABLES, require  # noqa: E402
from _labels import canonical_models  # noqa: E402

RNR = DERIVED
OUT = TABLES

HORIZONS = [1, 3, 6, 12]
# The hedonic ladder, over which "the best linear specification" is taken. H3g rather than H3
# because labels are canonicalised before this list is applied; see code/_labels.py. H3c belongs to
# the ladder too and is included, though it never attains the maximum at any horizon in either
# panel — it is the weakest hedonic specification in the amenity sample.
HEDONIC = ["H1", "H2", "H3g", "H3c", "H4", "H5"]


def half_up(x: float, places: int) -> float:
    """Round half away from zero, once, from the full-precision estimate.

    Two things matter here and they were conflated once already.

    The convention: Python and pandas round half to even, so 0.3385 would print as 0.338 where
    conventional presentation gives 0.339. Half-up is the right convention for a printed table.

    The input: the rounding must be applied to the FULL-PRECISION estimate, never to an already
    rounded intermediate. `{full,subsample}_table2_multi_horizon.csv` stores four decimals, and
    rounding those to three pushes any value that landed on x.xxx5 up by one unit in the last
    place. Eight cells across the two panels were affected, the worst being the amenity-subsample
    XGBoost h=1 R2: the estimate is 0.8644826, which prints as 0.864, but via the four-decimal
    0.8645 it printed as 0.865 — and the derived gap as 19.8 pp instead of 19.7. This module
    therefore aggregates from `*_window_level_results.csv`, which carries per-window values at
    full precision, and rounds exactly once.
    """
    q = Decimal(1).scaleb(-places)
    # repr() of a numpy scalar is "np.float64(...)", which Decimal cannot parse; cast first.
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


# panel -> the window-level file that carries per-window values at full precision
WINDOW_LEVEL = {"Full": "full_window_level_results.csv",
                "Amenity": "subsample_window_level_results.csv"}


def _aggregate(panel: str) -> pd.DataFrame:
    """Model x Horizon means of R2, RMSE and MAE, computed at full precision.

    Table 2's published quantity is the mean across rolling windows. Taking it from the
    window-level file rather than the pre-rounded aggregate CSV is what keeps the single rounding
    step honest; see `half_up`.
    """
    w = pd.read_csv(RNR / WINDOW_LEVEL[panel])
    out = (w.groupby(["model", "horizon"])[["oos_r2", "rmse", "mae"]]
             .mean().reset_index()
             .rename(columns={"model": "Model", "horizon": "Horizon",
                              "oos_r2": "R2", "rmse": "RMSE", "mae": "MAE"}))
    out = canonical_models(out, "Model", dedupe_on=["Horizon"])
    return out[out["Horizon"].isin(HORIZONS)]


def _oos_table(panel: str) -> pd.DataFrame:
    """Model x Horizon OOS R2 at full precision."""
    wide = _aggregate(panel).pivot(index="Model", columns="Horizon", values="R2")
    return wide.reindex(columns=HORIZONS)


def _manuscript_layout(panel: str) -> pd.DataFrame:
    """Table 2 as the manuscript prints it: Model x Horizon rows, R2 / RMSE / MAE columns.

    The pivot above carries only R2, which is what the headline claims use, but the manuscript
    table has three value columns. Emitting both keeps the deposit's Table 2 a full match rather
    than a reduction of the printed table.
    """
    out = _aggregate(panel).sort_values(["Model", "Horizon"]).reset_index(drop=True)
    for c in ["R2", "RMSE", "MAE"]:
        out[c] = out[c].map(lambda v: half_up(v, 3))
    return out


def build_table2() -> dict[str, float]:
    """Table 2: OOS R2 by model x horizon, Full and Amenity panels, + gaps."""
    checks: dict[str, float] = {}
    for panel in ("Full", "Amenity"):
        _manuscript_layout(panel).to_csv(OUT / f"Table2_{panel}_R2_RMSE_MAE.csv", index=False)
        wide = _oos_table(panel)
        wide.to_csv(OUT / f"Table2_{panel}.csv")
        xgb = wide.loc["XGBoost"]
        # best hedonic per horizon among H1..H5 present in this panel
        hed = wide.loc[[m for m in HEDONIC if m in wide.index]]
        best = hed.max()
        gap = (xgb - best) * 100.0
        gap_tbl = pd.DataFrame({"XGBoost": xgb, "BestHedonic": best,
                                "BestHedonicModel": hed.idxmax(),
                                "Gap_pp": gap})
        gap_tbl.to_csv(OUT / f"Table2_{panel}_gap.csv")
        # Rounding once from full precision, the two conventions agree for the gaps this check
        # covers: Full 28.2 and Amenity 19.7 come out the same whether you subtract the displayed
        # values or round the full-precision difference. The registry's "mixed convention" was, for
        # THESE gaps, an artefact of the double rounding rather than a methodological choice.
        #
        # It is not artefactual everywhere. The XGBoost-versus-H5 gap in the full panel still
        # differs by convention — 82.0 - 51.3 = 30.7 against a full-precision 30.64 that rounds to
        # 30.6 — because subtracting two separately rounded operands is not the same as rounding
        # their difference. That one is a real choice and the manuscript takes the full-precision
        # reading. Corrected 2026-07-31; an earlier version of this comment claimed the mixed
        # convention had disappeared entirely, which overstated the fix.
        xgb_pct = half_up(xgb[1] * 100, 1)
        best_pct = half_up(best[1] * 100, 1)
        checks[f"{panel}_XGB_h1"] = xgb_pct
        checks[f"{panel}_gap_h1"] = half_up(xgb_pct - best_pct, 1)
    return checks


def build_dm() -> dict[str, float]:
    """DM (HAC), XGBoost vs H4, all horizons — AMENITY SUBSAMPLE.

    `dm_tests_hac.csv` is the subsample file: it matches
    `table3_all_pairwise_dm_hac_subsample.csv` exactly and differs from the full-sample pairwise
    file by up to 5.42. The output filename does not say so, hence this note and the column added
    below.
    """
    dm = pd.read_csv(RNR / "dm_tests_hac.csv")
    sub = dm[((dm.Model_1 == "XGBoost") & (dm.Model_2 == "H4")) |
             ((dm.Model_1 == "H4") & (dm.Model_2 == "XGBoost"))].copy()
    sub = sub.sort_values("Horizon")[["Horizon", "DM_Statistic", "P_Value",
                                      "Significance", "Better_Model"]]
    sub.insert(0, "Sample", "Amenity subsample")
    sub.to_csv(OUT / "Table3_DM_XGB_vs_H4.csv", index=False)
    return {"DM_h1": float(sub[sub.Horizon == 1].DM_Statistic.iloc[0])}


def build_shap() -> dict[str, float]:
    """SHAP feature-category decomposition by district (Figure 6)."""
    shap = pd.read_csv(RNR / "shap_district_decomp.csv")
    shap.to_csv(OUT / "Figure6_SHAP_district_decomp.csv", index=False)
    gangnam = shap[shap.gu == "강남구"].iloc[0]
    return {"Gangnam_location_pct": round(float(gangnam.location_pct), 1)}


def main() -> int:
    require(RNR / "full_window_level_results.csv",
            RNR / "subsample_window_level_results.csv",
            RNR / "dm_tests_hac.csv",
            RNR / "shap_district_decomp.csv")
    checks = {**build_table2(), **build_dm(), **build_shap()}

    # The manuscript's printed headline values.
    # Corrected 2026-07-30: the amenity figures were 86.5 / 19.8, which came from rounding the
    # four-decimal intermediate rather than the estimate. See `half_up`.
    # Updated 2026-07-31: Gangnam's location share was 63.7, the value from refit 72 alone. The
    # SHAP bridges now average all seven refits (see code/build/02_shap_bridges.py for why), which
    # gives 58.0. The forecasting checks above are untouched by that change — they read the stored
    # 84-window predictions and never load a model.
    expected = {
        "Full_XGB_h1": 82.0, "Full_gap_h1": 28.2,
        "Amenity_XGB_h1": 86.4, "Amenity_gap_h1": 19.7,
        "DM_h1": 9.4532, "Gangnam_location_pct": 58.0,
    }
    print(f"{'check':24s} {'got':>10s} {'expected':>10s}  result")
    ok = True
    for k, exp in expected.items():
        got = checks[k]
        # Exact to the printed precision. The old 0.15 tolerance existed to absorb a rounding
        # convention mismatch, not numerical noise; with one convention applied throughout, every
        # check reproduces the manuscript's own displayed value exactly.
        match = abs(got - exp) < (0.01 if k == "DM_h1" else 0.051)
        ok = ok and match
        print(f"{k:24s} {got:>10.4f} {exp:>10.4f}  {'PASS' if match else 'FAIL'}")
    print("\nEXHIBITS written to:", OUT)
    print("OVERALL:", "PASS — exhibits reproduce manuscript" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
