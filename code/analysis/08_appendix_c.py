#!/usr/bin/env python3
"""Appendix C — Table C1 (predictors by specification) and Table C2 (out-of-sample R2).

Table C1 is a specification map rather than an estimate. It is generated from the same variable
sets the estimation code uses, not typed by hand, so a change to a specification shows up here
instead of silently disagreeing with the manuscript.

Table C2 reports mean out-of-sample R2 for H4, H5, H6 and XGBoost. H4, H5 and XGBoost come from
the shipped window-level results; H6 comes from code/build/04_appendix_c.py. The Diebold-Mariano
statistics the appendix text quotes are recomputed here and asserted, even though the printed
table carries R2 only, so every number in Appendix C traces to code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401  (starts the run log; see docs/ and the README's Run logs section)
from _paths import DERIVED, TABLES, engine_path, require  # noqa: E402

sys.path.insert(0, str(engine_path()))
from evaluation_metrics import diebold_mariano_test_hac  # noqa: E402

H6_WINDOWS = DERIVED / "appendix_c_window_level.csv"
SUB_WINDOWS = DERIVED / "subsample_window_level_results.csv"
HORIZONS = [1, 3, 6, 12]

# Table C1. Rows are predictor groups, values are the specifications that receive each group.
# "separate" marks H4, which is estimated within each district instead of carrying district
# indicators. Kept as data so the table and the manuscript cannot drift apart unnoticed.
SPECS = ["H1", "H2", "H3g", "H3c", "H4", "H5", "Ridge", "H6", "XGBoost"]
C1: list[tuple[str, dict[str, str]]] = [
    ("Floor area, floor, building age", {s: "Y" for s in SPECS}),
    ("Linear time trend", {s: "Y" for s in SPECS if s != "H1"}),
    ("Month fixed effects (11)", {s: "Y" for s in SPECS if s not in ("H1", "XGBoost")}),
    ("Contract month and year, as levels", {"XGBoost": "Y"}),
    ("District fixed effects (24)", {"H1": "Y", "H2": "Y", "H3g": "Y", "H4": "separate",
                                     "H5": "Y", "Ridge": "Y", "XGBoost": "Y"}),
    ("Dong fixed effects (241)", {"H6": "Y", "XGBoost": "Y"}),
    ("Apartment complex identifier", {"H3c": "Y", "XGBoost": "Y"}),
    ("Lot number, road name", {"XGBoost": "Y"}),
    ("Binary: parking, elevator, subway",
     {s: "Y" for s in ("H3g", "H3c", "H4", "H5", "Ridge", "H6", "XGBoost")}),
    ("Counts: parking, elevators, cameras, education",
     {s: "Y" for s in ("H3g", "H3c", "H4", "H5", "Ridge", "H6", "XGBoost")}),
    ("Subway walking band and line", {"H6": "Y", "XGBoost": "Y"}),
    ("Counts: welfare facilities, chargers", {"H6": "Y", "XGBoost": "Y"}),
    ("Binary: cameras, charger", {"H6": "Y", "XGBoost": "Y"}),
    ("Facility descriptions", {"XGBoost": "Y"}),
    ("log(area), area², age², area × floor", {"H5": "Y", "H6": "Y", "XGBoost": "Y"}),
    ("Area × district (24)", {"H5": "Y", "H6": "Y", "XGBoost": "Y"}),
    ("Transaction type (constant here)", {"XGBoost": "Y"}),
    ("Parameters or inputs", {"H1": "28", "H2": "40", "H3g": "47", "H3c": "-",
                              "H4": "23 per district", "H5": "75", "Ridge": "47",
                              "H6": "440", "XGBoost": "61"}),
]

# What the manuscript prints in Table C2 and quotes in the Appendix C text.
C2_EXPECTED = {
    ("H4", 1): 0.667, ("H4", 12): 0.370,
    ("H5", 1): 0.635, ("H5", 12): 0.378,
    ("H6", 1): 0.773, ("H6", 12): 0.493,
    ("XGBoost", 1): 0.864, ("XGBoost", 12): 0.661,
}
DM_EXPECTED = {1: 5.03, 3: 2.40, 6: 1.40, 12: 0.77}   # XGBoost against H6, appendix text
DM_TOL = 0.005


def main() -> int:
    require(H6_WINDOWS, SUB_WINDOWS)
    h6 = pd.read_csv(H6_WINDOWS)
    sub = pd.read_csv(SUB_WINDOWS)

    # ---- Table C1 ---------------------------------------------------------------------------
    c1 = pd.DataFrame(
        [{"Predictor group": g, **{s: row.get(s, "") for s in SPECS}} for g, row in C1])
    c1.to_csv(TABLES / "TableC1_predictors.csv", index=False)
    print(f"[appendix-c] TableC1_predictors.csv ({len(c1)} groups x {len(SPECS)} specifications)")

    # ---- Table C2 ---------------------------------------------------------------------------
    xgb = (sub[sub.model == "XGBoost"].groupby("horizon")["oos_r2"].mean()
             .rename("XGBoost").reset_index().assign(model="XGBoost")
             .rename(columns={"XGBoost": "oos_r2"}))
    agg = pd.concat([h6.groupby(["model", "horizon"])["oos_r2"].mean().reset_index(),
                     xgb[["model", "horizon", "oos_r2"]]])
    c2 = (agg[agg.horizon.isin(HORIZONS)]
          .pivot(index="model", columns="horizon", values="oos_r2")
          .reindex(["H4", "H5", "H6", "XGBoost"])[HORIZONS].round(3))
    c2.columns = [f"h={h}" for h in HORIZONS]
    c2.reset_index().rename(columns={"model": "Model"}).to_csv(
        TABLES / "TableC2_oos_r2.csv", index=False)
    print("[appendix-c] TableC2_oos_r2.csv")
    print(c2.to_string())

    # ---- self-checks ------------------------------------------------------------------------
    bad = 0
    for (m, h), want in sorted(C2_EXPECTED.items()):
        got = float(c2.loc[m, f"h={h}"])
        if abs(got - want) > 5e-4:
            print(f"[appendix-c] MISMATCH Table C2 {m} h={h}: {got:.3f} vs manuscript {want:.3f}")
            bad += 1

    # The appendix text quotes these four; the printed table does not carry them, so assert them
    # here rather than leaving a prose number with no executable source.
    mse = lambda d: (d.set_index(["window_id", "horizon"])["rmse"] ** 2)
    x = mse(sub[sub.model == "XGBoost"])
    h6w = h6[h6.model == "H6"].set_index(["window_id", "horizon"])
    ss = sub[sub.model == "H5"].set_index(["window_id", "horizon"])["std_y_true"]
    m6 = (1 - h6w["oos_r2"]) * ss.reindex(h6w.index) ** 2          # identity: rmse^2=(1-R2)std^2
    print("[appendix-c] Diebold-Mariano, XGBoost against H6")
    for h in HORIZONS:
        idx = sorted(i for i in h6w.index if i[1] == h)
        stat, _, p = diebold_mariano_test_hac(x.reindex(idx).to_numpy(),
                                              m6.reindex(idx).to_numpy(), h=h)
        got, want = abs(stat), DM_EXPECTED[h]
        flag = "OK" if abs(got - want) <= DM_TOL else "MISMATCH"
        print(f"[appendix-c]   h={h:>2} DM={got:6.2f} (manuscript {want:.2f}) p={p:.4f} {flag}")
        if flag == "MISMATCH":
            bad += 1

    if bad:
        print(f"[appendix-c] FAIL: {bad} values disagree with the manuscript.")
        return 6
    print("[appendix-c] self-check passed: Table C2 and the appendix DM statistics match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
