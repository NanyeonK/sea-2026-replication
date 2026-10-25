#!/usr/bin/env python3
"""Generate Appendix C specification and performance tables.

H4, H5 and H6 use code/build/04_appendix_c.py outputs. XGBoost uses the
prepared amenity-sample window results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401
from _paths import DERIVED_DATA_DIR, TABLES_DIR, engine_path, require  # noqa: E402

sys.path.insert(0, str(engine_path()))
from evaluation_metrics import diebold_mariano_test_hac  # noqa: E402

APPENDIX_RESULTS_FILE = DERIVED_DATA_DIR / "appendix_c_window_level.csv"
AMENITY_RESULTS_FILE = DERIVED_DATA_DIR / "subsample_window_level_results.csv"
HORIZONS = [1, 3, 6, 12]

MODEL_SPECIFICATIONS = ["H1", "H2", "H3g", "H3c", "H4", "H5", "Ridge", "H6", "XGBoost"]
PREDICTOR_GROUPS: list[tuple[str, dict[str, str]]] = [
    ("Structural attributes", {s: "✓" for s in MODEL_SPECIFICATIONS}),
    ("Linear time trend", {s: "✓" for s in MODEL_SPECIFICATIONS if s != "H1"}),
    (
        "Month fixed effects",
        {s: "✓" for s in MODEL_SPECIFICATIONS if s not in ("H1", "XGBoost")},
    ),
    ("Contract month and year, as levels", {"XGBoost": "✓"}),
    (
        "District",
        {s: "✓" for s in ("H1", "H2", "H3g", "H4", "H5", "Ridge", "XGBoost")},
    ),
    ("Dong", {"H6": "✓", "XGBoost": "✓"}),
    ("Complex name", {"H3c": "✓", "XGBoost": "✓"}),
    ("Lot number, road name", {"XGBoost": "✓"}),
    (
        "Binary: parking, elevator, subway",
        {s: "✓" for s in ("H3g", "H3c", "H4", "H5", "Ridge", "H6", "XGBoost")},
    ),
    (
        "Counts: parking, elevators, cameras, education",
        {s: "✓" for s in ("H3g", "H3c", "H4", "H5", "Ridge", "H6", "XGBoost")},
    ),
    ("Subway walking band and line", {"H6": "✓", "XGBoost": "✓"}),
    ("Counts: welfare facilities, chargers", {"H6": "✓", "XGBoost": "✓"}),
    ("Binary: cameras, charger", {"H6": "✓", "XGBoost": "✓"}),
    ("Facility descriptions", {"XGBoost": "✓"}),
    ("log(area), area², age², area × floor", {"H5": "✓", "H6": "✓", "XGBoost": "✓"}),
    ("Area × district", {"H5": "✓", "H6": "✓", "XGBoost": "✓"}),
    ("Transaction type (constant here)", {"XGBoost": "✓"}),
]

C2_EXPECTED = {
    ("H4", 1): 0.667,
    ("H4", 12): 0.370,
    ("H5", 1): 0.635,
    ("H5", 12): 0.378,
    ("H6", 1): 0.773,
    ("H6", 12): 0.493,
    ("XGBoost", 1): 0.864,
    ("XGBoost", 12): 0.661,
}
DM_EXPECTED = {1: 5.03, 3: 2.40, 6: 1.40, 12: 0.77}
DM_TOLERANCE = 0.005


def main() -> int:
    require(APPENDIX_RESULTS_FILE, AMENITY_RESULTS_FILE)
    h6 = pd.read_csv(APPENDIX_RESULTS_FILE)
    sub = pd.read_csv(AMENITY_RESULTS_FILE)

    c1 = pd.DataFrame(
        [
            {"Predictor group": g, **{s: row.get(s, "") for s in MODEL_SPECIFICATIONS}}
            for g, row in PREDICTOR_GROUPS
        ]
    )
    c1.to_csv(TABLES_DIR / "TableC1_predictors.csv", index=False)
    print(
        f"[appendix-c] TableC1_predictors.csv ({len(c1)} groups x {len(MODEL_SPECIFICATIONS)} specifications)"
    )

    xgb = (
        sub[sub.model == "XGBoost"]
        .groupby("horizon")["oos_r2"]
        .mean()
        .rename("XGBoost")
        .reset_index()
        .assign(model="XGBoost")
        .rename(columns={"XGBoost": "oos_r2"})
    )
    agg = pd.concat(
        [
            h6.groupby(["model", "horizon"])["oos_r2"].mean().reset_index(),
            xgb[["model", "horizon", "oos_r2"]],
        ]
    )
    c2 = (
        agg[agg.horizon.isin(HORIZONS)]
        .pivot(index="model", columns="horizon", values="oos_r2")
        .reindex(["H4", "H5", "H6", "XGBoost"])[HORIZONS]
        .round(3)
    )
    c2.columns = [f"h={h}" for h in HORIZONS]
    c2.reset_index().rename(columns={"model": "Model"}).to_csv(
        TABLES_DIR / "TableC2_oos_r2.csv", index=False
    )
    print("[appendix-c] TableC2_oos_r2.csv")
    print(c2.to_string())

    bad = 0
    for (m, h), want in sorted(C2_EXPECTED.items()):
        got = float(c2.loc[m, f"h={h}"])
        if abs(got - want) > 5e-4:
            print(f"[appendix-c] MISMATCH Table C2 {m} h={h}: {got:.3f} vs manuscript {want:.3f}")
            bad += 1

    x = sub[sub.model == "XGBoost"].set_index(["window_id", "horizon"])["rmse"] ** 2
    h6w = h6[h6.model == "H6"].set_index(["window_id", "horizon"])
    ss = sub[sub.model == "H5"].set_index(["window_id", "horizon"])["std_y_true"]
    m6 = (1 - h6w["oos_r2"]) * ss.reindex(h6w.index) ** 2
    print("[appendix-c] Diebold-Mariano, XGBoost against H6")
    for h in HORIZONS:
        idx = sorted(i for i in h6w.index if i[1] == h)
        stat, _, p = diebold_mariano_test_hac(
            x.reindex(idx).to_numpy(), m6.reindex(idx).to_numpy(), h=h
        )
        got, want = -stat, DM_EXPECTED[h]  # MSE(H6) - MSE(XGBoost)
        flag = "OK" if abs(got - want) <= DM_TOLERANCE else "MISMATCH"
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
