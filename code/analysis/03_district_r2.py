#!/usr/bin/env python3
"""Aggregate stored predictions into district-level out-of-sample R2.

The input must include district labels for each test observation. R2 uses
the pooled test mean within each district. Figure 4 uses the one-month horizon.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401
from _paths import DERIVED_DATA_DIR, require  # noqa: E402

SLICE = DERIVED_DATA_DIR / "subsample_predictions_by_horizon.csv.gz"
DISTRICT_RESULTS_FILE = DERIVED_DATA_DIR / "subsample_district_r2_all_horizons.csv"

HORIZONS = [1, 3, 6, 12]

MODEL_COLS = {
    "y_pred_H1": "H1",
    "y_pred_H2": "H2",
    "y_pred_H3c": "H3c",
    "y_pred_H3g": "H3g",
    "y_pred_H4": "H4",
    "y_pred_H5": "H5",
    "y_pred_Ridge": "Ridge",
    "y_pred_XGBoost": "XGBoost",
}

EXPECTED_ROWS = 553_532
EXPECTED_DISTRICTS = 25

EXPECTED_PER_HORIZON = {1: (147_059, 84), 3: (142_786, 82), 6: (133_654, 79), 12: (130_033, 73)}


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def main() -> int:
    require(SLICE)
    df = pd.read_csv(SLICE)
    print(f"[district-r2] loaded {len(df):,} prediction rows across horizons "
          f"{sorted(df['horizon'].unique())}, {df['district'].nunique()} districts")

    problems = []
    if len(df) != EXPECTED_ROWS:
        problems.append(f"expected {EXPECTED_ROWS:,} rows, found {len(df):,}")
    if df["district"].nunique() != EXPECTED_DISTRICTS:
        problems.append(f"expected {EXPECTED_DISTRICTS} districts, "
                        f"found {df['district'].nunique()}")
    if int(df["district"].isna().sum()):
        problems.append(f"{int(df['district'].isna().sum())} rows carry no district")
    for h, (n_rows, n_win) in EXPECTED_PER_HORIZON.items():
        g = df[df["horizon"] == h]
        if len(g) != n_rows:
            problems.append(f"h={h}: expected {n_rows:,} rows, found {len(g):,}")
        if g["window_id"].nunique() != n_win:
            problems.append(f"h={h}: expected {n_win} windows, "
                            f"found {g['window_id'].nunique()}")
    if problems:
        print("[district-r2] FAIL: the prediction file does not match the expected input layout:")
        for p in problems:
            print(f"    {p}")
        return 2
    print("[district-r2] shape checks passed for all four horizons")

    rows = []
    for h in HORIZONS:
        dh = df[df["horizon"] == h]
        for col, label in MODEL_COLS.items():
            if col not in dh.columns:
                continue
            for dist, g in dh.groupby("district"):
                rows.append({
                    "district": dist,
                    "model": label,
                    "horizon": h,
                    "R2": oos_r2(g["y_true"].to_numpy(), g[col].to_numpy()),
                    "n_obs": len(g),
                })
    out = (pd.DataFrame(rows)
           .sort_values(["horizon", "model", "district"]).reset_index(drop=True))
    out.to_csv(DISTRICT_RESULTS_FILE, index=False)

    print(f"[district-r2] wrote {DISTRICT_RESULTS_FILE.name} "
          f"({out['district'].nunique()} districts x {out['model'].nunique()} models "
          f"x {out['horizon'].nunique()} horizons = {len(out)} rows)")
    for h in HORIZONS:
        xgb = out[(out["model"] == "XGBoost") & (out["horizon"] == h)]["R2"]
        print(f"[district-r2]   h={h:>2} XGB across districts: min={xgb.min():.4f} "
              f"max={xgb.max():.4f} spread={xgb.max() - xgb.min():.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
