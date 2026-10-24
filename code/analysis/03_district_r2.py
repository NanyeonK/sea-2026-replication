#!/usr/bin/env python3
"""Per-district out-of-sample R2 — the Figure 4 Panel A input.

Reads the shipped prediction slice `data/derived/subsample_predictions_by_horizon.csv.gz`, which carries
the district of every h=1 test observation (147,059 rows), and reduces it to a district x model
table of out-of-sample R2. No raw data and no estimation: stored predictions are read and
aggregated.

Why the district is pre-attached. The estimation engine stores predictions keyed by
(test_month, obs_idx), where obs_idx is the position of an observation inside its test-month
block. Recovering the district from those keys means rebuilding the 261,049-row estimation
sample from the 900 MB raw file. `code/build/00_assemble_derived.py` performs that join once, in
the author's tree, and ships the result, which is what lets this stage run without raw data. The
join was validated when it was made: zero unmatched rows, and all 84 window block sizes equal to
the engine's recorded n_test.

R2 definition follows the estimation engine (`evaluation_metrics.calculate_oos_r2`):
1 - SSres/SStot with SStot taken about the test mean, here each district's own pooled test mean.

Writes: data/derived/subsample_district_r2_all_horizons.csv  (district, model, horizon, R2, n_obs)

Background. Figure 4 Panel A used to have no input at all: the figure builder looked for this
file, never found it, and silently drew a bar chart of mean OOS R2 by model instead. See
the original archive's provenance notes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401  (starts the run log; see docs/ and the README's Run logs section)
from _paths import DERIVED, require  # noqa: E402

SLICE = DERIVED / "subsample_predictions_by_horizon.csv.gz"
OUT = DERIVED / "subsample_district_r2_all_horizons.csv"

# Manuscript Figure 4 plots h=1 only. All four horizons are computed and shipped because the
# pre-2026-02-21 pipeline kept a district_r2_all_horizons.csv that the restructure destroyed;
# restoring the full grid costs nothing here and removes a "why only h=1?" gap.
HORIZONS = [1, 3, 6, 12]

# prediction column -> model label used in the figure legend.
#
# `y_pred_H3` is deliberately absent. H3 and H3g are the same specification (see code/_labels.py),
# the engine estimated it twice, and the two prediction columns are identical to 0.0 over the first
# 500,000 rows. Reading only `y_pred_H3g` keeps one row per specification and matches the label the
# manuscript's Table 2 prints in both panels. H3c stays: it carries apartment-complex fixed effects
# instead of district ones and is a different model.
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
# rows and windows per horizon, as recorded by the engine
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
        print("[district-r2] FAIL: the shipped slice is not what this step expects:")
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
    out.to_csv(OUT, index=False)

    print(f"[district-r2] wrote {OUT.name} "
          f"({out['district'].nunique()} districts x {out['model'].nunique()} models "
          f"x {out['horizon'].nunique()} horizons = {len(out)} rows)")
    for h in HORIZONS:
        xgb = out[(out["model"] == "XGBoost") & (out["horizon"] == h)]["R2"]
        print(f"[district-r2]   h={h:>2} XGB across districts: min={xgb.min():.4f} "
              f"max={xgb.max():.4f} spread={xgb.max() - xgb.min():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
