#!/usr/bin/env python3
"""Appendix C build stage — estimate H6, the maximal linear specification, over the same windows.

Appendix C asks how much of the gap between XGBoost and H4 is the predictor set rather than the
functional form. Answering it needs one specification that does not exist in Table 2, so this is
the only step in the package that estimates a model rather than reducing stored output. It needs
data/raw and takes roughly twenty minutes.

H6 = H5 + dong fixed effects + the remaining amenity and accessibility measures.

  H5 already carries every engineered term XGBoost holds, including all 25 area-by-district
  interactions. What it lacks is finer geography and a handful of complex-level measures. The raw
  location field is dong level, 242 values inside the amenity subsample, against the 25 districts
  the linear ladder uses. H6 swaps the 24 district dummies for 241 dong dummies, which nest within
  districts so both cannot enter, and adds the subway walking-time band, the subway line, the
  welfare facility count, the charger count and two indicators.

  The apartment-complex, lot-number and road-name identifiers stay out. H3c already places a
  high-cardinality location identifier in a linear specification and is the weakest specification
  in Table 2, so that question is answered rather than skipped.

H4 and H5 are re-estimated here as well and the script asserts they reproduce their published
Table 2 values. A new specification is only interpretable if the harness reproduces the old ones,
so a mismatch aborts rather than writing output.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401  (starts the run log; see docs/ and the README's Run logs section)
from _paths import DERIVED, RAW_TRANSACTIONS, engine_path, require  # noqa: E402

sys.path.insert(0, str(engine_path()))
from data_preprocessor import DualEncodingProcessor  # noqa: E402
from model_definitions import HedonicModelBuilder, ModelFactory  # noqa: E402
from rolling_window_engine import RollingWindowEngine  # noqa: E402

warnings.filterwarnings("ignore")

OUT = DERIVED / "appendix_c_window_level.csv"
HORIZONS = [1, 3, 6, 12]

# Published Table 2, amenity panel. The re-estimation must land on these or the harness is not the
# one that produced the deposit, and H6 cannot be read against it.
PUBLISHED = {
    ("H4", 1): 0.667153, ("H4", 3): 0.617393, ("H4", 6): 0.540189, ("H4", 12): 0.370290,
    ("H5", 1): 0.634923, ("H5", 3): 0.591332, ("H5", 6): 0.525379, ("H5", 12): 0.377766,
}
TOL = 5e-4

DONG_COL = "시군구"
EXTRA_CATEGORICAL = ["kaptd_wtimesub", "subway_line"]
EXTRA_NUMERIC = ["welfare_count", "total_el_chargers"]
EXTRA_BINARY = ["has_cctv", "has_el_charger"]


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Out-of-sample R2 against the mean of the test window, as the frozen engine computes it."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def level_maps(df: pd.DataFrame) -> dict[str, list[str]]:
    """Fix the dummy level set on the whole sample, as the project's pipeline does for districts.

    Levels are fixed once rather than per window so a column means the same thing in the training
    and prediction matrices. Missing values become their own level, because XGBoost routes missing
    values rather than dropping the row and a linear specification that dropped them would be
    handed an easier sample than the model it is compared with.
    """
    maps: dict[str, list[str]] = {}
    for c in EXTRA_CATEGORICAL + [DONG_COL]:
        v = df[c].fillna("__MISSING__").astype(str)
        ref = v.value_counts().index[0]          # most frequent level is the reference
        maps[c] = [x for x in sorted(v.unique()) if x != ref]
    return maps


def dummies(s: pd.Series, levels: list[str], prefix: str) -> pd.DataFrame:
    v = s.fillna("__MISSING__").astype(str)
    return pd.DataFrame({f"{prefix}_{lv}": (v == lv).astype(float).to_numpy() for lv in levels},
                        index=s.index)


def build_h6(df: pd.DataFrame, builder: HedonicModelBuilder,
             maps: dict[str, list[str]]) -> pd.DataFrame:
    X, _ = builder.build_feature_matrix(df, "H5")
    X = X.drop(columns=[c for c in X.columns if c.startswith("district_")])
    blocks = [X, dummies(df[DONG_COL], maps[DONG_COL], "dong")]
    blocks += [dummies(df[c], maps[c], c) for c in EXTRA_CATEGORICAL]
    num = df[EXTRA_NUMERIC].astype(float)
    blocks.append(num.fillna(num.median(numeric_only=True)))
    blocks.append(df[EXTRA_BINARY].astype(float))
    return pd.concat(blocks, axis=1)


def fit_predict(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame) -> np.ndarray:
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)
    model = sm.OLS(y_tr, sm.add_constant(X_tr, has_constant="add")).fit()
    return np.asarray(model.predict(sm.add_constant(X_te, has_constant="add")))


def main() -> int:
    require(RAW_TRANSACTIONS)
    df = DualEncodingProcessor().load_and_prepare_data(str(RAW_TRANSACTIONS))

    mask = df["total_parking"].notna() & df["elevator_count"].notna()
    df = df[mask].copy().reset_index(drop=True)
    print(f"[appendix-c] amenity subsample {len(df):,} rows, "
          f"{df['district'].nunique()} districts, {df[DONG_COL].nunique()} dong")

    y = df["log_price_per_sqm"]
    months = df["date"].dt.to_period("M")
    builder, factory = HedonicModelBuilder(), ModelFactory()
    maps = level_maps(df)
    print(f"[appendix-c] H6 adds {len(maps[DONG_COL])} dong, {len(maps['subway_line'])} subway "
          f"line and {len(maps['kaptd_wtimesub'])} walking-band dummies")

    windows = RollingWindowEngine(prediction_horizons=HORIZONS).create_rolling_windows(df["date"])

    rows = []
    for w in windows:
        tr = (months >= w["final_train_start"]) & (months <= w["final_train_end"])
        train_df, y_tr = df[tr], y[tr]
        if len(train_df) < 500:
            continue
        X5, X6 = builder.build_feature_matrix(train_df, "H5")[0], build_h6(train_df, builder, maps)
        X4, _ = builder.build_feature_matrix(train_df, "H4")
        h4 = factory.create_h4_district_models(train_df, X4, y_tr, min_observations=100)

        for h in HORIZONS:
            info = w["prediction_horizons"][h]
            if not info["valid"]:
                continue
            te = months == info["pred_month"]
            test_df, y_te = df[te], y[te]
            if len(test_df) == 0:
                continue

            preds = {
                "H5": fit_predict(X5, y_tr, builder.build_feature_matrix(test_df, "H5")[0]),
                "H6": fit_predict(X6, y_tr, build_h6(test_df, builder, maps)),
            }
            X4_te, _ = builder.build_feature_matrix(test_df, "H4")
            p4 = np.full(len(test_df), np.nan)
            for d, res in h4.items():
                m = (test_df["district"] == d).to_numpy()
                if m.sum():
                    p4[m] = res["model"].predict(
                        sm.add_constant(X4_te[m].reindex(columns=X4.columns, fill_value=0.0),
                                        has_constant="add"))
            preds["H4"] = p4

            for name, p in preds.items():
                ok = ~np.isnan(p)
                if ok.sum():
                    rows.append({"window_id": w["window_id"], "horizon": h, "model": name,
                                 "n_test": int(ok.sum()),
                                 "oos_r2": oos_r2(y_te.to_numpy()[ok], p[ok])})
        if w["window_id"] % 12 == 0:
            print(f"[appendix-c]   window {w['window_id']:>3} (train n={len(train_df):,})")

    res = pd.DataFrame(rows)
    agg = res.groupby(["model", "horizon"])["oos_r2"].mean()

    bad = []
    for (m, h), want in sorted(PUBLISHED.items()):
        got = float(agg.loc[(m, h)])
        flag = "OK" if abs(got - want) <= TOL else "MISMATCH"
        print(f"[appendix-c] {m} h={h:>2} published={want:.6f} re-estimated={got:.6f} {flag}")
        if flag == "MISMATCH":
            bad.append((m, h))
    if bad:
        print(f"[appendix-c] FAIL: {len(bad)} mismatches against Table 2. Not writing output, "
              f"because a harness that cannot reproduce the published specifications cannot be "
              f"used to interpret a new one.")
        return 4

    res.to_csv(OUT, index=False)
    print(f"[appendix-c] wrote {OUT.name} ({len(res)} rows, "
          f"H6 mean R2 {agg.loc[('H6', 1)]:.4f} at h=1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
