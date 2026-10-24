#!/usr/bin/env python3
"""Generate SHAP inputs for Figures 5-8 and Appendix A.

Average absolute SHAP values across the seven fitted XGBoost models on the
same 5,000-row draw from the final tenth of the amenity sample. Use --window
to select one refit or --district-out-of-location to separate district identifiers.
"""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401

SEED = _runlog.seed(42)
from _paths import DERIVED_DATA_DIR, MODEL_DIR, TRANSACTIONS_FILE, engine_path, require  # noqa: E402

WINDOWS = [0, 12, 24, 36, 48, 60, 72]
parser = argparse.ArgumentParser()
parser.add_argument(
    "--window",
    default="mean",
    help='XGBoost refit to explain: "mean" (default, averages all seven) '
    "or one of 0|12|24|36|48|60|72",
)
parser.add_argument("--out-suffix", default="", help="Suffix for generated CSV filenames")
parser.add_argument(
    "--district-out-of-location",
    action="store_true",
    help="Separate district identifiers from the location category",
)
args = parser.parse_args()
AVERAGE_REFITS = str(args.window).lower() == "mean"
if AVERAGE_REFITS:
    selected_windows = WINDOWS
else:
    selected_window = int(args.window)
    if selected_window not in WINDOWS:
        raise SystemExit(f"--window must be 'mean' or one of {WINDOWS}; got {args.window}")
    selected_windows = [selected_window]
MODEL_PATHS = [MODEL_DIR / f"xgb_window_{w}.pkl" for w in selected_windows]
OUTPUT_SUFFIX = args.out_suffix or ("" if AVERAGE_REFITS else f"_w{selected_windows[0]}")
if args.district_out_of_location and not args.out_suffix:
    OUTPUT_SUFFIX = OUTPUT_SUFFIX + "_nodistrict"

require(TRANSACTIONS_FILE, *MODEL_PATHS)
sys.path.insert(0, str(engine_path()))
from data_preprocessor import DualEncodingProcessor  # noqa: E402

processor = DualEncodingProcessor()
df = processor.load_and_prepare_data(str(TRANSACTIONS_FILE))
df = df[df["total_parking"].notna() & df["elevator_count"].notna()].copy().reset_index(drop=True)
processor.fit(df)
feature_cols = processor.get_feature_columns(df)
X_all = processor.get_xgboost_features(df, feature_cols)

n_test = int(len(df) * 0.1)
df_test = df.iloc[-n_test:].copy()
X_test = X_all.iloc[-n_test:]

np.random.seed(SEED)
idx = np.random.choice(len(X_test), min(5000, len(X_test)), replace=False)
X_sample = X_test.iloc[idx]
districts = df_test.iloc[idx]["district"].values
district_names = sorted(df_test["district"].unique())

SHAP_BY_REFIT = {}
for w, mp in zip(selected_windows, MODEL_PATHS):
    bundle = joblib.load(mp)
    if bundle["feature_cols"] != feature_cols:
        raise SystemExit(f"{mp.name} feature columns do not match the prepared input.")
    SHAP_BY_REFIT[w] = np.abs(shap.TreeExplainer(bundle["model"]).shap_values(X_sample))
    print(f"[shap]   explained {mp.name}")
ABS_SHAP_VALUES = sum(SHAP_BY_REFIT.values()) / len(SHAP_BY_REFIT)
mean_abs_shap = ABS_SHAP_VALUES.mean(axis=0)
feature_mean_abs_shap = dict(zip(feature_cols, mean_abs_shap))
refit_description = (
    f"mean of {len(MODEL_PATHS)} refits {WINDOWS}"
    if AVERAGE_REFITS
    else f"refit {selected_windows[0]} only"
)
print(f"[shap] basis={refit_description}")
print(f"[shap] amenity rows={len(df)} n_features={len(feature_cols)} shap={ABS_SHAP_VALUES.shape}")

structural = {
    "area_numeric",
    "floor",
    "build_year",
    "building_age",
    "area_sq",
    "log_area",
    "building_age_sq",
    "area_x_floor",
}
amenity = {
    "total_parking",
    "cctv_count",
    "elevator_count",
    "welfare_facility",
    "welfare_count",
    "education_facility",
    "edu_count",
    "convenient_facility",
    "has_parking",
    "has_cctv",
    "has_elevator",
    "has_el_charger",
    "total_el_chargers",
}
accessibility = {"subway_station", "subway_line", "kaptd_wtimesub", "has_subway"}
temporal = {"contract_ym", "year", "month", "time_trend"}
location = {"district", "시군구", "번지", "단지명", "road_name", "kapt_code"}

DISTRICT_ID = {"district", "시군구"}
SEPARATE_DISTRICT_EFFECTS = args.district_out_of_location


def feature_category(f):
    if f in structural:
        return "structural"
    if f in amenity:
        return "amenity"
    if f in accessibility:
        return "accessibility"
    if f in temporal:
        return "time"
    if SEPARATE_DISTRICT_EFFECTS and (f in DISTRICT_ID or f.startswith("area_x_district_")):
        return "district_level"
    if f in location:
        return "location"
    if f.startswith("area_x_district_"):
        return "location"
    return "other"


categories = {f: feature_category(f) for f in feature_cols}
rows = []
for gu in district_names:
    m = districts == gu
    if m.sum() < 10:
        continue
    district_shap = ABS_SHAP_VALUES[m]
    if district_shap.sum() == 0:
        continue
    category_totals = {}
    for j, fn in enumerate(feature_cols):
        category_totals[categories[fn]] = (
            category_totals.get(categories[fn], 0) + district_shap[:, j].sum()
        )
    category_names = [
        "structural",
        "amenity",
        "accessibility",
        "location",
        "district_level",
        "time",
        "other",
    ]
    denom = sum(category_totals.get(c, 0) for c in category_names)
    rows.append(
        {
            "gu": gu,
            "structural_pct": category_totals.get("structural", 0) / denom * 100,
            "amenity_pct": category_totals.get("amenity", 0) / denom * 100,
            "accessibility_pct": category_totals.get("accessibility", 0) / denom * 100,
            "location_pct": category_totals.get("location", 0) / denom * 100,
            "district_level_pct": category_totals.get("district_level", 0) / denom * 100,
            "time_pct": category_totals.get("time", 0) / denom * 100,
            "other_pct": category_totals.get("other", 0) / denom * 100,
        }
    )
pd.DataFrame(rows).to_csv(
    DERIVED_DATA_DIR / f"shap_district_decomp{OUTPUT_SUFFIX}.csv", index=False
)
print(f"[shap] wrote data/derived/shap_district_decomp{OUTPUT_SUFFIX}.csv (Fig 6/7)")

fig8 = [
    "area_numeric",
    "floor",
    "building_age",
    "build_year",
    "total_parking",
    "cctv_count",
    "elevator_count",
    "edu_count",
    "has_subway",
    "has_el_charger",
    "kaptd_wtimesub",
    "total_el_chargers",
]
rows8 = []
for gu in district_names:
    m = districts == gu
    if m.sum() < 10:
        continue
    r = {"district": gu}
    for f in fig8:
        r[f] = ABS_SHAP_VALUES[m, feature_cols.index(f)].mean()
    rows8.append(r)

if not args.district_out_of_location:
    pd.DataFrame(rows8).to_csv(
        DERIVED_DATA_DIR / f"figure8_building_shap_by_district{OUTPUT_SUFFIX}.csv", index=False
    )
    print(f"[shap] wrote data/derived/figure8_building_shap_by_district{OUTPUT_SUFFIX}.csv (Fig 8)")

loc_gu_family = {"district", "시군구"} | {
    f for f in feature_cols if f.startswith("area_x_district_")
}
RENAME = {
    "단지명": "location_complex",
    "번지": "location_address",
    "road_name": "location_road",
    "kapt_code": "location_kapt",
    "month": "month_effect",
}
agg = {
    "location_gu": sum(
        feature_mean_abs_shap[f] for f in loc_gu_family if f in feature_mean_abs_shap
    )
}
for f, v in feature_mean_abs_shap.items():
    if f in loc_gu_family:
        continue
    k = RENAME.get(f, f)
    agg[k] = agg.get(k, 0.0) + v
top = (
    pd.DataFrame({"feature": list(agg), "mean_abs_shap": list(agg.values())})
    .sort_values("mean_abs_shap", ascending=False)
    .head(15)
    .reset_index(drop=True)
)
top.to_csv(DERIVED_DATA_DIR / f"shap_top15{OUTPUT_SUFFIX}.csv", index=False)
print(f"[shap] wrote data/derived/shap_top15{OUTPUT_SUFFIX}.csv (Fig 5)")
print(top.to_string(index=False))

if AVERAGE_REFITS:

    def _summary(A):
        am = A.mean(axis=0)
        m = dict(zip(feature_cols, am))
        out = {"location_gu": sum(m[f] for f in loc_gu_family if f in m)}
        for f in ("contract_ym", "build_year", "area_numeric", "total_parking", "time_trend"):
            out[f] = m.get(f, float("nan"))
        shares = []
        for gu in district_names:
            g = districts == gu
            if g.sum() < 10:
                continue
            s = A[g]
            tot = s.sum()
            if tot == 0:
                continue
            shares.append(
                sum(s[:, j].sum() for j, fn in enumerate(feature_cols) if categories[fn] == "time")
                / tot
                * 100
            )
        out["district_time_pct_mean"] = float(np.mean(shares))
        out["district_time_pct_min"] = float(np.min(shares))
        out["district_time_pct_max"] = float(np.max(shares))
        return out

    rows_s = [{"refit": str(w), **_summary(A)} for w, A in SHAP_BY_REFIT.items()]
    rows_s.append({"refit": "mean", **_summary(ABS_SHAP_VALUES)})
    pd.DataFrame(rows_s).to_csv(DERIVED_DATA_DIR / "window_sensitivity.csv", index=False)
    print("[shap] wrote data/derived/window_sensitivity.csv (appendix)")
print("[shap] DONE")
