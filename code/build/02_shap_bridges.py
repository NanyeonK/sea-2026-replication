#!/usr/bin/env python3
"""Generate SHAP inputs for Figures 5-8 and Appendix A.

Average absolute SHAP values across the seven fitted XGBoost models on the
same 5,000-row draw from the final tenth of the amenity sample. Use --window
to select one refit or --district-out-of-location to separate district identifiers.
"""
import argparse, sys, os, numpy as np, pandas as pd, joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401

SEED = _runlog.seed(42)
from _paths import DERIVED_DATA_DIR, FROZEN_MODEL, RAW_TRANSACTIONS, engine_path, require  # noqa: E402

WINDOWS = [0, 12, 24, 36, 48, 60, 72]
_ap = argparse.ArgumentParser()
_ap.add_argument("--window", default="mean",
                 help='XGBoost refit to explain: "mean" (default, averages all seven) '
                      "or one of 0|12|24|36|48|60|72")
_ap.add_argument("--out-suffix", default="",
                 help="Suffix for generated CSV filenames")
_ap.add_argument("--district-out-of-location", action="store_true",
                 help="Separate district identifiers from the location category")
_args = _ap.parse_args()
_AVERAGE_REFITS = str(_args.window).lower() == "mean"
if _AVERAGE_REFITS:
    _selected_windows = WINDOWS
else:
    _w = int(_args.window)
    if _w not in WINDOWS:
        raise SystemExit(f"--window must be 'mean' or one of {WINDOWS}; got {_args.window}")
    _selected_windows = [_w]
MODEL_PATHS = [FROZEN_MODEL.parent / f"xgb_window_{w}.pkl" for w in _selected_windows]
OUTPUT_SUFFIX = _args.out_suffix or ("" if _AVERAGE_REFITS else f"_w{_selected_windows[0]}")
if _args.district_out_of_location and not _args.out_suffix:
    OUTPUT_SUFFIX = OUTPUT_SUFFIX + "_nodistrict"

require(RAW_TRANSACTIONS, *MODEL_PATHS)
sys.path.insert(0, str(engine_path()))
from data_preprocessor import DualEncodingProcessor  # noqa: E402
import shap

processor = DualEncodingProcessor()
df = processor.load_and_prepare_data(str(RAW_TRANSACTIONS))
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
gus = sorted(df_test["district"].unique())

PER_WINDOW = {}
for w, mp in zip(_selected_windows, MODEL_PATHS):
    bundle = joblib.load(mp)
    if bundle["feature_cols"] != feature_cols:
        raise SystemExit(f"{mp.name} feature_cols differ from the processor's; cannot average.")
    PER_WINDOW[w] = np.abs(shap.TreeExplainer(bundle["model"]).shap_values(X_sample))
    print(f"[bridges]   explained {mp.name}")
ABS = sum(PER_WINDOW.values()) / len(PER_WINDOW)
absmean = ABS.mean(axis=0)
mabs = dict(zip(feature_cols, absmean))
_lab = f"mean of {len(MODEL_PATHS)} refits {WINDOWS}" if _AVERAGE_REFITS else f"refit {_selected_windows[0]} only"
print(f"[bridges] basis={_lab}")
print(f"[bridges] amenity rows={len(df)} n_features={len(feature_cols)} shap={ABS.shape}")

structural = {'area_numeric','floor','build_year','building_age','area_sq','log_area','building_age_sq','area_x_floor'}
amenity = {'total_parking','cctv_count','elevator_count','welfare_facility','welfare_count','education_facility','edu_count','convenient_facility','has_parking','has_cctv','has_elevator','has_el_charger','total_el_chargers'}
accessibility = {'subway_station','subway_line','kaptd_wtimesub','has_subway'}
temporal = {'contract_ym','year','month','time_trend'}
location = {'district','시군구','번지','단지명','road_name','kapt_code'}

DISTRICT_ID = {'district','시군구'}
_SPLIT = _args.district_out_of_location
def cat(f):
    if f in structural: return 'structural'
    if f in amenity: return 'amenity'
    if f in accessibility: return 'accessibility'
    if f in temporal: return 'time'
    if _SPLIT and (f in DISTRICT_ID or f.startswith('area_x_district_')): return 'district_level'
    if f in location: return 'location'
    if f.startswith('area_x_district_'): return 'location'
    return 'other'
categories = {f: cat(f) for f in feature_cols}
rows = []
for gu in gus:
    m = districts == gu
    if m.sum() < 10: continue
    svg = ABS[m]
    if svg.sum() == 0: continue
    cs = {}
    for j, fn in enumerate(feature_cols):
        cs[categories[fn]] = cs.get(categories[fn], 0) + svg[:, j].sum()
    allc = ['structural','amenity','accessibility','location','district_level','time','other']
    denom = sum(cs.get(c, 0) for c in allc)
    rows.append({'gu': gu,
        'structural_pct': cs.get('structural',0)/denom*100, 'amenity_pct': cs.get('amenity',0)/denom*100,
        'accessibility_pct': cs.get('accessibility',0)/denom*100, 'location_pct': cs.get('location',0)/denom*100,
        'district_level_pct': cs.get('district_level',0)/denom*100,
        'time_pct': cs.get('time',0)/denom*100, 'other_pct': cs.get('other',0)/denom*100})
pd.DataFrame(rows).to_csv(DERIVED_DATA_DIR / f"shap_district_decomp{OUTPUT_SUFFIX}.csv", index=False)
print(f"[bridges] wrote data/derived/shap_district_decomp{OUTPUT_SUFFIX}.csv (Fig 6/7)")

fig8 = ['area_numeric','floor','building_age','build_year','total_parking','cctv_count','elevator_count',
        'edu_count','has_subway','has_el_charger','kaptd_wtimesub','total_el_chargers']
rows8 = []
for gu in gus:
    m = districts == gu
    if m.sum() < 10: continue
    r = {'district': gu}
    for f in fig8:
        r[f] = ABS[m, feature_cols.index(f)].mean()
    rows8.append(r)

if not _args.district_out_of_location:
    pd.DataFrame(rows8).to_csv(DERIVED_DATA_DIR / f"figure8_building_shap_by_district{OUTPUT_SUFFIX}.csv", index=False)
print(f"[bridges] wrote data/derived/figure8_building_shap_by_district{OUTPUT_SUFFIX}.csv (Fig 8)")

loc_gu_family = {"district", "시군구"} | {f for f in feature_cols if f.startswith("area_x_district_")}
RENAME = {"단지명": "location_complex", "번지": "location_address", "road_name": "location_road",
          "kapt_code": "location_kapt", "month": "month_effect"}
agg = {"location_gu": sum(mabs[f] for f in loc_gu_family if f in mabs)}
for f, v in mabs.items():
    if f in loc_gu_family:
        continue
    k = RENAME.get(f, f)
    agg[k] = agg.get(k, 0.0) + v
top = (pd.DataFrame({"feature": list(agg), "mean_abs_shap": list(agg.values())})
       .sort_values("mean_abs_shap", ascending=False).head(15).reset_index(drop=True))
top.to_csv(DERIVED_DATA_DIR / f"shap_top15{OUTPUT_SUFFIX}.csv", index=False)
print(f"[bridges] wrote data/derived/shap_top15{OUTPUT_SUFFIX}.csv (Fig 5)")
print(top.to_string(index=False))

if _AVERAGE_REFITS:
    def _summary(A):
        am = A.mean(axis=0)
        m = dict(zip(feature_cols, am))
        out = {"location_gu": sum(m[f] for f in loc_gu_family if f in m)}
        for f in ("contract_ym", "build_year", "area_numeric", "total_parking", "time_trend"):
            out[f] = m.get(f, float("nan"))
        shares = []
        for gu in gus:
            g = districts == gu
            if g.sum() < 10:
                continue
            s = A[g]
            tot = s.sum()
            if tot == 0:
                continue
            shares.append(sum(s[:, j].sum() for j, fn in enumerate(feature_cols)
                              if categories[fn] == "time") / tot * 100)
        out["district_time_pct_mean"] = float(np.mean(shares))
        out["district_time_pct_min"] = float(np.min(shares))
        out["district_time_pct_max"] = float(np.max(shares))
        return out

    rows_s = [{"refit": str(w), **_summary(A)} for w, A in PER_WINDOW.items()]
    rows_s.append({"refit": "mean", **_summary(ABS)})
    pd.DataFrame(rows_s).to_csv(DERIVED_DATA_DIR / "window_sensitivity.csv", index=False)
    print("[bridges] wrote data/derived/window_sensitivity.csv (appendix)")
print("[bridges] DONE")
