#!/usr/bin/env python3
"""Canonical SHAP-bridge generator — sea_2026 (added 2026-07-21; averaging added 2026-07-31).

Computes absolute SHAP from each of the seven shipped XGBoost refits, averages them, and writes the
three bridge CSVs the manuscript SHAP figures consume:
  - data/derived/shap_top15.csv                     -> Figure 5  (top-15 mean|SHAP|, District (Gu) aggregated)
  - data/derived/shap_district_decomp.csv           -> Figures 6, 7 (per-district 6-category shares; time_pct)
  - data/derived/figure8_building_shap_by_district.csv -> Figure 8 (25x12 per-district building-feature mean|SHAP|)

No re-estimation anywhere: the seven models are loaded, never retrained. SHAP is deterministic
(TreeExplainer, seed 42, last-10% 5,000-sample draw), and the draw depends only on the seed and the
row count, so every refit is explained on the same rows.

Usage:
  python3 code/build/02_shap_bridges.py                # seven-refit mean; writes the shipped CSVs
  python3 code/build/02_shap_bridges.py --window 24    # one refit; writes *_w24.csv
"""
import argparse, sys, os, numpy as np, pandas as pd, joblib
from pathlib import Path

# BUILD-stage script: needs the raw transactions file, which is not shipped. The three CSVs it
# writes ARE shipped in data/derived/, so the analysis stage renders Figures 5-8 without it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401  (starts the run log; see docs/ and the README's Run logs section)
# The seed is a literal, set before any work, and recorded in the run log.
SEED = _runlog.seed(42)
from _paths import DERIVED, FROZEN_MODEL, RAW_TRANSACTIONS, engine_path, require  # noqa: E402

# Which refit to explain. The default, "mean", averages absolute SHAP over all seven refits.
#
# That default is the one the manuscript figures use, and the reason is the question the SHAP
# section asks: it decomposes the forecasting advantage, and that advantage is measured over 84
# evaluation windows. XGBoost refits annually, so seven models produced those 84 forecasts, twelve
# windows each. Explaining one refit answers the question for twelve of the eighty-four.
#
# The choice is not cosmetic. Attribution to the time category runs 22.6-28.2 per cent of the
# per-district total in refits 0 through 48 and collapses to 6.6 and 3.3 per cent in refits 60 and
# 72, which train on 2021 onward, after the Seoul price run-up ended and a monotone time index
# stopped carrying predictive weight. Reporting refit 72 alone would report the extreme.
#
# Pass --window 0|12|24|36|48|60|72 to explain a single refit; that is how the window-sensitivity
# appendix is produced, and how a replicator checks the spread for themselves.
WINDOWS = [0, 12, 24, 36, 48, 60, 72]
_ap = argparse.ArgumentParser()
_ap.add_argument("--window", default="mean",
                 help='XGBoost refit to explain: "mean" (default, averages all seven) '
                      "or one of 0|12|24|36|48|60|72")
_ap.add_argument("--out-suffix", default="",
                 help="suffix for the written CSVs, so a single-window run does not "
                      "overwrite the shipped ones")
_ap.add_argument("--district-out-of-location", action="store_true",
                 help="Move the district identifiers OUT of the location category into their own "
                      "'district_level' category. Figure 6's location share is otherwise dominated by "
                      "the district dummy, whose per-district |SHAP| is that district's level shift "
                      "from the city baseline: measured on the shipped output, location_pct correlates "
                      "+0.83 with |district mean log price - city mean|. Splitting the two lets the "
                      "figure show whether the NON-district location features (complex, lot, road, "
                      "district-by-area interactions) still vary across the city once the level shift "
                      "is taken out. Writes to a separate file; the shipped default is unchanged.")
_args = _ap.parse_args()
_MEAN = str(_args.window).lower() == "mean"
if _MEAN:
    _sel = WINDOWS
else:
    _w = int(_args.window)
    if _w not in WINDOWS:
        raise SystemExit(f"--window must be 'mean' or one of {WINDOWS}; got {_args.window}")
    _sel = [_w]
MODEL_PATHS = [FROZEN_MODEL.parent / f"xgb_window_{w}.pkl" for w in _sel]
SFX = _args.out_suffix or ("" if _MEAN else f"_w{_sel[0]}")
if _args.district_out_of_location and not _args.out_suffix:
    SFX = SFX + "_nodistrict"

require(RAW_TRANSACTIONS, *MODEL_PATHS)
sys.path.insert(0, str(engine_path()))
from data_preprocessor import DualEncodingProcessor  # noqa: E402
import shap

# ---- data + frozen model ----
processor = DualEncodingProcessor()
df = processor.load_and_prepare_data(str(RAW_TRANSACTIONS))
df = df[df["total_parking"].notna() & df["elevator_count"].notna()].copy().reset_index(drop=True)
processor.fit(df)
feature_cols = processor.get_feature_columns(df)
X_all = processor.get_xgboost_features(df, feature_cols)

n_test = int(len(df) * 0.1)
df_test = df.iloc[-n_test:].copy()
X_test = X_all.iloc[-n_test:]
# SEED was set at import time (see the top of this file); re-assert it here so the draw below
# is reproducible even if an earlier statement consumed entropy. The draw depends only on the
# seed and the row count, so every refit is explained on the SAME 5,000 rows and the averaging
# below compares like with like.
np.random.seed(SEED)
idx = np.random.choice(len(X_test), min(5000, len(X_test)), replace=False)
X_sample = X_test.iloc[idx]
districts = df_test.iloc[idx]["district"].values
gus = sorted(df_test["district"].unique())

# Absolute SHAP, averaged over the selected refits. Averaging the ABSOLUTE values (not the signed
# ones) is what keeps this a magnitude statement: signed values from models fit on different
# training windows would partially cancel, which is not the quantity the figures report. All seven
# models carry identical feature_cols (verified: 61 features, field-for-field), so the columns of
# every matrix align.
PER_WINDOW = {}
for w, mp in zip(_sel, MODEL_PATHS):
    bundle = joblib.load(mp)
    if bundle["feature_cols"] != feature_cols:
        raise SystemExit(f"{mp.name} feature_cols differ from the processor's; cannot average.")
    PER_WINDOW[w] = np.abs(shap.TreeExplainer(bundle["model"]).shap_values(X_sample))
    print(f"[bridges]   explained {mp.name}")
ABS = sum(PER_WINDOW.values()) / len(PER_WINDOW)     # (5000, n_features)
absmean = ABS.mean(axis=0)
mabs = dict(zip(feature_cols, absmean))
_lab = f"mean of {len(MODEL_PATHS)} refits {WINDOWS}" if _MEAN else f"refit {_sel[0]} only"
print(f"[bridges] basis={_lab}")
print(f"[bridges] amenity rows={len(df)} n_features={len(feature_cols)} shap={ABS.shape}")

# ---- (1) shap_district_decomp.csv : 6-category shares per Gu (Fig 6/7) ----
structural = {'area_numeric','floor','build_year','building_age','area_sq','log_area','building_age_sq','area_x_floor'}
amenity = {'total_parking','cctv_count','elevator_count','welfare_facility','welfare_count','education_facility','edu_count','convenient_facility','has_parking','has_cctv','has_elevator','has_el_charger','total_el_chargers'}
accessibility = {'subway_station','subway_line','kaptd_wtimesub','has_subway'}
temporal = {'contract_ym','year','month','time_trend'}
location = {'district','시군구','번지','단지명','road_name','kapt_code'}
# The district identifiers. Inside one district these are constant, so TreeSHAP assigns them that
# district's shift from the model's city-wide baseline. Keeping them inside `location` is a defensible
# label, but it makes the location share track |district mean price - city mean| at r = +0.83, which
# is a level effect, not evidence that the model weights characteristics differently by district.
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
pd.DataFrame(rows).to_csv(DERIVED / f"shap_district_decomp{SFX}.csv", index=False)
print(f"[bridges] wrote data/derived/shap_district_decomp{SFX}.csv (Fig 6/7)")

# ---- (2) figure8_building_shap_by_district.csv : 25x12 building-feature mean|SHAP| (Fig 8) ----
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
# The district split changes only the category decomposition above. This file and shap_top15 are
# byte-identical with and without --district-out-of-location, so the flag does not write a copy.
if not _args.district_out_of_location:
    pd.DataFrame(rows8).to_csv(DERIVED / f"figure8_building_shap_by_district{SFX}.csv", index=False)
print(f"[bridges] wrote data/derived/figure8_building_shap_by_district{SFX}.csv (Fig 8)")

# ---- (3) shap_top15.csv : top-15 mean|SHAP| with renderer-compatible names (Fig 5) ----
# The renderer code/analysis/04_figures.py runs its OWN categorize_feature/display_name
# (code/analysis/figure_config.py) and sums all Location-category rows into a single 'District' bar. So the
# district-identity family (district + 시군구 + area_x_district_*) is pre-aggregated into ONE 'location_gu'
# row (to preserve its full magnitude in the top-15); other location strings are renamed to location_* and
# month->month_effect, kaptd_wtimesub->subway_time, so categorize_feature places them correctly.
loc_gu_family = {"district", "시군구"} | {f for f in feature_cols if f.startswith("area_x_district_")}
RENAME = {"단지명": "location_complex", "번지": "location_address", "road_name": "location_road",
          "kapt_code": "location_kapt", "month": "month_effect"}  # subway_time kept as kaptd_wtimesub (already Accessibility)
agg = {"location_gu": sum(mabs[f] for f in loc_gu_family if f in mabs)}
for f, v in mabs.items():
    if f in loc_gu_family:
        continue
    k = RENAME.get(f, f)
    agg[k] = agg.get(k, 0.0) + v
top = (pd.DataFrame({"feature": list(agg), "mean_abs_shap": list(agg.values())})
       .sort_values("mean_abs_shap", ascending=False).head(15).reset_index(drop=True))
top.to_csv(DERIVED / f"shap_top15{SFX}.csv", index=False)
print(f"[bridges] wrote data/derived/shap_top15{SFX}.csv (Fig 5)")
print(top.to_string(index=False))

# ---- (4) window_sensitivity.csv : the same quantities, refit by refit (appendix table) ----
# Written only on the default (mean) path, where every refit has already been explained, so this
# costs nothing extra. It exists so the averaging above is inspectable rather than asserted: a
# reader can see how far each refit departs from the mean, and in particular that attribution to
# time is an order of magnitude smaller in the last two refits than in the first five.
if _MEAN:
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
    pd.DataFrame(rows_s).to_csv(DERIVED / "window_sensitivity.csv", index=False)
    print("[bridges] wrote data/derived/window_sensitivity.csv (appendix)")
print("[bridges] DONE")
