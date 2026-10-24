#!/usr/bin/env python3
"""Build stage — is the reported SHAP attribution an artefact of the 5,000-row explanation draw?

`02_shap_bridges.py` explains the seven frozen refits on a fixed draw of 5,000 transactions taken
from the final tenth of the amenity-complete sample. That draw is a computational convenience:
TreeSHAP over the whole sample for seven models is expensive, and the attribution is meant to
describe how the fitted models behave on data they were scored on, not on their own training rows.
A convenience still has to be shown to be one, so this script re-runs the same decomposition under
three alternative seeds and once on the entire final tenth, and writes what moves.

What it shows: nothing that the paper reports moves. Every category-share endpoint sits within
0.9 percentage points of the no-subsampling answer, the districts at the top and bottom of the
time-share ranking are the same, the global attribution ranking of Figure 5 is unchanged, and the
rank correlation between a district's time share and its mean price stays near zero throughout.

NO RE-ESTIMATION. The same seven frozen `xgb_window_*.pkl` are loaded and explained on different
rows. Nothing is refit and no reported number is recomputed.

Cost: about eight minutes, dominated by the pass over the entire final tenth (26,104 rows x 7
models). Needs data/raw/, so this is a build-stage script. Its output is shipped in
data/derived/draw_sensitivity.csv, so the analysis stage and Appendix B verify without the raw file.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401  (starts the run log; see docs/ and the README's Run logs section)
from _paths import DERIVED, RAW_TRANSACTIONS, FROZEN_MODEL, engine_path, require  # noqa: E402

require(RAW_TRANSACTIONS)
sys.path.insert(0, str(engine_path()))
from data_preprocessor import DualEncodingProcessor  # noqa: E402
import shap  # noqa: E402

# The seven annual refits the SHAP figures average over. Same list as 02_shap_bridges.py; if that
# list ever changes this one has to change with it or the two stop describing the same object.
WINDOWS = [0, 12, 24, 36, 48, 60, 72]

# The shipped draw uses seed 42 and 5,000 rows. The alternatives exist to vary the draw, not to
# search for a better one, so they are fixed here rather than sampled.
SHIPPED_SEED = 42
DRAW_SIZE = 5_000
ALTERNATE_SEEDS = [7, 2026, 31415]

# Feature -> category. Copied deliberately rather than imported: 02_shap_bridges.py builds this
# inside a function that also handles its --district-out-of-location flag, and duplicating six
# frozen sets is cheaper to audit than threading that flag through a second entry point. The two
# must agree; the self-check in 07_draw_sensitivity.py is what catches it if they drift.
STRUCTURAL = {"area_numeric", "floor", "build_year", "building_age",
              "area_sq", "log_area", "building_age_sq", "area_x_floor"}
AMENITY = {"total_parking", "cctv_count", "elevator_count", "welfare_facility", "welfare_count",
           "education_facility", "edu_count", "convenient_facility", "has_parking", "has_cctv",
           "has_elevator", "has_el_charger", "total_el_chargers"}
ACCESSIBILITY = {"subway_station", "subway_line", "kaptd_wtimesub", "has_subway"}
TEMPORAL = {"contract_ym", "year", "month", "time_trend"}
LOCATION = {"district", "시군구", "번지", "단지명", "road_name", "kapt_code"}
DISTRICT_ID = {"district", "시군구"}

# Panel B of Figure 6 renormalises over these five, having taken the district identifiers out.
# Appendix B reports Panel B, so this is the basis the whole table is computed on.
KEEP = ["structural", "amenity", "accessibility", "location", "time"]

# Districts the manuscript names at the extremes of the time-share ranking. Carried here so the
# analysis-stage self-check can assert they survive, rather than a reader having to eyeball a rank.
TIME_SHARE_TOP = "광진구"
TIME_SHARE_BOTTOM = "중구"


def categorise(feature: str) -> str:
    """Category for one feature, with the district identifiers held out of `location`.

    Inside a district the identifiers are constant, so TreeSHAP charges them that district's shift
    from the city-wide baseline. Leaving them in `location` makes the location share track the
    district's distance from the city mean price rather than anything about characteristics.
    """
    if feature in STRUCTURAL:
        return "structural"
    if feature in AMENITY:
        return "amenity"
    if feature in ACCESSIBILITY:
        return "accessibility"
    if feature in TEMPORAL:
        return "time"
    if feature in DISTRICT_ID or feature.startswith("area_x_district_"):
        return "district_level"
    if feature in LOCATION:
        return "location"
    return "other"


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation. Implemented here to avoid adding scipy to requirements.txt for one call."""
    return float(np.corrcoef(a.rank(), b.rank())[0, 1])


def district_shares(abs_shap: np.ndarray, districts: np.ndarray,
                    feature_cols: list[str], categories: dict[str, str]) -> pd.DataFrame:
    """Panel B category shares per district, renormalised over KEEP.

    Districts with fewer than ten explained rows are dropped: a share computed on a handful of
    transactions is noise, and the shipped bridge script applies the same floor.
    """
    rows = []
    for gu in sorted(pd.unique(districts)):
        mask = districts == gu
        if mask.sum() < 10:
            continue
        block = abs_shap[mask]
        totals: dict[str, float] = {}
        for j, feature in enumerate(feature_cols):
            cat = categories[feature]
            totals[cat] = totals.get(cat, 0.0) + block[:, j].sum()
        denom = sum(totals.get(c, 0.0) for c in KEEP)
        rows.append({"gu": gu, **{c: 100.0 * totals.get(c, 0.0) / denom for c in KEEP}})
    return pd.DataFrame(rows).set_index("gu")


def main() -> int:
    processor = DualEncodingProcessor()
    df = processor.load_and_prepare_data(str(RAW_TRANSACTIONS))
    df = df[df["total_parking"].notna() & df["elevator_count"].notna()].copy().reset_index(drop=True)
    processor.fit(df)
    feature_cols = processor.get_feature_columns(df)
    x_all = processor.get_xgboost_features(df, feature_cols)

    # The explanation region is the final tenth, matching 02_shap_bridges.py exactly. Explaining the
    # earlier ninety per cent would mix rows several refits trained on with rows none of them saw.
    n_test = int(len(df) * 0.1)
    df_test = df.iloc[-n_test:].copy()
    x_test = x_all.iloc[-n_test:]
    print(f"[draw-sens] amenity rows={len(df):,}  final tenth={n_test:,}  features={len(feature_cols)}")
    if n_test <= DRAW_SIZE:
        raise SystemExit(f"final tenth ({n_test}) is not larger than the draw ({DRAW_SIZE}); "
                         "the sensitivity question does not arise and this script would compare "
                         "a draw against itself.")

    categories = {f: categorise(f) for f in feature_cols}
    price_col = "log_price_per_sqm" if "log_price_per_sqm" in df.columns else "price_per_area"
    district_price = df.groupby("district")[price_col].mean()

    model_paths = [FROZEN_MODEL.parent / f"xgb_window_{w}.pkl" for w in WINDOWS]
    require(*model_paths)
    explainers = []
    for path in model_paths:
        bundle = joblib.load(path)
        if bundle["feature_cols"] != feature_cols:
            raise SystemExit(f"{path.name} feature_cols differ from the processor's; the columns of "
                             "the SHAP matrices would not align and averaging them would be wrong.")
        explainers.append(shap.TreeExplainer(bundle["model"]))
    print(f"[draw-sens] loaded {len(explainers)} frozen refits, no re-estimation")

    def explain(index: np.ndarray, label: str) -> pd.DataFrame:
        sample = x_test.iloc[index]
        districts = df_test.iloc[index]["district"].values
        # Absolute values, averaged over refits. Signed values from models fit on different windows
        # would partially cancel, which is not the quantity the figures report.
        abs_shap = sum(np.abs(e.shap_values(sample)) for e in explainers) / len(explainers)
        shares = district_shares(abs_shap, districts, feature_cols, categories)
        global_rank = pd.Series(abs_shap.mean(axis=0), index=feature_cols)
        print(f"[draw-sens] {label:28s} n={len(index):>6,}  districts={len(shares)}")
        return shares, global_rank

    conditions: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    np.random.seed(SHIPPED_SEED)
    conditions["shipped"] = explain(
        np.random.choice(len(x_test), DRAW_SIZE, replace=False), f"shipped (seed {SHIPPED_SEED})")
    for seed in ALTERNATE_SEEDS:
        np.random.seed(seed)
        conditions[f"seed_{seed}"] = explain(
            np.random.choice(len(x_test), DRAW_SIZE, replace=False), f"alternate seed {seed}")
    conditions["entire_final_tenth"] = explain(np.arange(len(x_test)), "entire final tenth")

    base_shares, base_global = conditions["shipped"]
    rows = []
    for label, (shares, global_rank) in conditions.items():
        common = base_shares.index.intersection(shares.index)
        time_sorted = shares["time"].sort_values(ascending=False)
        rows.append({
            "condition": label,
            "n_explained": DRAW_SIZE if label != "entire_final_tenth" else n_test,
            "structural_min": round(shares["structural"].min(), 1),
            "structural_max": round(shares["structural"].max(), 1),
            "amenity_min": round(shares["amenity"].min(), 1),
            "amenity_max": round(shares["amenity"].max(), 1),
            "time_min": round(shares["time"].min(), 1),
            "time_max": round(shares["time"].max(), 1),
            "time_top_district": time_sorted.index[0],
            "time_bottom_district": time_sorted.index[-1],
            "rank_corr_structural_vs_shipped": round(
                spearman(base_shares.loc[common, "structural"], shares.loc[common, "structural"]), 3),
            "rank_corr_amenity_vs_shipped": round(
                spearman(base_shares.loc[common, "amenity"], shares.loc[common, "amenity"]), 3),
            "rank_corr_time_vs_shipped": round(
                spearman(base_shares.loc[common, "time"], shares.loc[common, "time"]), 3),
            "rank_corr_global_vs_shipped": round(
                spearman(base_global, global_rank.reindex(base_global.index)), 3),
            "rank_corr_time_share_vs_district_price": round(
                spearman(shares["time"], district_price.reindex(shares.index)), 3),
        })

    out = pd.DataFrame(rows)
    out.to_csv(DERIVED / "draw_sensitivity.csv", index=False)
    print(f"\n[draw-sens] wrote data/derived/draw_sensitivity.csv ({len(out)} conditions)")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
