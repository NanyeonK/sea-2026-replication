#!/usr/bin/env python3
"""Evaluate SHAP attribution sensitivity to the explanation sample.

Explain the seven fitted XGBoost models using the baseline draw, three
alternative seeds and the entire final tenth of the amenity sample.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401
from _paths import DERIVED_DATA_DIR, TRANSACTIONS_FILE, MODEL_DIR, engine_path, require  # noqa: E402

require(TRANSACTIONS_FILE)
sys.path.insert(0, str(engine_path()))
from data_preprocessor import DualEncodingProcessor  # noqa: E402
import shap  # noqa: E402

WINDOWS = [0, 12, 24, 36, 48, 60, 72]

# Fixed baseline and alternative draws for the sensitivity comparison.
BASELINE_SEED = 42
DRAW_SIZE = 5_000
ALTERNATE_SEEDS = [7, 2026, 31415]

STRUCTURAL = {
    "area_numeric",
    "floor",
    "build_year",
    "building_age",
    "area_sq",
    "log_area",
    "building_age_sq",
    "area_x_floor",
}
AMENITY = {
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
ACCESSIBILITY = {"subway_station", "subway_line", "kaptd_wtimesub", "has_subway"}
TEMPORAL = {"contract_ym", "year", "month", "time_trend"}
LOCATION = {"district", "시군구", "번지", "단지명", "road_name", "kapt_code"}
DISTRICT_ID = {"district", "시군구"}

SHARE_CATEGORIES = ["structural", "amenity", "accessibility", "location", "time"]


def categorise(feature: str) -> str:
    """Assign a feature category, keeping district identifiers separate."""
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
    """Calculate Spearman rank correlation."""
    return float(np.corrcoef(a.rank(), b.rank())[0, 1])


def district_shares(
    abs_shap: np.ndarray, districts: np.ndarray, feature_cols: list[str], categories: dict[str, str]
) -> pd.DataFrame:
    """Calculate category shares excluding district identifiers.

    Omit districts with fewer than ten explained observations.
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
        denom = sum(totals.get(c, 0.0) for c in SHARE_CATEGORIES)
        rows.append({"gu": gu, **{c: 100.0 * totals.get(c, 0.0) / denom for c in SHARE_CATEGORIES}})
    return pd.DataFrame(rows).set_index("gu")


def main() -> int:
    processor = DualEncodingProcessor()
    df = processor.load_and_prepare_data(str(TRANSACTIONS_FILE))
    df = (
        df[df["total_parking"].notna() & df["elevator_count"].notna()].copy().reset_index(drop=True)
    )
    processor.fit(df)
    feature_cols = processor.get_feature_columns(df)
    x_all = processor.get_xgboost_features(df, feature_cols)

    n_test = int(len(df) * 0.1)
    df_test = df.iloc[-n_test:].copy()
    x_test = x_all.iloc[-n_test:]
    print(
        f"[draw-sens] amenity rows={len(df):,}  final tenth={n_test:,}  features={len(feature_cols)}"
    )
    if n_test <= DRAW_SIZE:
        raise SystemExit(
            f"The final tenth must contain more than {DRAW_SIZE} observations; found {n_test}."
        )

    categories = {f: categorise(f) for f in feature_cols}
    price_col = "log_price_per_sqm" if "log_price_per_sqm" in df.columns else "price_per_area"
    district_price = df.groupby("district")[price_col].mean()

    model_paths = [MODEL_DIR / f"xgb_window_{w}.pkl" for w in WINDOWS]
    require(*model_paths)
    explainers = []
    for path in model_paths:
        bundle = joblib.load(path)
        if bundle["feature_cols"] != feature_cols:
            raise SystemExit(f"{path.name} feature columns do not match the prepared input.")
        explainers.append(shap.TreeExplainer(bundle["model"]))
    print(f"[draw-sens] loaded {len(explainers)} frozen refits, no re-estimation")

    def explain(index: np.ndarray, label: str) -> tuple[pd.DataFrame, pd.Series]:
        sample = x_test.iloc[index]
        districts = df_test.iloc[index]["district"].values

        abs_shap = sum(np.abs(e.shap_values(sample)) for e in explainers) / len(explainers)
        shares = district_shares(abs_shap, districts, feature_cols, categories)
        global_rank = pd.Series(abs_shap.mean(axis=0), index=feature_cols)
        print(f"[draw-sens] {label:28s} n={len(index):>6,}  districts={len(shares)}")
        return shares, global_rank

    conditions: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    np.random.seed(BASELINE_SEED)
    conditions["shipped"] = explain(
        np.random.choice(len(x_test), DRAW_SIZE, replace=False), f"baseline (seed {BASELINE_SEED})"
    )
    for seed in ALTERNATE_SEEDS:
        np.random.seed(seed)
        conditions[f"seed_{seed}"] = explain(
            np.random.choice(len(x_test), DRAW_SIZE, replace=False), f"alternate seed {seed}"
        )
    conditions["entire_final_tenth"] = explain(np.arange(len(x_test)), "entire final tenth")

    base_shares, base_global = conditions["shipped"]
    rows = []
    for label, (shares, global_rank) in conditions.items():
        common = base_shares.index.intersection(shares.index)
        time_sorted = shares["time"].sort_values(ascending=False)
        rows.append(
            {
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
                    spearman(
                        base_shares.loc[common, "structural"], shares.loc[common, "structural"]
                    ),
                    3,
                ),
                "rank_corr_amenity_vs_shipped": round(
                    spearman(base_shares.loc[common, "amenity"], shares.loc[common, "amenity"]), 3
                ),
                "rank_corr_time_vs_shipped": round(
                    spearman(base_shares.loc[common, "time"], shares.loc[common, "time"]), 3
                ),
                "rank_corr_global_vs_shipped": round(
                    spearman(base_global, global_rank.reindex(base_global.index)), 3
                ),
                "rank_corr_time_share_vs_district_price": round(
                    spearman(shares["time"], district_price.reindex(shares.index)), 3
                ),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(DERIVED_DATA_DIR / "draw_sensitivity.csv", index=False)
    print(f"\n[draw-sens] wrote data/derived/draw_sensitivity.csv ({len(out)} conditions)")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
