#!/usr/bin/env python3
"""
Global visual strip: remove axis labels, grid lines, and panel titles
from all active figures (1-8). No other changes.
"""
import os
# Pin matplotlib's PDF creation timestamp so the PDFs are byte-reproducible. run_all.py sets this
# for every step; the default here keeps a standalone invocation of this script reproducible too.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401  (starts the run log; see docs/ and the README's Run logs section)
from _labels import canonical_models  # noqa: E402
from _paths import DERIVED, FIGURES, RAW_GEOJSON, require  # noqa: E402
from figure_config import (CATEGORY_PALETTE, HORIZON_COLORS, VARIABLE_DISPLAY,  # noqa: E402
                           categorize_feature, display_name)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

outdir = FIGURES
outdir.mkdir(parents=True, exist_ok=True)

gu_map = {'강남구':'Gangnam','강동구':'Gangdong','강북구':'Gangbuk','강서구':'Gangseo',
    '관악구':'Gwanak','광진구':'Gwangjin','구로구':'Guro','금천구':'Geumcheon',
    '노원구':'Nowon','도봉구':'Dobong','동대문구':'Dongdaemun','동작구':'Dongjak',
    '마포구':'Mapo','서대문구':'Seodaemun','서초구':'Seocho','성동구':'Seongdong',
    '성북구':'Seongbuk','송파구':'Songpa','양천구':'Yangcheon','영등포구':'Yeongdeungpo',
    '용산구':'Yongsan','은평구':'Eunpyeong','종로구':'Jongno','중구':'Jung','중랑구':'Jungnang'}

results = {}

# ======================== FIGURE 1 ========================
print("Figure 1...")
df1 = pd.read_csv(DERIVED / "full_window_level_results.csv")
df1 = df1[(df1["horizon"] == 1) & (df1["model"].isin(["H1", "XGBoost"]))]
df1["year"] = pd.to_datetime(df1["test_month"]).dt.year
yearly = df1.groupby(["year", "model"])["oos_r2"].mean().unstack("model")

fig, ax = plt.subplots(figsize=(6, 3.5))
x = np.arange(len(yearly))
w = 0.35
ax.bar(x - w/2, yearly["H1"], w, label="H1", color=CATEGORY_PALETTE["Building"], edgecolor="white", linewidth=0.5)
ax.bar(x + w/2, yearly["XGBoost"], w, label="XGBoost", color=CATEGORY_PALETTE["Amenity"], edgecolor="white", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(yearly.index.astype(int))
# STRIPPED: no ylabel, no grid, no title
ax.legend(framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(outdir / "figure1_yearly_h1_xgb_fullsample.png")
fig.savefig(outdir / "figure1_yearly_h1_xgb_fullsample.pdf")
plt.close()
results["figure1"] = "OK"
print("  Done")

# ======================== FIGURE 2 ========================
print("Figure 2...")
import warnings
warnings.filterwarnings('ignore')

# Third-party data is not redistributed, so the Seoul boundary GeoJSON is
# not in this package. Figure 2 is the only exhibit that needs it. Its rendered PNGs are shipped
# in output/figures/; this step regenerates them only if a reader supplies the boundary file.
geojson_path = RAW_GEOJSON
if not geojson_path.exists():
    results["figure2"] = "SKIPPED (boundary GeoJSON not distributed; see data/README.md)"
    print(f"  SKIPPED — {geojson_path.name} is a third-party file and is not")
    print("           redistributed. The rendered Figure 2 PNGs are shipped in output/figures/.")
    print("           To regenerate, place the file there; the source is named in")
    print("           data/README.md.")
try:
    if not geojson_path.exists():
        raise FileNotFoundError(geojson_path)
    import geopandas as gpd
    from matplotlib.colors import LinearSegmentedColormap

    panel_e = pd.read_csv(DERIVED / 'table1_panel_e_spatial_all25_english.csv')
    seoul_gdf = gpd.read_file(geojson_path)

    district_name_map = {
        '강남구': 'Gangnam-gu', '강동구': 'Gangdong-gu', '강북구': 'Gangbuk-gu',
        '강서구': 'Gangseo-gu', '관악구': 'Gwanak-gu', '광진구': 'Gwangjin-gu',
        '구로구': 'Guro-gu', '금천구': 'Geumcheon-gu', '노원구': 'Nowon-gu',
        '도봉구': 'Dobong-gu', '동대문구': 'Dongdaemun-gu', '동작구': 'Dongjak-gu',
        '마포구': 'Mapo-gu', '서대문구': 'Seodaemun-gu', '서초구': 'Seocho-gu',
        '성동구': 'Seongdong-gu', '성북구': 'Seongbuk-gu', '송파구': 'Songpa-gu',
        '양천구': 'Yangcheon-gu', '영등포구': 'Yeongdeungpo-gu', '용산구': 'Yongsan-gu',
        '은평구': 'Eunpyeong-gu', '종로구': 'Jongno-gu', '중구': 'Jung-gu',
        '중랑구': 'Jungnang-gu',
    }
    seoul_gdf['District'] = seoul_gdf['SIG_KOR_NM'].map(district_name_map)
    seoul_gdf = seoul_gdf.merge(panel_e, on='District', how='left')

    volume_colors = ['#FFF5F0', '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A',
                     '#EF3B2C', '#CB181D', '#A50F15', '#67000D']
    volume_cmap = LinearSegmentedColormap.from_list('volume', volume_colors)
    price_colors = ['#F7FBFF', '#DEEBF7', '#C6DBEF', '#9ECAE1', '#6BAED6',
                    '#4292C6', '#2171B5', '#08519C', '#08306B']
    price_cmap = LinearSegmentedColormap.from_list('price', price_colors)

    # Panel A
    fig_a, ax_a = plt.subplots(figsize=(12, 10))
    seoul_gdf.plot(column='N', cmap=volume_cmap, linewidth=0.8, edgecolor='#333333', legend=False, ax=ax_a)
    for idx, row in seoul_gdf.iterrows():
        if pd.notna(row['N']) and pd.notna(row.geometry):
            centroid = row.geometry.centroid
            district_name = row['District'].replace('-gu', '')
            ax_a.text(centroid.x, centroid.y, district_name, fontsize=10, fontweight='bold',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))
    # STRIPPED: no title
    ax_a.set_xlim([126.76, 127.27]); ax_a.set_ylim([37.42, 37.71])
    ax_a.axis('off')
    sm_a = plt.cm.ScalarMappable(cmap=volume_cmap, norm=plt.Normalize(vmin=panel_e['N'].min(), vmax=panel_e['N'].max()))
    sm_a._A = []
    cbar_a = fig_a.colorbar(sm_a, ax=ax_a, orientation='vertical', pad=0.02, shrink=0.8)
    cbar_a.set_label('Number of Transactions', fontsize=12, fontweight='bold')
    cbar_a.ax.tick_params(labelsize=11)
    cbar_a.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.tight_layout()
    fig_a.savefig(outdir / 'figure2_panel_a_with_names.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig_a.savefig(outdir / 'figure2_panel_a_with_names.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # Panel B
    fig_b, ax_b = plt.subplots(figsize=(12, 10))
    seoul_gdf.plot(column='Mean_Price_per_sqm_Million_KRW', cmap=price_cmap, linewidth=0.8, edgecolor='#333333', legend=False, ax=ax_b)
    for idx, row in seoul_gdf.iterrows():
        if pd.notna(row['Mean_Price_per_sqm_Million_KRW']) and pd.notna(row.geometry):
            centroid = row.geometry.centroid
            district_name = row['District'].replace('-gu', '')
            ax_b.text(centroid.x, centroid.y, district_name, fontsize=10, fontweight='bold',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))
    # STRIPPED: no title
    ax_b.set_xlim([126.76, 127.27]); ax_b.set_ylim([37.42, 37.71])
    ax_b.axis('off')
    sm_b = plt.cm.ScalarMappable(cmap=price_cmap,
                                  norm=plt.Normalize(vmin=panel_e['Mean_Price_per_sqm_Million_KRW'].min(),
                                                    vmax=panel_e['Mean_Price_per_sqm_Million_KRW'].max()))
    sm_b._A = []
    cbar_b = fig_b.colorbar(sm_b, ax=ax_b, orientation='vertical', pad=0.02, shrink=0.8)
    cbar_b.set_label('Price (Million KRW/m²)', fontsize=12, fontweight='bold')
    cbar_b.ax.tick_params(labelsize=11)
    cbar_b.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₩{x:.0f}M'))
    plt.tight_layout()
    fig_b.savefig(outdir / 'figure2_panel_b_with_names.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig_b.savefig(outdir / 'figure2_panel_b_with_names.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    results["figure2"] = "OK"
    print("  Done")
except FileNotFoundError:
    pass                      # already reported as SKIPPED above
except Exception as e:
    results["figure2"] = f"ERROR: {e}"
    print(f"  ERROR: {e}")

# ======================== FIGURE 3 ========================
print("Figure 3...")
# Same source as Table 2. Reading the pre-rounded *_table2_multi_horizon.csv here while Table 2
# aggregates from the window-level file would leave two shipped sources for one quantity, differing
# by up to 5e-5 per cell. Aggregate from the window-level results in both places.
def _panel_r2(window_file: str) -> pd.DataFrame:
    w = pd.read_csv(DERIVED / window_file)
    return (w.groupby(["model", "horizon"])["oos_r2"].mean()
             .reset_index().rename(columns={"oos_r2": "R2"}))

full = _panel_r2("full_window_level_results.csv")
sub = _panel_r2("subsample_window_level_results.csv")
# Labels are canonical throughout the package; see code/_labels.py. H3 is H3g, and XGBoost is
# never abbreviated, so the figure legends read the same as Table 2.
full = canonical_models(full, "model", dedupe_on=["horizon"])
sub = canonical_models(sub, "model", dedupe_on=["horizon"])
models_full = [m for m in ["H1","H2","H3g","H4","H5","Ridge","XGBoost"] if m in set(full["model"])]
models_sub  = [m for m in ["H1","H2","H3g","H3c","H4","H5","Ridge","XGBoost"] if m in set(sub["model"])]
horizons = [1, 3, 6, 12]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, dfp, models in [
    (axes[0], full, models_full),
    (axes[1], sub, models_sub),
]:
    xp = np.arange(len(models))
    w = 0.8 / len(horizons)
    for i, h in enumerate(horizons):
        hdf = dfp[dfp["horizon"] == h].set_index("model")
        vals = [hdf.loc[m, "R2"] if m in hdf.index else np.nan for m in models]
        ax.bar(xp + (i - 1.5) * w, vals, w, label=f"h = {h}", color=HORIZON_COLORS[i],
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(xp); ax.set_xticklabels(models)
    # STRIPPED: no title, no grid, no xlabel, no ylabel
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, framealpha=0.95)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(outdir / "figure3_full_vs_subsample_split.png")
fig.savefig(outdir / "figure3_full_vs_subsample_split.pdf")
plt.close()
results["figure3"] = "OK"
print("  Done")

# ======================== FIGURE 4 ========================
print("Figure 4...")
# Figure 4 = manuscript Fig 4 (2-panel: by district + over time, AMENITY SUBSAMPLE h=1).
# The manuscript uses the subsample variant; both panels therefore read subsample sources.
df4 = pd.read_csv(DERIVED / "subsample_window_level_results.csv")
df4 = df4[df4["horizon"] == 1].copy()
df4 = canonical_models(df4, "model", dedupe_on=["test_month", "horizon"])
# The hedonic ladder as Table 2 reports it. H3c is a distinct specification (apartment-complex
# fixed effects in place of district FE, model_definitions.py L97) and is kept: it is the
# finest-grained FE competitor and the one that degrades most in some districts. H3g is NOT
# listed separately because H3 and H3g are one specification under two labels -- identical field
# for field in model_definitions.py (L68 vs L83), and identical to 0.0 in both the pooled and
# per-district results -- so both are canonicalised to H3g and drawn once.
target_models = ["H1", "H2", "H3g", "H3c", "H4", "H5", "Ridge", "XGBoost"]
df4 = df4[df4["model"].isin(target_models)].copy()
df4["date"] = pd.to_datetime(df4["test_month"])

model_colors = {
    'H1': '#CC3311', 'H2': '#EE7733', 'H3g': '#0077BB', 'H3c': '#AA4499',
    'H4': '#009988', 'H5': '#882255', 'Ridge': '#DDCC77', 'XGBoost': '#33BBEE'
}
model_markers = {
    'H1': 'o', 'H2': '^', 'H3g': 's', 'H3c': '<',
    'H4': 'D', 'H5': 'v', 'Ridge': 'P', 'XGBoost': 'X'
}

# Panel A input. This file used to be absent, which silently degraded Panel A to a bar chart of
# mean OOS R2 by model — a different figure from the one the manuscript shows. It is now produced
# by scripts/figures/gen_district_r2.py and a missing or empty file is a hard failure.
dist_r2_path = DERIVED / "subsample_district_r2_all_horizons.csv"
if not dist_r2_path.exists():
    raise FileNotFoundError(
        f"{dist_r2_path} is missing. Run code/analysis/03_district_r2.py first "
        "(run_all.py wires it ahead of this step)."
    )
dist_r2 = pd.read_csv(dist_r2_path)
# The table carries all four horizons; manuscript Figure 4 is h=1.
dist_r2 = dist_r2[dist_r2["horizon"] == 1].copy()
if len(dist_r2) == 0:
    raise ValueError(f"{dist_r2_path} is empty; Figure 4 Panel A cannot be built.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: by district
ax = axes[0]
dist_r2["gu_en"] = dist_r2["district"].map(gu_map)
xgb_order = dist_r2[dist_r2["model"] == "XGBoost"].sort_values("R2")
if len(xgb_order) == 0:
    raise ValueError(
        f"{dist_r2_path} carries no XGBoost rows, so the district ordering cannot be set."
    )
district_order = xgb_order["district"].values
x = np.arange(len(district_order))
for model in target_models:
    msub = dist_r2[dist_r2["model"] == model].set_index("district")
    vals = [msub.loc[d, "R2"] if d in msub.index else np.nan for d in district_order]
    ax.plot(x, vals, color=model_colors.get(model, "grey"),
            marker=model_markers.get(model, "o"), markersize=3,
            linewidth=0.9, label=model)
ax.set_xticks(x)
gu_labels = [gu_map.get(d, d) for d in district_order]
ax.set_xticklabels(gu_labels, rotation=45, ha="right", fontsize=6)

# One value sits far below the rest — H3c in Gangnam, where apartment-complex fixed effects
# overfit badly — and plotting down to it compresses the other seven series into the top half of
# the panel. Clip the axis and name what was clipped, so the outlier is disclosed rather than
# hidden. Everything else lies within [-0.48, 0.90].
_CLIP = -0.6
_below = dist_r2[(dist_r2["model"].isin(target_models)) & (dist_r2["R2"] < _CLIP)]
if len(_below):
    ax.set_ylim(bottom=_CLIP)
    _note = "; ".join(f"{r.model} {gu_map.get(r.district, r.district)} {r.R2:.2f}"
                      for r in _below.sort_values("R2").itertuples())
    ax.annotate(f"below axis: {_note}", xy=(0.015, 0.02), xycoords="axes fraction",
                fontsize=6, color="0.35")
# STRIPPED: no title, no ylabel, no grid
ax.axhline(0, color="grey", linestyle="--", linewidth=0.6)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(fontsize=6, ncol=2)

# Panel B: by time
ax = axes[1]
for model in target_models:
    msub = df4[df4["model"] == model].sort_values("date")
    if len(msub) > 0:
        ax.plot(msub["date"], msub["oos_r2"], color=model_colors.get(model, "grey"),
                linewidth=0.9, label=model, alpha=0.8)
# STRIPPED: no title, no ylabel, no xlabel, no grid
ax.axhline(0, color="grey", linestyle="--", linewidth=0.6)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(fontsize=6, ncol=2)
fig.autofmt_xdate()

fig.tight_layout()
fig.savefig(outdir / "figure4_appendix1_maintext_ready_subsample.png")
fig.savefig(outdir / "figure4_appendix1_maintext_ready_subsample.pdf")
plt.close()
results["figure4"] = "OK"
print("  Done")

# ======================== FIGURE 5 ========================
print("Figure 5...")
df5_raw = pd.read_csv(DERIVED / "shap_top15.csv")
df5_raw["cat"] = df5_raw["feature"].apply(categorize_feature)

loc_sum = df5_raw.loc[df5_raw["cat"] == "Location", "mean_abs_shap"].sum()
non_loc = df5_raw[df5_raw["cat"] != "Location"][["feature", "mean_abs_shap", "cat"]].copy()
if loc_sum > 0:
    loc_row = pd.DataFrame([{"feature": "District", "mean_abs_shap": loc_sum, "cat": "Location"}])
    panel_a_df = pd.concat([non_loc, loc_row], ignore_index=True)
else:
    panel_a_df = non_loc.copy()
panel_a_df = panel_a_df.sort_values("mean_abs_shap", ascending=False).head(15)
panel_a_df = panel_a_df.sort_values("mean_abs_shap", ascending=True)
panel_a_df["display"] = panel_a_df["feature"].apply(display_name)

panel_b_df = df5_raw[df5_raw["cat"] != "Location"][["feature", "mean_abs_shap", "cat"]].copy()
panel_b_df = panel_b_df.sort_values("mean_abs_shap", ascending=False).head(15)
panel_b_df = panel_b_df.sort_values("mean_abs_shap", ascending=True)
panel_b_df["display"] = panel_b_df["feature"].apply(display_name)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
for ax, pdf in [(ax1, panel_a_df), (ax2, panel_b_df)]:
    ax.barh(range(len(pdf)), pdf["mean_abs_shap"].values,
            color=[CATEGORY_PALETTE.get(c, "grey") for c in pdf["cat"]])
    ax.set_yticks(range(len(pdf)))
    ax.set_yticklabels(pdf["display"].values, fontsize=7)
    # STRIPPED: no xlabel, no title, no grid
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

used_cats = set(panel_a_df["cat"].values) | set(panel_b_df["cat"].values)
legend_order = ["Building", "Amenity", "Accessibility", "Location", "Time"]
handles = [Patch(color=CATEGORY_PALETTE[c], label=c) for c in legend_order if c in used_cats]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7, framealpha=0.9)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(outdir / "figure5_mean_shap_district_dummy.png")
fig.savefig(outdir / "figure5_mean_shap_district_dummy.pdf")
plt.close()
results["figure5"] = "OK"
print("  Done")

# ======================== FIGURE 6 ========================
print("Figure 6...")
df6 = pd.read_csv(DERIVED / "shap_district_decomp.csv")
df6["gu_en"] = df6["gu"].map(gu_map)
df6 = df6.sort_values("location_pct", ascending=False)

has_acc = "accessibility_pct" in df6.columns
if not has_acc:
    acc_shap = df5_raw.loc[df5_raw["cat"] == "Accessibility", "mean_abs_shap"].sum()
    amen_shap = df5_raw.loc[df5_raw["cat"] == "Amenity", "mean_abs_shap"].sum()
    ratio = acc_shap / (acc_shap + amen_shap) if (acc_shap + amen_shap) > 0 else 0.3
    df6["accessibility_pct"] = df6["amenity_pct"] * ratio
    df6["amenity_pct"] = df6["amenity_pct"] * (1 - ratio)

# Two panels. Panel A is the shipped six-category decomposition. Panel B repeats it with the
# district identifiers pulled out of the Location category into their own band.
#
# Why the split is needed. Inside one district the district dummy is constant, so TreeSHAP charges it
# that district's shift from the model's city-wide baseline. The Location share tracks how far a
# district sits from the city price level, and the manuscript prints that correlation as 0.83. It is
# computed below rather than asserted here, because a number quoted only in a comment is a number
# with no source. Panel B shows what is left once the level band is set aside, and those remaining
# shares do not track district price levels.
# See code/build/02_shap_bridges.py --district-out-of-location.


def _location_share_price_corr() -> float:
    """Correlation the manuscript prints for Figure 6 Panel A, recomputed from shipped data.

    Definition, stated because several defensible ones exist and they do not agree. The price
    series is the mean of log price per square metre over the h=1 test observations, by district,
    and the reference is the grand mean over those same observations rather than the mean of the
    district means. Measured across the 25 districts. Using the mean of district means as the
    reference instead gives 0.88, and using the raw file's full-sample means gives 0.84, so the
    choice matters and is fixed here.
    """
    cols = ["horizon", "y_true", "district"]
    px = pd.read_csv(DERIVED / "subsample_predictions_by_horizon.csv.gz", usecols=cols)
    px = px[px.horizon == 1]
    by_gu = px.groupby("district")["y_true"].mean()
    share = pd.read_csv(DERIVED / "shap_district_decomp.csv").set_index("gu")["location_pct"]
    joined = pd.concat([share, by_gu.rename("mean_log_price")], axis=1).dropna()
    gap = (joined["mean_log_price"] - px["y_true"].mean()).abs()
    return float(np.corrcoef(joined["location_pct"], gap)[0, 1])


_R_LOC_PRICE = _location_share_price_corr()
if abs(_R_LOC_PRICE - 0.83) > 0.005:
    raise SystemExit(f"[figures] Figure 6 self-check: the location-share/price correlation is "
                     f"{_R_LOC_PRICE:.4f}, which does not print as the manuscript's 0.83.")
print(f"[figures] Figure 6 self-check: location share vs |district mean log price - city mean| "
      f"r = {_R_LOC_PRICE:.4f} (manuscript 0.83)")
df6b = pd.read_csv(DERIVED / "shap_district_decomp_nodistrict.csv")
df6b["gu_en"] = df6b["gu"].map(gu_map)
_share = ["structural_pct", "amenity_pct", "accessibility_pct", "location_pct", "time_pct"]
df6b[_share] = df6b[_share].div(df6b[_share].sum(axis=1), axis=0) * 100  # renormalise without the level band
df6b = df6b.set_index("gu").loc[df6.set_index("gu").index].reset_index()  # same district order as Panel A

cats6 = [
    ("structural_pct", "Building", CATEGORY_PALETTE["Building"]),
    ("amenity_pct", "Amenity", CATEGORY_PALETTE["Amenity"]),
    ("accessibility_pct", "Accessibility", CATEGORY_PALETTE["Accessibility"]),
    ("location_pct", "Location", CATEGORY_PALETTE["Location"]),
    ("time_pct", "Time", CATEGORY_PALETTE["Time"]),
]

def _stack(ax, frame, cols):
    x = np.arange(len(frame))
    bottom = np.zeros(len(frame))
    for col, label, color in cols:
        vals = frame[col].values
        ax.bar(x, vals, bottom=bottom, color=color, label=label, edgecolor="white", linewidth=0.3)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(frame["gu_en"].values, rotation=45, ha="right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

for _tag, _frame, _cols in (("PanelA", df6, cats6),
                            ("PanelB", df6b, cats6)):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    _stack(ax, _frame, _cols)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / f"figure6_shap_decomp_{_tag}.png")
    fig.savefig(outdir / f"figure6_shap_decomp_{_tag}.pdf")
    plt.close()

# Superseded 2026-08-01: the single-panel figure6_shap_decomp.* is no longer written. Panel A is
# the same content; see 99_archive/2026-08-01_shap_district_decomp_pre_district_split/.
results["figure6"] = "OK"
print("  Done")

# ======================== FIGURE 7 ========================
print("Figure 7...")
# Rebased 2026-08-01 onto the district-removed decomposition, for the same reason Figure 6 gained
# Panel B. On the all-features basis the time share is mechanically what is left after the district
# dummy takes its cut, so it inherits that dummy's level effect with the sign flipped: it correlates
# -0.85 with |district mean log price - city mean|, and the three districts with the largest dummies
# (Gangnam, Seocho, Yongsan) land at the bottom of the ranking for that reason alone. On the
# district-removed basis the correlation is +0.06 and the ranking changes completely (Spearman 0.26
# between the two orderings; Yongsan moves from 3rd-lowest to 24th). The all-features version is not
# shown separately because it is the mirror image of Figure 6 Panel A. See decision_log.md.
df7 = pd.read_csv(DERIVED / "shap_district_decomp_nodistrict.csv")
df7["gu_en"] = df7["gu"].map(gu_map)
_c7 = ["structural_pct", "amenity_pct", "accessibility_pct", "location_pct", "time_pct"]
df7[_c7] = df7[_c7].div(df7[_c7].sum(axis=1), axis=0) * 100  # renormalise without the level band
df7 = df7.sort_values("time_pct", ascending=False)

fig, ax = plt.subplots(figsize=(5, 6))
ax.barh(range(len(df7)), df7["time_pct"].values, color=CATEGORY_PALETTE["Time"], edgecolor="white", linewidth=0.3)
ax.set_yticks(range(len(df7)))
ax.set_yticklabels(df7["gu_en"].values)
# STRIPPED: no xlabel, no title, no grid
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(outdir / "figure7_time_share_ranking.png")
fig.savefig(outdir / "figure7_time_share_ranking.pdf")
plt.close()
results["figure7"] = "OK"
print("  Done")

# ======================== FIGURE 8 ========================
print("Figure 8...")
df8 = pd.read_csv(DERIVED / "figure8_building_shap_by_district.csv")
df8["district_en"] = df8["district"].map(gu_map).fillna(df8["district"])
df8 = df8.set_index("district_en").drop(columns=["district"], errors="ignore")
df8 = df8.rename(columns={"area": "area_numeric"})
feature_importance = df8.mean(axis=0).sort_values(ascending=False)
df8 = df8[feature_importance.index]
df8.columns = [display_name(c) for c in df8.columns]
mat = df8.T

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(mat.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(mat.shape[1]))
ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(mat.shape[0]))
ax.set_yticklabels(mat.index, fontsize=7)
# STRIPPED: no xlabel, no ylabel (already empty), no title
plt.colorbar(im, ax=ax, shrink=0.8, label="Mean |SHAP|")
fig.tight_layout()
fig.savefig(outdir / "figure8_building_shap_heatmap.png")
fig.savefig(outdir / "figure8_building_shap_heatmap.pdf")
plt.close()
results["figure8"] = "OK"
print("  Done")

print("\n=== Results ===")
for k, v in results.items():
    print(f"  {k}: {v}")

# Figure 2 is built inside a try/except so a geopandas problem does not abort the other seven.
# That must not turn into a silent pass: any figure that did not report OK fails the step, the
# same way the Figure 4 fallback was removed.
failed = {k: v for k, v in results.items()
          if v != "OK" and not str(v).startswith("SKIPPED")}
if failed or len(results) != 8:
    print("\nFAILED figures:", failed or "(some figures never ran)")
    raise SystemExit(1)
print("All done!")
