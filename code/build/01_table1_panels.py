#!/usr/bin/env python3
"""Generate Table 1 descriptive statistics from the merged transaction data."""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: E402, F401
from _paths import TRANSACTIONS_FILE, TABLES_DIR, require  # noqa: E402

warnings.filterwarnings("ignore")

require(TRANSACTIONS_FILE)

print("=" * 80)
print("Generating Table 1 panels A-G")
print("Applying transaction filters")
print("=" * 80)

print("\nLoading and filtering data...")
df = pd.read_csv(TRANSACTIONS_FILE)
print(f"Raw data: {len(df):,} rows")

df = df[df["transaction_type"] == "매매"].copy()
print(f"After filtering for sales (매매): {len(df):,} rows")

df["date"] = pd.to_datetime(df["contract_ym"], format="%Y%m", errors="coerce")
df = df.dropna(subset=["date"])
print(f"After filtering valid dates: {len(df):,} rows")

valid_mask = (df["price_per_sqm"] > 0) & (~df["price_per_sqm"].isna())
df = df[valid_mask].copy()
print(f"After filtering valid price_per_sqm: {len(df):,} rows")

df["log_price_per_sqm"] = np.log(df["price_per_sqm"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df["year_month"] = df["date"].dt.strftime("%Y-%m")

if "시군구" in df.columns:
    df["district"] = df["시군구"].str.extract(r"서울특별시\s+(\S+구)")

if "building_age" not in df.columns and "건축년도" in df.columns:
    df["building_age"] = df["year"] - df["건축년도"]

print(f"\nAnalysis sample: {len(df):,} observations")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

print("\n" + "=" * 60)
print("PANEL A: SAMPLE OVERVIEW")
print("=" * 60)

panel_a_data = {
    "Metric": [
        "Total Transactions",
        "Sample Period",
        "Start Date",
        "End Date",
        "Number of Years",
        "Number of Districts",
        "Number of Buildings",
        "Number of Unique Addresses",
        "Transaction Type",
        "Data Completeness",
    ],
    "Value": [
        f"{len(df):,}",
        f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}",
        df["date"].min().strftime("%Y-%m-%d"),
        df["date"].max().strftime("%Y-%m-%d"),
        f"{(df['date'].max() - df['date'].min()).days / 365.25:.1f}",
        str(df["district"].nunique()),
        str(df["단지명"].nunique() if "단지명" in df.columns else "N/A"),
        str(df["도로명"].nunique() if "도로명" in df.columns else "N/A"),
        "Sales (매매) only",
        f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))):.1%}",
    ],
    "Description": [
        "Total unique apartment sales transactions",
        "Full temporal coverage",
        "First transaction date",
        "Last transaction date",
        "Total years covered",
        "Seoul administrative districts",
        "Unique apartment complexes",
        "Unique street addresses",
        "Transaction type filter applied",
        "Percentage of non-null values",
    ],
}

panel_a = pd.DataFrame(panel_a_data)
panel_a.to_csv(TABLES_DIR / "table1_panel_a_sample.csv", index=False)
print(panel_a.to_string(index=False))

print("\n" + "=" * 60)
print("PANEL B: PRICE VARIABLES")
print("=" * 60)

price_stats = pd.DataFrame(
    {
        "Total Price (₩M)": df["price_numeric"] / 100,
        "Price per m² (₩M)": df["price_per_sqm"] / 1000000,
        "Log(Price per m²)": df["log_price_per_sqm"],
    }
).describe()

price_cv = pd.DataFrame(
    {
        "Total Price (₩M)": [
            price_stats.loc["std", "Total Price (₩M)"] / price_stats.loc["mean", "Total Price (₩M)"]
        ],
        "Price per m² (₩M)": [
            price_stats.loc["std", "Price per m² (₩M)"]
            / price_stats.loc["mean", "Price per m² (₩M)"]
        ],
        "Log(Price per m²)": [
            price_stats.loc["std", "Log(Price per m²)"]
            / price_stats.loc["mean", "Log(Price per m²)"]
        ],
    },
    index=["cv"],
)

price_stats = pd.concat([price_stats, price_cv])

panel_b_data = {
    "Variable": ["Total Price", "Price per m²", "Log(Price per m²)"],
    "Mean": [
        f"₩{price_stats.loc['mean', 'Total Price (₩M)']:.1f}M",
        f"₩{price_stats.loc['mean', 'Price per m² (₩M)']:.2f}M",
        f"{price_stats.loc['mean', 'Log(Price per m²)']:.3f}",
    ],
    "Std Dev": [
        f"₩{price_stats.loc['std', 'Total Price (₩M)']:.1f}M",
        f"₩{price_stats.loc['std', 'Price per m² (₩M)']:.2f}M",
        f"{price_stats.loc['std', 'Log(Price per m²)']:.3f}",
    ],
    "Min": [
        f"₩{price_stats.loc['min', 'Total Price (₩M)']:.0f}M",
        f"₩{price_stats.loc['min', 'Price per m² (₩M)']:.2f}M",
        f"{price_stats.loc['min', 'Log(Price per m²)']:.3f}",
    ],
    "Q1": [
        f"₩{price_stats.loc['25%', 'Total Price (₩M)']:.0f}M",
        f"₩{price_stats.loc['25%', 'Price per m² (₩M)']:.2f}M",
        f"{price_stats.loc['25%', 'Log(Price per m²)']:.3f}",
    ],
    "Median": [
        f"₩{price_stats.loc['50%', 'Total Price (₩M)']:.0f}M",
        f"₩{price_stats.loc['50%', 'Price per m² (₩M)']:.2f}M",
        f"{price_stats.loc['50%', 'Log(Price per m²)']:.3f}",
    ],
    "Q3": [
        f"₩{price_stats.loc['75%', 'Total Price (₩M)']:.0f}M",
        f"₩{price_stats.loc['75%', 'Price per m² (₩M)']:.2f}M",
        f"{price_stats.loc['75%', 'Log(Price per m²)']:.3f}",
    ],
    "Max": [
        f"₩{price_stats.loc['max', 'Total Price (₩M)']:.0f}M",
        f"₩{price_stats.loc['max', 'Price per m² (₩M)']:.2f}M",
        f"{price_stats.loc['max', 'Log(Price per m²)']:.3f}",
    ],
    "CV": [
        f"{price_stats.loc['cv', 'Total Price (₩M)']:.3f}",
        f"{price_stats.loc['cv', 'Price per m² (₩M)']:.3f}",
        f"{price_stats.loc['cv', 'Log(Price per m²)']:.3f}",
    ],
}

panel_b = pd.DataFrame(panel_b_data)
panel_b.to_csv(TABLES_DIR / "table1_panel_b_prices.csv", index=False)
print(panel_b.to_string(index=False))

print("\n" + "=" * 60)
print("PANEL C: BUILDING CHARACTERISTICS")
print("=" * 60)

area_col = "area_numeric" if "area_numeric" in df.columns else "전용면적"
floor_col = "floor" if "floor" in df.columns else "층"

building_stats = pd.DataFrame(
    {
        "Area (m²)": df[area_col],
        "Floor": df[floor_col],
        "Building Age (years)": df["building_age"],
        "Year Built": df["건축년도"]
        if "건축년도" in df.columns
        else df["year"] - df["building_age"],
    }
).describe()

panel_c_data = {
    "Variable": ["Area (m²)", "Floor", "Building Age", "Year Built"],
    "Mean": [
        f"{building_stats.loc['mean', 'Area (m²)']:.1f}",
        f"{building_stats.loc['mean', 'Floor']:.1f}",
        f"{building_stats.loc['mean', 'Building Age (years)']:.1f}",
        f"{building_stats.loc['mean', 'Year Built']:.0f}",
    ],
    "Std Dev": [
        f"{building_stats.loc['std', 'Area (m²)']:.1f}",
        f"{building_stats.loc['std', 'Floor']:.1f}",
        f"{building_stats.loc['std', 'Building Age (years)']:.1f}",
        f"{building_stats.loc['std', 'Year Built']:.1f}",
    ],
    "Min": [
        f"{building_stats.loc['min', 'Area (m²)']:.0f}",
        f"{building_stats.loc['min', 'Floor']:.0f}",
        f"{building_stats.loc['min', 'Building Age (years)']:.0f}",
        f"{building_stats.loc['min', 'Year Built']:.0f}",
    ],
    "Q1": [
        f"{building_stats.loc['25%', 'Area (m²)']:.1f}",
        f"{building_stats.loc['25%', 'Floor']:.0f}",
        f"{building_stats.loc['25%', 'Building Age (years)']:.0f}",
        f"{building_stats.loc['25%', 'Year Built']:.0f}",
    ],
    "Median": [
        f"{building_stats.loc['50%', 'Area (m²)']:.1f}",
        f"{building_stats.loc['50%', 'Floor']:.0f}",
        f"{building_stats.loc['50%', 'Building Age (years)']:.0f}",
        f"{building_stats.loc['50%', 'Year Built']:.0f}",
    ],
    "Q3": [
        f"{building_stats.loc['75%', 'Area (m²)']:.1f}",
        f"{building_stats.loc['75%', 'Floor']:.0f}",
        f"{building_stats.loc['75%', 'Building Age (years)']:.0f}",
        f"{building_stats.loc['75%', 'Year Built']:.0f}",
    ],
    "Max": [
        f"{building_stats.loc['max', 'Area (m²)']:.0f}",
        f"{building_stats.loc['max', 'Floor']:.0f}",
        f"{building_stats.loc['max', 'Building Age (years)']:.0f}",
        f"{building_stats.loc['max', 'Year Built']:.0f}",
    ],
}

panel_c = pd.DataFrame(panel_c_data)
panel_c.to_csv(TABLES_DIR / "table1_panel_c_building.csv", index=False)
print(panel_c.to_string(index=False))

print("\n" + "=" * 60)
print("PANEL D: TEMPORAL DISTRIBUTION")
print("=" * 60)

temporal_dist = df.groupby("year").size().reset_index(name="count")
temporal_dist["percentage"] = (temporal_dist["count"] / len(df) * 100).round(1)

quarterly_dist = df.groupby("quarter").size()
quarterly_avg = quarterly_dist.mean()

monthly_dist = df.groupby("month").size()

panel_d_data = {"Period": [], "Transactions": [], "Percentage": [], "Cumulative %": []}

cumulative_count = 0
for _, row in temporal_dist.iterrows():
    cumulative_count += row["count"]
    cumulative = cumulative_count / len(df) * 100
    panel_d_data["Period"].append(str(row["year"]))
    panel_d_data["Transactions"].append(f"{row['count']:,}")
    panel_d_data["Percentage"].append(f"{row['percentage']:.1f}%")
    panel_d_data["Cumulative %"].append(f"{cumulative:.1f}%")

panel_d = pd.DataFrame(panel_d_data)
panel_d.to_csv(TABLES_DIR / "table1_panel_d_temporal.csv", index=False)
print(panel_d.to_string(index=False))

print("\n" + "=" * 60)
print("PANEL E: SPATIAL DISTRIBUTION")
print("=" * 60)

district_stats = (
    df.groupby("district")
    .agg({"price_per_sqm": ["count", "mean", "std"], "log_price_per_sqm": ["mean", "std"]})
    .round(0)
)

district_stats.columns = ["Count", "Mean Price/m²", "SD Price/m²", "Mean Log Price", "SD Log Price"]
district_stats["CV"] = (district_stats["SD Price/m²"] / district_stats["Mean Price/m²"]).round(3)

district_stats = district_stats.sort_values("Count", ascending=False)

panel_e_data = {
    "District": [],
    "N": [],
    "Mean Price/m² (₩M)": [],
    "SD": [],
    "CV": [],
    "Share (%)": [],
}

DISTRICT_NAMES_EN = {
    "강남구": "Gangnam",
    "강동구": "Gangdong",
    "강북구": "Gangbuk",
    "강서구": "Gangseo",
    "관악구": "Gwanak",
    "광진구": "Gwangjin",
    "구로구": "Guro",
    "금천구": "Geumcheon",
    "노원구": "Nowon",
    "도봉구": "Dobong",
    "동대문구": "Dongdaemun",
    "동작구": "Dongjak",
    "마포구": "Mapo",
    "서대문구": "Seodaemun",
    "서초구": "Seocho",
    "성동구": "Seongdong",
    "성북구": "Seongbuk",
    "송파구": "Songpa",
    "양천구": "Yangcheon",
    "영등포구": "Yeongdeungpo",
    "용산구": "Yongsan",
    "은평구": "Eunpyeong",
    "종로구": "Jongno",
    "중구": "Jung",
    "중랑구": "Jungnang",
}
_unmapped = [d for d in district_stats.index if d not in DISTRICT_NAMES_EN]
if _unmapped:
    raise SystemExit(f"Panel E: no English name for {_unmapped}; add it to DISTRICT_NAMES_EN.")

for district, row in district_stats.iterrows():
    panel_e_data["District"].append(DISTRICT_NAMES_EN[district])
    panel_e_data["N"].append(f"{int(row['Count']):,}")
    panel_e_data["Mean Price/m² (₩M)"].append(f"{row['Mean Price/m²'] / 1000000:.1f}")
    panel_e_data["SD"].append(f"{row['SD Price/m²'] / 1000000:.1f}")
    panel_e_data["CV"].append(f"{row['CV']:.2f}")
    panel_e_data["Share (%)"].append(f"{row['Count'] / len(df) * 100:.1f}")

panel_e = pd.DataFrame(panel_e_data)
panel_e.to_csv(TABLES_DIR / "table1_panel_e_spatial.csv", index=False)
print(panel_e.to_string(index=False))

print("\n" + "=" * 60)
print("PANEL F: CATEGORICAL DISTRIBUTION")
print("=" * 60)

age_bins = [0, 5, 10, 15, 20, 25, 30, 100]
age_labels = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "30+"]
df["age_category"] = pd.cut(df["building_age"], bins=age_bins, labels=age_labels)

size_bins = [0, 60, 85, 115, 500]
size_labels = ["Small (<60m²)", "Medium (60-85m²)", "Large (85-115m²)", "Extra Large (>115m²)"]
df["size_category"] = pd.cut(df[area_col], bins=size_bins, labels=size_labels)

floor_bins = [-10, 0, 5, 10, 15, 20, 100]
floor_labels = ["Underground", "1-5F", "6-10F", "11-15F", "16-20F", "20+F"]
df["floor_category"] = pd.cut(df[floor_col], bins=floor_bins, labels=floor_labels)

panel_f_data = {"Category": [], "Group": [], "N": [], "Percentage": [], "Mean Price/m² (₩M)": []}

for cat in age_labels:
    cat_data = df[df["age_category"] == cat]
    if len(cat_data) > 0:
        panel_f_data["Category"].append("Building Age")
        panel_f_data["Group"].append(f"{cat} years")
        panel_f_data["N"].append(f"{len(cat_data):,}")
        panel_f_data["Percentage"].append(f"{len(cat_data) / len(df) * 100:.1f}%")
        panel_f_data["Mean Price/m² (₩M)"].append(
            f"{cat_data['price_per_sqm'].mean() / 1000000:.1f}"
        )

for cat in size_labels:
    cat_data = df[df["size_category"] == cat]
    if len(cat_data) > 0:
        panel_f_data["Category"].append("Size")
        panel_f_data["Group"].append(cat)
        panel_f_data["N"].append(f"{len(cat_data):,}")
        panel_f_data["Percentage"].append(f"{len(cat_data) / len(df) * 100:.1f}%")
        panel_f_data["Mean Price/m² (₩M)"].append(
            f"{cat_data['price_per_sqm'].mean() / 1000000:.1f}"
        )

panel_f = pd.DataFrame(panel_f_data)
panel_f.to_csv(TABLES_DIR / "table1_panel_f_categorical.csv", index=False)
print(panel_f.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("PANEL G: CORRELATION MATRIX")
print("=" * 60)

corr_vars = {
    "Log(Price/m²)": df["log_price_per_sqm"],
    "Area (m²)": df[area_col],
    "Floor": df[floor_col],
    "Building Age": df["building_age"],
    "Year": df["year"],
    "Month": df["month"],
}

corr_df = pd.DataFrame(corr_vars)
corr_matrix = corr_df.corr().round(3)

panel_g = corr_matrix.copy()
panel_g.to_csv(TABLES_DIR / "table1_panel_g_correlations.csv")
print(panel_g.to_string())

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\nSummary statistics:")
print(f"  • Total transactions: {len(df):,}")
print(f"  • Mean transaction price: ₩{df['price_numeric'].mean() / 100:.1f}M")
print(f"  • Mean price per m²: ₩{df['price_per_sqm'].mean() / 1000000:.2f}M")
print(f"  • Mean log(price/m²): {df['log_price_per_sqm'].mean():.3f}")
print(f"  • Mean area: {df[area_col].mean():.1f}m²")
print(f"  • Mean building age: {df['building_age'].mean():.1f} years")

total_var = df["log_price_per_sqm"].var()
district_means = df.groupby("district")["log_price_per_sqm"].mean()
df["district_mean"] = df["district"].map(district_means)
between_var = ((df["district_mean"] - df["log_price_per_sqm"].mean()) ** 2).mean()
within_var = total_var - between_var

print("\nVariance Decomposition:")
print(f"  • Total variance: {total_var:.4f}")
print(f"  • Between-district: {between_var:.4f} ({between_var / total_var * 100:.1f}%)")
print(f"  • Within-district: {within_var:.4f} ({within_var / total_var * 100:.1f}%)")

print("\nAll panels saved successfully:")
print("  • table1_panel_a_sample.csv")
print("  • table1_panel_b_prices.csv")
print("  • table1_panel_c_building.csv")
print("  • table1_panel_d_temporal.csv")
print("  • table1_panel_e_spatial.csv")
print("  • table1_panel_f_categorical.csv")
print("  • table1_panel_g_correlations.csv")

print("\n" + "=" * 80)
print("Table 1 generation complete")
print("=" * 80)
