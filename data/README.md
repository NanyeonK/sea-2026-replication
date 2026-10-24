# Data access and input files

## Public sources

The study uses Seoul apartment sales from June 2015 to May 2025 and complex-level building attributes. Access the source data through Korea's public data portal:

- [MOLIT apartment sale transactions](https://www.data.go.kr/data/15126469/openapi.do)
- [K-APT apartment complex information](https://www.data.go.kr/data/15058453/openapi.do)

API access is free and requires registration and an API key. Consult each provider for current terms and documentation. Keep API keys outside the repository.

The transaction register can change after contracts are cancelled, corrected or reported late. A fresh download for the same period may differ from the extract used in the study.

## Input for main estimation

Save the prepared, merged transaction dataset as:

```text
data/raw/seoul_apartment_transactions_with_building_info.csv
```

The code expects the study's merged input schema. Prepare the API responses in that schema before estimation. The original collection and record-linkage scripts are outside this code release. Contact the corresponding author for those scripts or the prepared input file.

Key fields read by the code include:

| Fields | Meaning |
|---|---|
| `transaction_type` | Transaction category; apartment sales are `매매` |
| `contract_ym` | Contract month as `YYYYMM` |
| `시군구` | Korean city and district name, used to derive `district` |
| `price_per_sqm` | Price per square metre in KRW; the target is its natural logarithm |
| `price_numeric` | Total transaction price in units of KRW 10,000, used by Table 1 |
| `area_numeric`, `floor`, `build_year`, `building_age` | Unit and building characteristics |
| `total_parking`, `elevator_count`, `cctv_count`, `edu_count` | Complex amenities used by the hedonic specifications |
| `has_parking`, `has_elevator`, `has_subway` | Binary amenity indicators |

This table identifies key fields; the original merged schema contains additional predictors. Field handling and feature exclusions are defined in `code/engine/data_preprocessor.py`. Supply the original schema when reproducing the reported models.

## Inputs for the exhibit scripts

The exhibit scripts require the prepared inputs below. These files are supplied separately. Running the main estimation commands does not automatically create this intermediate file layout.

```text
data/derived/
  full_table2_multi_horizon.csv
  subsample_table2_multi_horizon.csv
  full_window_level_results.csv
  subsample_window_level_results.csv
  dm_tests_hac.csv
  table3_all_pairwise_dm_hac_full.csv
  table3_all_pairwise_dm_hac_subsample.csv
  subsample_predictions_by_horizon.csv.gz
  table1_panel_e_spatial_all25_english.csv
  shap_top15.csv
  shap_district_decomp.csv
  shap_district_decomp_nodistrict.csv
  figure8_building_shap_by_district.csv
  window_sensitivity.csv
  draw_sensitivity.csv
  appendix_c_window_level.csv
data/model/
  xgb_window_0.pkl
  xgb_window_12.pkl
  xgb_window_24.pkl
  xgb_window_36.pkl
  xgb_window_48.pkl
  xgb_window_60.pkl
  xgb_window_72.pkl
data/geo/
  seoul_districts_2017.geojson
```

SHAP rebuilding requires the fitted models. Each pickle contains the model and feature metadata expected by the build scripts. Figure 2 uses the separately supplied GeoJSON. The exact boundary edition and prepared input files can be requested from the corresponding author.

The default exhibit run derives `subsample_district_r2_all_horizons.csv` from the prediction file. The build option regenerates Table 1, the main SHAP summaries, draw sensitivity and Appendix C inputs. It still needs the other prepared inputs above. The existing numerical assertions compare selected outputs with the study's reference values and may fail on a different data vintage.
