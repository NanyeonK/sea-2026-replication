# Implementation and reporting notes

Updated 25 October 2026. These notes distinguish the implemented specifications from broader interpretations of their labels. They do not report a new estimation run.

## Reporting corrections

Table 3 now preserves the sign of MSE(linear) minus MSE(XGBoost), reversing the stored DM statistic only when XGBoost is Model_1. The appendix H6 comparison uses the same direction. Previously, the displays used absolute values. All 52 preserved XGBoost comparisons for Table 3 already have positive statistics in the stated direction, so this correction does not alter the reported entries or significance marks.

The default DM comparison list now comes from models with stored forecast errors, including H5 and H3c when available and loaded XGBoost predictions. It drops the duplicate H3 label when H3g is also present. Table C1's generator distinguishes dummy columns from numerical categorical inputs and reports 24 area-by-district terms for H5/H6 and 25 for XGBoost. To match the final manuscript, it groups building characteristics under “Structural attributes” without a separate construction-year row. This presentation change does not alter model inputs. Counts are design columns or inputs, not necessarily separately identified coefficients.

## Sample and inputs

- The amenity-complete subsample requires non-missing total parking and elevator counts. All six audited numerical amenity counts are observed in the 261,049 study observations; some raw accessibility strings remain missing. Both full and subsample specifications receive amenity inputs where specified.
- In the preserved study extract, `building_age` equals 2025 minus `build_year`, not transaction year minus construction year. The loader preserves this existing field.
- H3c selects the 200 most frequent complex-name categories on its first training call and retains that set thereafter, pooling other names as the reference. Names need not uniquely identify physical complexes. The preserved historical code has this rule; an original per-window H3c design-column manifest was not retained.
- XGBoost retains the top 100 levels of each encoded categorical input and pools the remainder before numerical encoding. H6's 241 dong indicators are a different representation of the locality field. The seven preserved XGBoost models contain 61 inputs, including construction year and 25 area-by-district terms.

## Information set and attribution

Category mappings use full-sample covariate frequencies, and numerical missing values use medians of the batch being transformed. This differs from preprocessing fitted exclusively on each training window. Its effect on reported accuracy has not been quantified by a training-only rerun; the exercise should not be described as a strictly real-time forecasting simulation.

SHAP uses the same seed-42 draw of 5,000 observations from the final tenth of the amenity sample in stored file order for seven annual refits. Stored row order is not strictly chronological. Each annual fitted model supplies all four forecast horizons.

In Figure 5, the existing label “District” aggregates gu and recorded dong identifiers, area-by-gu interactions and the road-name entry present in the saved leading-feature list. Its displayed value is 0.3065444396; the first three components alone total 0.29922786. It is not an exhaustive sum of every location input. Figures 6B and 7 exclude gu and dong identifiers and area-by-gu interactions, then renormalise existing mean absolute SHAP summaries without refitting. Figure 8 displays 12 selected features. Absolute SHAP shares describe attribution magnitude, not components of the predicted price or marginal effects.

The Yongsan/Nowon price ranks of third/twenty-first use mean log price in the h=1 amenity evaluation observations. They need not equal the ranks of full-sample arithmetic mean prices in Table 1.

## Reproduction scope

This GitHub repository contains code and instructions. Prepared data, model files and outputs are not included. Missing inputs in a code-only checkout do not establish a numerical error, and the availability of preserved inputs does not establish a completed independent raw-to-paper replication. Source-provider terms and the boundary-file provenance remain separate from the code's MIT license. No new model fitting or full reproduction was performed for these reporting corrections.
