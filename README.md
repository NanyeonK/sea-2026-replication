# Seoul housing valuation: analysis code

**One city, multiple markets: District-specific housing valuations using interpretable machine learning**  
Yeonchan Kang and Doojin Ryu — *Spatial Economic Analysis*

This repository contains the Python code for the study: data preprocessing, rolling-window estimation, tables, figures, SHAP attribution and appendix analyses. Data, predictions, fitted models and generated exhibits are supplied separately. See [data/README.md](data/README.md) for data sources and the required input layout.

## Setup

Use Python 3.12 and the package versions recorded in `requirements.txt`.

```bash
git clone https://github.com/NanyeonK/sea-2026-replication.git
cd sea-2026-replication
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The estimation code can use a CPU. GPU detection uses an optional PyTorch installation. Runtime depends on the data, hardware and estimation options.

## Main estimation

Prepare the merged transaction file described in [data/README.md](data/README.md), then run each sample from the repository root:

```bash
python code/estimation/main.py --sample-type full --no-shap
python code/estimation/main.py --sample-type subsample --no-shap
```

These commands write predictions, performance summaries and fitted models under `output/rnr/full/` and `output/rnr/subsample/`. The entrypoint uses the same engine modules as the replication archive. Settings remain in `code/estimation/config/analysis_config.yaml` and `code/engine/model_definitions.py`. The XGBoost refit interval defaults to 12 windows, and the engine retains random seed 42.

The `--no-shap` option skips the earlier per-window SHAP routine. The manuscript's attribution figures use the seven-refit averaging code in `code/build/02_shap_bridges.py`.

## Tables, figures and appendices

The exhibit scripts use prepared intermediate CSVs, prediction files and fitted models. Place those files in the locations listed in [data/README.md](data/README.md) before running:

```bash
python run_all.py --list       # list steps without running them
python run_all.py              # produce exhibits from prepared inputs
python run_all.py --with-build # also rebuild Table 1, SHAP inputs and Appendix C
```

The build option needs the merged transaction file and fitted models. Figure 2 needs the district boundary GeoJSON. If that file is unavailable, rendering the map is skipped; the figure-packaging step then needs the existing Figure 2 outputs from the replication archive. The scripts write outputs to `output/tables/` and `output/figures/manuscript/`.

| Component | Code |
|---|---|
| Preprocessing, specifications, rolling windows and evaluation | `code/engine/` |
| Main model estimation | `code/estimation/main.py` |
| Table 1 | `code/build/01_table1_panels.py` |
| Tables 2 and 3 | `code/analysis/01_table2.py`, `02_table3.py` |
| Figures 1–8 | `code/analysis/03_district_r2.py` through `05_package_figures.py` |
| SHAP attribution and draw sensitivity | `code/build/02_shap_bridges.py`, `03_draw_sensitivity.py` |
| Appendix C estimation | `code/build/04_appendix_c.py` |
| Appendix exhibit formatting | `code/analysis/06_window_sensitivity.py` through `08_appendix_c.py` |

## Version and scope

The exhibit code and engine come from the replication archive dated 6 August 2026. The main estimation entrypoint and its output helpers were added from the research source, with package-local imports and separate output directories for the two samples. The engine retains the August 6 correction excluding the reference district's area interaction in H5. The Table 1 generator now calculates cumulative percentages from counts before rounding, matching the production correction.

This publication supplies code and access instructions. It does not include an automated download-and-link pipeline, and a fresh checkout alone cannot recreate the exhibits. Exact reproduction needs the study's prepared inputs; fresh public-data downloads may differ because the transaction register is revised retrospectively. The publication preparation did not rerun estimation or certify end-to-end reproduction. Comments about “shipped” inputs refer to the original replication archive.

## License and contact

The code uses the [MIT terms](LICENSE). Data remain subject to their providers' terms. For the study's prepared inputs or questions about the code, contact the corresponding author, Doojin Ryu, at [sharpjin@skku.edu](mailto:sharpjin@skku.edu).

This work was supported by a grant from the National Research Foundation of Korea (NRF), funded by the Korean government (MSIT) [RS-2025-00518388].
