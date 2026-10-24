# Seoul housing valuation analysis code

**One city, multiple markets: District-specific housing valuations using interpretable machine learning**  
Yeonchan Kang and Doojin Ryu — *Spatial Economic Analysis*

Python code for data preprocessing, rolling-window estimation, SHAP analysis, tables and figures. Source data are available through Korea's public data portal. Required files and preparation details are listed in [data/README.md](data/README.md).

## Installation

Use Python 3.12 with the versions in `requirements.txt`.

```bash
git clone https://github.com/NanyeonK/sea-2026-replication.git
cd sea-2026-replication
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Estimation supports CPU execution. GPU detection uses an optional PyTorch installation.

## Model estimation

Place the prepared transaction file in `data/raw/`, then run:

```bash
python code/estimation/main.py --sample-type full --no-shap
python code/estimation/main.py --sample-type subsample --no-shap
```

Results are written to `output/estimation/full/` and `output/estimation/subsample/`. Settings are defined in `code/estimation/config/analysis_config.yaml` and `code/engine/model_definitions.py`. XGBoost is refitted every 12 windows by default, with random seed 42.

The `--no-shap` option skips per-window SHAP calculations. Attribution figures use the seven-model averaging procedure in `code/build/02_shap_bridges.py`.

## Tables and figures

Supply the intermediate inputs listed in [data/README.md](data/README.md), then run:

```bash
python run_all.py --list       # list available steps
python run_all.py              # generate exhibits from prepared inputs
python run_all.py --with-build # rebuild intermediate inputs and generate exhibits
```

The build steps require merged transactions and fitted models. They generate both SHAP category decompositions, including the version that separates district identifiers and their interactions from the location category for Figures 6B and 7. Figure 2 requires district boundaries or separately supplied map images. Outputs are written to `output/tables/` and `output/figures/manuscript/`; execution logs are written to `output/logs/`.

| Component | Location |
|---|---|
| Preprocessing, models and rolling windows | `code/engine/` |
| Main estimation | `code/estimation/` |
| Table 1, SHAP summaries and appendix estimation | `code/build/` |
| Tables, figures and appendix formatting | `code/analysis/` |

## Data and reproduction

Data, fitted models and generated outputs are supplied separately. This repository requires prepared inputs and does not include data collection or record-linkage scripts. Estimation outputs also require intermediate preparation before the exhibit scripts can use them. See [data/README.md](data/README.md) for file locations and contact details. Retrospective revisions to public transaction records may affect results from a new download.

## Code authorship and AI assistance

The authors wrote the original code for this repository and the analyses reported in the article. GPT-5 was used to assist with code revision and organization. The authors take full responsibility for the work.

## License and contact

Code is released under the [MIT License](LICENSE). Data are subject to their providers' terms. For prepared study inputs or questions, contact Doojin Ryu at [sharpjin@skku.edu](mailto:sharpjin@skku.edu).

This work was supported by a grant from the National Research Foundation of Korea (NRF), funded by the Korean government (MSIT) [RS-2025-00518388].
