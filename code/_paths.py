"""Repository paths resolved relative to this file.

Required inputs and their locations are listed in data/README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DERIVED_DATA_DIR = DATA_DIR / "derived"
MODEL_DIR = DATA_DIR / "model"
OUTPUT_DIR = PROJECT_ROOT / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_TRANSACTIONS = RAW_DATA_DIR / "seoul_apartment_transactions_with_building_info.csv"
RAW_GEOJSON = DATA_DIR / "geo" / "seoul_districts_2017.geojson"
FROZEN_MODEL = MODEL_DIR / "xgb_window_72.pkl"

for _d in (TABLES_DIR, FIGURES_DIR, DERIVED_DATA_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def require(*paths: Path) -> None:
    """Register input paths in the run log and exit if any required file is missing."""
    try:
        from _runlog import register_input
        for p in paths:
            register_input(p)
    except ImportError:
        pass
    missing = [p for p in paths if not p.exists()]
    if not missing:
        return
    print("MISSING INPUT — this step cannot run:")
    for p in missing:
        try:
            shown = p.relative_to(PROJECT_ROOT)
        except ValueError:
            shown = p
        print(f"    {shown}")
    if any(RAW_DATA_DIR in p.parents for p in missing):
        print(
            "\nSupply these input files in data/raw/.\n"
            "See data/README.md for the required input layout."
        )
    sys.exit(2)


def engine_path() -> Path:
    """Return the directory containing the estimation engine modules."""
    return CODE_DIR / "engine"
