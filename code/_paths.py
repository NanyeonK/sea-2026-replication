"""Single place where every path in this replication package is defined.

Every script imports from here, so the deposit can be moved or renamed without editing code.
Paths resolve from this file's own location, never from the working directory.

Input files are supplied separately. See data/README.md.
No data, fitted models, predictions, or generated outputs are committed.
"""
from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]

CODE = PKG / "code"
DATA = PKG / "data"
RAW = DATA / "raw"
DERIVED = DATA / "derived"
MODEL = DATA / "model"
OUTPUT = PKG / "output"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
DOCS = PKG / "docs"

# Named inputs, so a rename is a one-line change here rather than a grep across the package.
RAW_TRANSACTIONS = RAW / "seoul_apartment_transactions_with_building_info.csv"
RAW_GEOJSON = DATA / "geo" / "seoul_districts_2017.geojson"
FROZEN_MODEL = MODEL / "xgb_window_72.pkl"

for _d in (TABLES, FIGURES, DERIVED, DOCS):
    _d.mkdir(parents=True, exist_ok=True)


def require(*paths: Path) -> None:
    """Abort with an actionable message rather than a stack trace deep inside pandas.

    Also registers each declared input with the run log, which is how the README's
    "input file paths with SHA256" gets satisfied without threading calls through every script.
    """
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
            shown = p.relative_to(PKG)
        except ValueError:
            shown = p
        print(f"    {shown}")
    if any(RAW in p.parents for p in missing):
        print(
            "\nThese live in data/raw/, which is not shipped with the package.\n"
            "See data/README.md for the required input layout."
        )
    sys.exit(2)


def engine_path() -> Path:
    """Vendored engine modules, added to sys.path by the scripts that need them."""
    return CODE / "engine"
