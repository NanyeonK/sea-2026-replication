"""Standardize model labels for tables and figures.

H3 and H3g denote the same district-effects specification; use H3g in exhibits.
H3c denotes the separate apartment-complex-effects specification.
"""

from __future__ import annotations
import pandas as pd

CANONICAL = {"H3": "H3g", "XGB": "XGBoost"}


def standardize_model_labels(
    df: pd.DataFrame, col: str, dedupe_on: list[str] | None = None
) -> pd.DataFrame:
    """Standardize model labels and optionally remove duplicate model rows.

    Use dedupe_on to identify a row within each model, for example ["Horizon"].
    """
    out = df.copy()
    out[col] = out[col].replace(CANONICAL)
    if dedupe_on is not None:
        before = len(out)
        out = out.drop_duplicates(subset=[col] + list(dedupe_on), keep="first")
        if before != len(out):
            print(f"[labels] collapsed {before - len(out)} duplicate H3/H3g row(s) in '{col}'")
    return out.reset_index(drop=True)
