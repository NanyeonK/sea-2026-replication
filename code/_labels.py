"""Canonical model labels.

Two labels in the estimation output name one specification, and one label differs between the
estimation output and the manuscript. This module fixes both at the point tables are built, so the
raw window-level files keep the engine's own record and every exhibit prints one name.

**H3 and H3g are the same specification.** `model_definitions.py` gives them identical fields —
same continuous, temporal, binary and count variables, `district: True` and `month: True` in both,
47 expected coefficients in both — and H3g's own description reads "H3 with explicit gu (district)
fixed effects -- subsample label". The engine's subsample list is
`['H1','H2','H3','H3g','H3c','H5']`, so it estimates that specification twice; H3 and H3g are
bit-identical across all 318 window-by-horizon observations in `subsample_window_level_results.csv`,
difference exactly 0.0. The manuscript's Table 2 prints H3g in both panels, so H3g is the surviving
label and the duplicate row is dropped.

**H3c is NOT the same and is never merged.** It carries apartment-complex fixed effects in place of
district fixed effects (`'district': False, 'apt_complex': True`). The H3g/H3c pair exists so the
subsample panel reads as a gu-FE against complex-FE contrast, which is the comparison the second
revision was asked for. Collapsing it would remove that contrast from the paper.

**XGB is XGBoost.** The per-district files abbreviate; every table and figure label spells it out.
"""
from __future__ import annotations
import pandas as pd

CANONICAL = {"H3": "H3g", "XGB": "XGBoost"}


def canonical_models(df: pd.DataFrame, col: str, dedupe_on: list[str] | None = None
                     ) -> pd.DataFrame:
    """Rename model labels in `col` to their canonical form and drop rows the rename duplicates.

    `dedupe_on` names the columns that identify a row apart from the model label, e.g.
    ["Horizon"] or ["district", "horizon"]. Duplicates are only ever created by H3 -> H3g, and
    those two carry identical values, so keeping the first is not a choice between numbers.
    Passing None skips the dedupe, which is correct where the source has no duplicate to begin
    with, such as the pairwise Diebold-Mariano files.
    """
    out = df.copy()
    out[col] = out[col].replace(CANONICAL)
    if dedupe_on is not None:
        before = len(out)
        out = out.drop_duplicates(subset=[col] + list(dedupe_on), keep="first")
        if before != len(out):
            print(f"[labels] collapsed {before - len(out)} duplicate H3/H3g row(s) in '{col}'")
    return out.reset_index(drop=True)
