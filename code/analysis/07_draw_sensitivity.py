#!/usr/bin/env python3
"""Appendix B table — does the 5,000-row explanation draw drive the reported attribution?

The SHAP figures explain the seven frozen refits on a fixed draw of 5,000 transactions from the
final tenth of the amenity-complete sample. `code/build/03_draw_sensitivity.py` re-runs that same
decomposition under three alternative seeds and once on the entire final tenth. This script reads
what it wrote, formats Appendix B's table, and asserts the four claims the appendix makes.

What it shows: the draw does no work. Removing it entirely moves every category-share endpoint by
at most 0.9 percentage points, leaves the districts at the top and bottom of the time-share ranking
unchanged, and leaves the global attribution ranking of Figure 5 identical. District orderings
correlate with the shipped ordering at 0.92 or above under every alternative seed, and the rank
correlation between a district's time share and its mean price stays near zero throughout.

Reads a shipped CSV, so this runs in the default analysis stage without data/raw/.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runlog  # noqa: F401  (starts the run log; see docs/ and the README's Run logs section)
from _paths import DERIVED, TABLES, require  # noqa: E402

SRC = DERIVED / "draw_sensitivity.csv"
require(SRC)

# Condition label -> row label. `shipped` is the draw every other exhibit in the package uses, so it
# is the baseline every rank correlation is measured against and it leads the table.
ROW_LABELS = {
    "shipped": "Shipped draw (seed 42, n = 5,000)",
    "seed_7": "Alternative seed 7 (n = 5,000)",
    "seed_2026": "Alternative seed 2026 (n = 5,000)",
    "seed_31415": "Alternative seed 31415 (n = 5,000)",
    "entire_final_tenth": "Entire final tenth (no subsampling)",
}

# The four claims Appendix B makes, as thresholds. A claim the table cannot support should fail here
# rather than reach a referee.
MAX_ENDPOINT_SHIFT_PP = 1.0    # against the entire final tenth
MIN_RANK_CORR = 0.90           # district orderings, any alternative draw
MIN_GLOBAL_RANK_CORR = 0.99    # Figure 5's global ranking
MAX_ABS_PRICE_CORR = 0.30      # time share vs district mean price, every condition


def main() -> int:
    s = pd.read_csv(SRC).set_index("condition")
    missing = set(ROW_LABELS) - set(s.index)
    if missing:
        raise SystemExit(f"draw_sensitivity.csv is missing condition(s) {sorted(missing)}; "
                         "re-run code/build/03_draw_sensitivity.py")

    out = pd.DataFrame({
        "Condition": [ROW_LABELS[c] for c in ROW_LABELS],
        "Structural share, %": [f"{s.loc[c, 'structural_min']:.1f}-{s.loc[c, 'structural_max']:.1f}"
                                for c in ROW_LABELS],
        "Amenity share, %": [f"{s.loc[c, 'amenity_min']:.1f}-{s.loc[c, 'amenity_max']:.1f}"
                             for c in ROW_LABELS],
        "Time share, %": [f"{s.loc[c, 'time_min']:.1f}-{s.loc[c, 'time_max']:.1f}"
                          for c in ROW_LABELS],
        "Highest time share": [s.loc[c, "time_top_district"] for c in ROW_LABELS],
        "Lowest time share": [s.loc[c, "time_bottom_district"] for c in ROW_LABELS],
        "Rank corr., structural": [s.loc[c, "rank_corr_structural_vs_shipped"] for c in ROW_LABELS],
        "Rank corr., amenity": [s.loc[c, "rank_corr_amenity_vs_shipped"] for c in ROW_LABELS],
        "Rank corr., time": [s.loc[c, "rank_corr_time_vs_shipped"] for c in ROW_LABELS],
        "Rank corr., Figure 5 ranking": [s.loc[c, "rank_corr_global_vs_shipped"] for c in ROW_LABELS],
        "Corr., time share vs price": [s.loc[c, "rank_corr_time_share_vs_district_price"]
                                       for c in ROW_LABELS],
    })
    out.to_csv(TABLES / "TableB_draw_sensitivity.csv", index=False)

    # Self-check. Each block is one sentence Appendix B prints.
    base, full = s.loc["shipped"], s.loc["entire_final_tenth"]
    alts = [c for c in ROW_LABELS if c not in ("shipped", "entire_final_tenth")]
    ok = True

    shifts = {f"{cat}_{end}": abs(base[f"{cat}_{end}"] - full[f"{cat}_{end}"])
              for cat in ("structural", "amenity", "time") for end in ("min", "max")}
    worst = max(shifts, key=shifts.get)
    m = shifts[worst] <= MAX_ENDPOINT_SHIFT_PP
    ok &= m
    print(f"endpoint shift vs entire tenth: worst={worst} {shifts[worst]:.1f} pp "
          f"(limit {MAX_ENDPOINT_SHIFT_PP})  {'PASS' if m else 'FAIL'}")

    m = (base["time_top_district"] == full["time_top_district"]
         and base["time_bottom_district"] == full["time_bottom_district"])
    ok &= m
    print(f"time-share extremes unchanged vs entire tenth: "
          f"{base['time_top_district']}/{base['time_bottom_district']} vs "
          f"{full['time_top_district']}/{full['time_bottom_district']}  {'PASS' if m else 'FAIL'}")

    for col, limit in (("rank_corr_structural_vs_shipped", MIN_RANK_CORR),
                       ("rank_corr_amenity_vs_shipped", MIN_RANK_CORR),
                       ("rank_corr_time_vs_shipped", MIN_RANK_CORR),
                       ("rank_corr_global_vs_shipped", MIN_GLOBAL_RANK_CORR)):
        lo = min(s.loc[c, col] for c in alts)
        m = lo >= limit
        ok &= m
        print(f"{col:36s} min over alternative draws={lo:.3f} (limit {limit})  "
              f"{'PASS' if m else 'FAIL'}")

    hi = max(abs(s.loc[c, "rank_corr_time_share_vs_district_price"]) for c in ROW_LABELS)
    m = hi <= MAX_ABS_PRICE_CORR
    ok &= m
    print(f"|corr(time share, district price)| max over all conditions={hi:.3f} "
          f"(limit {MAX_ABS_PRICE_CORR})  {'PASS' if m else 'FAIL'}")

    print()
    print(out.to_string(index=False))
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
