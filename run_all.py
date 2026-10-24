#!/usr/bin/env python3
"""Run exhibit scripts after supplying the inputs listed in data/README.md.

python3 run_all.py --list       list exhibit steps without executing them
python3 run_all.py              tables and figures from prepared intermediate inputs
python3 run_all.py --with-build rebuild Table 1, SHAP inputs and Appendix C first

Main rolling-window estimation has a separate entrypoint: code/estimation/main.py.
This code-only repository does not include any input data or reference outputs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
CODE = PKG / "code"

BUILD_STEPS = [
    ("Table 1 panels (needs data/raw)", CODE / "build" / "01_table1_panels.py"),
    ("SHAP bridges for Figures 5-8 (needs data/raw + frozen model)",
     CODE / "build" / "02_shap_bridges.py"),
    ("Appendix B: SHAP draw sensitivity (needs data/raw + frozen models; ~8 min)",
     CODE / "build" / "03_draw_sensitivity.py"),
    ("Appendix C: estimate H6, the maximal linear specification (needs data/raw; ~20 min)",
     CODE / "build" / "04_appendix_c.py"),
]

ANALYSIS_STEPS = [
    ("Table 2 (Full + Amenity) + self-check", CODE / "analysis" / "01_table2.py"),
    ("Table 3 (Full + Subsample DM, HAC) + self-check", CODE / "analysis" / "02_table3.py"),
    ("Per-district OOS R2 (Figure 4 Panel A input)", CODE / "analysis" / "03_district_r2.py"),
    ("Figures 1-8", CODE / "analysis" / "04_figures.py"),
    ("Manuscript-numbered figure set + FIGURE_MAP", CODE / "analysis" / "05_package_figures.py"),
    ("Appendix: SHAP attribution by refit + self-check",
     CODE / "analysis" / "06_window_sensitivity.py"),
    ("Appendix B: SHAP draw sensitivity + self-check",
     CODE / "analysis" / "07_draw_sensitivity.py"),
    ("Appendix C: predictor map + H6 out-of-sample R2 + self-check",
     CODE / "analysis" / "08_appendix_c.py"),
]



# matplotlib stamps a creation timestamp into every PDF it writes, which used to make the PDFs
# non-reproducible and was the stated reason for excluding them from the integrity ledger. It is
# avoidable: matplotlib honours SOURCE_DATE_EPOCH, the reproducible-builds convention, and pins the
# stamp to it. Fixing the value here makes the PDFs byte-identical across runs, so they are hashed
# like everything else. The value is arbitrary but must not change; it is 2023-11-14T22:13:20Z.
SOURCE_DATE_EPOCH = "1700000000"


def run(steps: list[tuple[str, Path]]) -> list[str]:
    env = {**os.environ, "PYTHONPATH": str(CODE),
           "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH}
    failures = []
    for label, script in steps:
        print(f"\n>>> {label}\n    {script.relative_to(PKG)}")
        rc = subprocess.run([sys.executable, str(script)], cwd=str(PKG), env=env).returncode
        print(f"    -> {'OK' if rc == 0 else f'FAIL (exit {rc})'}")
        if rc != 0:
            failures.append(label)
            break          # stop at the first failure; later steps consume its output
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-build", action="store_true",
                    help="run the build stage first (requires data/raw/)")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    args = ap.parse_args()

    steps = (BUILD_STEPS if args.with_build else []) + ANALYSIS_STEPS

    if args.list:
        for i, (label, script) in enumerate(steps, 1):
            print(f"{i:2d}. {label}\n    {script.relative_to(PKG)}")
        return 0

    print("=" * 78)
    print("sea_2026 replication package")
    print("stage: " + ("build + analysis" if args.with_build else "analysis (user-supplied inputs)"))
    print("=" * 78)

    failures = run(steps)

    print("\n" + "=" * 78)
    if failures:
        print("STOPPED AT:", failures[0])
        print("Nothing after that step ran. Fix the cause and re-run.")
        return 1
    print("ALL STEPS OK.")
    print(f"  tables  -> {(PKG / 'output/tables').relative_to(PKG)}")
    print(f"  figures -> {(PKG / 'output/figures/manuscript').relative_to(PKG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
