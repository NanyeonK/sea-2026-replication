#!/usr/bin/env python3
"""Run the exhibit scripts using the inputs listed in data/README.md.

Use --list to display the steps or --with-build to rebuild intermediate inputs.
Main model estimation is provided by code/estimation/main.py.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"

BUILD_STEPS = [
    ("Table 1 panels (needs data/raw)", CODE_DIR / "build" / "01_table1_panels.py"),
    ("SHAP inputs for Figures 5-8 (needs data/raw + frozen model)",
     CODE_DIR / "build" / "02_shap_bridges.py"),
    ("Appendix B: SHAP draw sensitivity (needs data/raw + frozen models; ~8 min)",
     CODE_DIR / "build" / "03_draw_sensitivity.py"),
    ("Appendix C: estimate H6, the maximal linear specification (needs data/raw; ~20 min)",
     CODE_DIR / "build" / "04_appendix_c.py"),
]

ANALYSIS_STEPS = [
    ("Table 2 (Full + Amenity) and reference checks", CODE_DIR / "analysis" / "01_table2.py"),
    ("Table 3 (Full + Subsample DM, HAC) and reference checks", CODE_DIR / "analysis" / "02_table3.py"),
    ("Per-district OOS R2 (Figure 4 Panel A input)", CODE_DIR / "analysis" / "03_district_r2.py"),
    ("Figures 1-8", CODE_DIR / "analysis" / "04_figures.py"),
    ("Manuscript-numbered figure set + FIGURE_MAP", CODE_DIR / "analysis" / "05_package_figures.py"),
    ("Appendix: SHAP attribution by refit and reference checks",
     CODE_DIR / "analysis" / "06_window_sensitivity.py"),
    ("Appendix B: SHAP draw sensitivity and reference checks",
     CODE_DIR / "analysis" / "07_draw_sensitivity.py"),
    ("Appendix C: predictor map + H6 out-of-sample R2 and reference checks",
     CODE_DIR / "analysis" / "08_appendix_c.py"),
]

SOURCE_DATE_EPOCH = "1700000000"


def run(steps: list[tuple[str, Path]]) -> list[str]:
    env = {**os.environ, "PYTHONPATH": str(CODE_DIR),
           "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH}
    failures = []
    for label, script in steps:
        print(f"\n>>> {label}\n    {script.relative_to(PROJECT_ROOT)}")
        rc = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT), env=env).returncode
        print(f"    -> {'OK' if rc == 0 else f'FAIL (exit {rc})'}")
        if rc != 0:
            failures.append(label)
            break
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
            print(f"{i:2d}. {label}\n    {script.relative_to(PROJECT_ROOT)}")
        return 0

    print("=" * 78)
    print("Seoul housing valuation analysis code")
    print("stage: " + ("build + analysis" if args.with_build else "analysis (user-supplied inputs)"))
    print("=" * 78)

    failures = run(steps)

    print("\n" + "=" * 78)
    if failures:
        print("STOPPED AT:", failures[0])
        print("Nothing after that step ran. Fix the cause and re-run.")
        return 1
    print("ALL STEPS OK.")
    print(f"  tables  -> {(PROJECT_ROOT / 'output/tables').relative_to(PROJECT_ROOT)}")
    print(f"  figures -> {(PROJECT_ROOT / 'output/figures/manuscript').relative_to(PROJECT_ROOT)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
