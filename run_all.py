#!/usr/bin/env python3
"""Generate study exhibits using the inputs listed in data/README.md.

Use --list to display the steps or --with-build to rebuild intermediate inputs.
Main model estimation is provided by code/estimation/main.py.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
SOURCE_DATE_EPOCH = "1700000000"


class Step(NamedTuple):
    """An exhibit script and its command-line arguments."""

    label: str
    script: Path
    arguments: tuple[str, ...] = ()


BUILD_STEPS = [
    Step("Table 1", CODE_DIR / "build" / "01_table1_panels.py"),
    Step("SHAP inputs for Figures 5-8 and Appendix A", CODE_DIR / "build" / "02_shap_bridges.py"),
    Step(
        "SHAP inputs with district effects separated from location (Figures 6B and 7)",
        CODE_DIR / "build" / "02_shap_bridges.py",
        ("--district-out-of-location",),
    ),
    Step("Appendix B draw sensitivity", CODE_DIR / "build" / "03_draw_sensitivity.py"),
    Step("Appendix C estimation", CODE_DIR / "build" / "04_appendix_c.py"),
]

ANALYSIS_STEPS = [
    Step("Table 2", CODE_DIR / "analysis" / "01_table2.py"),
    Step("Table 3", CODE_DIR / "analysis" / "02_table3.py"),
    Step("District performance for Figure 4", CODE_DIR / "analysis" / "03_district_r2.py"),
    Step("Figures 1-8", CODE_DIR / "analysis" / "04_figures.py"),
    Step("Manuscript figure files", CODE_DIR / "analysis" / "05_package_figures.py"),
    Step("Appendix A refit sensitivity", CODE_DIR / "analysis" / "06_window_sensitivity.py"),
    Step("Appendix B draw sensitivity", CODE_DIR / "analysis" / "07_draw_sensitivity.py"),
    Step("Appendix C tables", CODE_DIR / "analysis" / "08_appendix_c.py"),
]


def run(steps: list[Step]) -> list[str]:
    """Execute steps in order, stopping at the first failure."""
    env = {**os.environ, "PYTHONPATH": str(CODE_DIR), "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH}
    for step in steps:
        command = [sys.executable, str(step.script), *step.arguments]
        print(f"\n>>> {step.label}", flush=True)
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
        if result.returncode != 0:
            print(f"    Failed with exit code {result.returncode}", flush=True)
            return [step.label]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-build",
        action="store_true",
        help="Rebuild intermediate inputs first; requires transactions and fitted models",
    )
    parser.add_argument(
        "--list", action="store_true", help="Display the selected steps without running them"
    )
    args = parser.parse_args()
    steps = (BUILD_STEPS if args.with_build else []) + ANALYSIS_STEPS

    if args.list:
        for index, step in enumerate(steps, 1):
            command = ["python", str(step.script.relative_to(PROJECT_ROOT)), *step.arguments]
            print(f"{index:2d}. {step.label}\n    {shlex.join(command)}")
        return 0

    print("Seoul housing valuation analysis code", flush=True)
    failures = run(steps)
    if failures:
        print(f"Stopped at: {failures[0]}. Resolve the error before rerunning.")
        return 1
    print("All selected steps completed.")
    print("  Tables: output/tables/")
    print("  Figures: output/figures/manuscript/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
