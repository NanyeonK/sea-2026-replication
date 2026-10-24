"""Record software versions, input hashes, output files and elapsed runtime.

Importing this module starts a log in output/logs/. Use seed(value) to set
and record the NumPy and Python random seeds.
"""
from __future__ import annotations

import atexit
import hashlib
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "output" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_START_WALL = time.time()
_START_STAMP = datetime.now(timezone.utc)
_SCRIPT = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path("interactive")
_STEM = _SCRIPT.stem or "interactive"
_LOG = LOG_DIR / f"{_STEM}_{_START_STAMP.strftime('%Y%m%dT%H%M%SZ')}.log"

_inputs: list[Path] = []
_seeds: list[tuple[str, int]] = []

_OUTPUT_DIRECTORIES = [PROJECT_ROOT / "output", PROJECT_ROOT / "data" / "derived", PROJECT_ROOT / "docs"]

_DEPENDENCIES = ["pandas", "numpy", "scipy", "sklearn", "statsmodels", "xgboost", "shap",
             "matplotlib", "geopandas", "joblib", "yaml", "pyarrow"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _git() -> str:
    """Return the repository commit and working-tree status, when available."""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=10)
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != PROJECT_ROOT:
            return "not a standalone git repository"
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, timeout=20)
        state = "dirty" if dirty.stdout.strip() else "clean"
        return f"{rev.stdout.strip()} ({state})"
    except Exception as exc:                                    # noqa: BLE001
        return f"unavailable: {exc}"


def _packages() -> list[str]:
    import importlib
    rows = []
    for name in _DEPENDENCIES:
        try:
            mod = importlib.import_module(name)
            rows.append(f"    {name:<14} {getattr(mod, '__version__', 'version unknown')}")
        except ImportError:
            rows.append(f"    {name:<14} not installed")
    return rows


def _memory() -> str:
    try:
        pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        return f"{pages / 1e9:.1f} GB"
    except (ValueError, OSError, AttributeError):
        return "unknown"


def register_input(path: Path) -> None:
    """Record an input path and its SHA-256 hash."""
    p = Path(path)
    if p.exists() and p not in _inputs:
        _inputs.append(p)


def seed(value: int, *, label: str = "global") -> int:
    """Set and record the NumPy and Python random seeds."""
    import random
    random.seed(value)
    try:
        import numpy as np
        np.random.seed(value)
    except ImportError:
        pass
    _seeds.append((label, int(value)))
    return value


def _write_header() -> None:
    lock = PROJECT_ROOT / "requirements.txt"
    lines = [
        f"# run log — {_STEM}",
        f"started_utc      : {_START_STAMP.isoformat()}",
        f"script           : {_rel(_SCRIPT)}",
        f"argv             : {' '.join(sys.argv[1:]) or '(none)'}",

        f"cwd              : {_rel(Path.cwd())}",
        "",
        "## environment",

        f"    python       {sys.version.split()[0]}",
        f"    os           {platform.system()} {platform.release()}",
        f"    machine      {platform.machine()}, {os.cpu_count()} logical CPUs, "
        f"{_memory()} RAM",
        f"    git          {_git()}",
        f"    lockfile     requirements.txt sha256 "
        f"{sha256(lock) if lock.exists() else 'MISSING'}",
        "",
        "## packages",
        *_packages(),
        "",
    ]
    _LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_footer() -> None:
    elapsed = time.time() - _START_WALL
    lines = ["## seeds"]
    lines += ([f"    {lab:<14} {val}" for lab, val in _seeds]
              or ["    (this script draws no randomness)"])

    lines += ["", "## inputs (sha256)"]
    lines += ([f"    {sha256(p)}  {_rel(p)}" for p in _inputs if p.exists()]
              or ["    (none declared)"])

    written: list[Path] = []
    for root in _OUTPUT_DIRECTORIES:
        if not root.exists():
            continue
        for p in root.rglob("*"):

            if p.is_file() and p != _LOG and p.suffix != ".pyc" \
                    and p.parent != LOG_DIR \
                    and p.stat().st_mtime > _START_WALL:
                written.append(p)
    lines += ["", "## outputs written by this run (sha256)"]
    lines += ([f"    {sha256(p)}  {_rel(p)}" for p in sorted(written)]
              or ["    (none)"])

    lines += ["", f"elapsed_seconds  : {elapsed:.2f}",
              f"finished_utc     : {datetime.now(timezone.utc).isoformat()}", ""]
    with _LOG.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

_write_header()
atexit.register(_write_footer)
