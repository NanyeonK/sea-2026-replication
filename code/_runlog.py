"""Run logging — every script records what the README has to state.

The Social Science Data Editors README template asks for software versions, package versions, hardware and
wall-clock runtime, and JF ships log files as Computational Materials. Those are measurements, so
they are collected at run time rather than recalled at submission time.

Importing this module starts a log. It writes a header immediately and, via `atexit`, a footer
with the outputs and the elapsed time — so a script that crashes still leaves a log saying how far
it got. Scripts do not have to call anything: `_paths.require()` registers inputs automatically,
and outputs are detected by modification time. The one thing a script must declare explicitly is a
random seed, because no automatic method can tell which integer was intended as one.

    from _runlog import seed
    seed(42)                     # sets numpy + stdlib RNGs and records the value

Log path: output/logs/<script>_<UTC timestamp>.log
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

PKG = Path(__file__).resolve().parents[1]
LOGDIR = PKG / "output" / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

_START_WALL = time.time()
_START_STAMP = datetime.now(timezone.utc)
_SCRIPT = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path("interactive")
_STEM = _SCRIPT.stem or "interactive"
_LOG = LOGDIR / f"{_STEM}_{_START_STAMP.strftime('%Y%m%dT%H%M%SZ')}.log"

_inputs: list[Path] = []
_seeds: list[tuple[str, int]] = []
# Directories whose newly written files count as this run's outputs.
_WATCH = [PKG / "output", PKG / "data" / "derived", PKG / "docs"]

# Packages worth recording. JFE asks for "all modules, libraries, toolboxes, packages"; these are
# the ones this package can import. Missing ones are reported as such rather than skipped.
_PACKAGES = ["pandas", "numpy", "scipy", "sklearn", "statsmodels", "xgboost", "shap",
             "matplotlib", "geopandas", "joblib", "yaml", "pyarrow"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(PKG))
    except ValueError:
        return str(p)


def _git() -> str:
    """Commit of the package itself, or a plain statement that there is none.

    This used to run `git rev-parse` with cwd=PKG, which walked up into the private parent
    repository and published ITS commit hash plus a "(dirty)" flag in every shipped log. A deposit
    should report its own provenance or nothing. If the deposit is not itself a repository, say so.
    """
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=PKG,
                             capture_output=True, text=True, timeout=10)
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != PKG:
            return "not a standalone git repository"
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PKG,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PKG,
                               capture_output=True, text=True, timeout=20)
        state = "dirty" if dirty.stdout.strip() else "clean"
        return f"{rev.stdout.strip()} ({state})"
    except Exception as exc:                                    # noqa: BLE001
        return f"unavailable: {exc}"


def _packages() -> list[str]:
    import importlib
    rows = []
    for name in _PACKAGES:
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
    """Called by _paths.require(); records an input and its hash."""
    p = Path(path)
    if p.exists() and p not in _inputs:
        _inputs.append(p)


def seed(value: int, *, label: str = "global") -> int:
    """Set and record an RNG seed. Seeds are literals in the code, and they are logged."""
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
    lock = PKG / "requirements.txt"
    lines = [
        f"# run log — {_STEM}",
        f"started_utc      : {_START_STAMP.isoformat()}",
        f"script           : {_rel(_SCRIPT)}",
        f"argv             : {' '.join(sys.argv[1:]) or '(none)'}",
        # Deliberately package-relative. Logging os.getcwd() published the author's username and
        # the private parent repository's directory name in every shipped log (found 2026-07-30).
        f"cwd              : {_rel(Path.cwd())}",
        "",
        "## environment",
        # Interpreter path omitted for the same reason as cwd: a venv under the author's home
        # would leak the username. The version is what the README has to state.
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
    for root in _WATCH:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            # Strictly after this run started. An earlier tolerance of one second pulled in the
            # previous step's outputs when steps ran back to back, which made the log claim
            # authorship of files it did not write.
            if p.is_file() and p != _LOG and p.suffix != ".pyc" \
                    and p.parent != LOGDIR \
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
