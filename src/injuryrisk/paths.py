"""Every filesystem location the project reads or writes.

The repo root is found by walking up from this file to the first directory
holding ``pyproject.toml``, so moving a module never breaks a path.
"""
from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def _discover_repo_root() -> Path:
    for candidate in _PACKAGE_DIR.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return _PACKAGE_DIR.parents[1]


REPO_ROOT = _discover_repo_root()

CONFIGS = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_EXTERNAL = DATA_DIR / "external"
REPORTS = REPO_ROOT / "reports"
EXPERIMENTS = REPO_ROOT / "experiments"

_WRITABLE = (DATA_RAW, DATA_PROCESSED, REPORTS, EXPERIMENTS)


def ensure_dirs() -> None:
    """Create output directories. Called from the CLI, never at import."""
    for d in _WRITABLE:
        d.mkdir(parents=True, exist_ok=True)
