from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _repo_root()

DOCS_DIR = REPO_ROOT / "docs"
DATASTORE_DIR = REPO_ROOT / "datastore"

DEFAULT_VERSION = os.getenv("LUXRIOT_DOCS_VERSION", "evo_1_32")


def _read_app_version() -> str:
    env_version = os.getenv("LUXRIOT_APP_VERSION")
    if env_version:
        return env_version.strip()
    path = REPO_ROOT / "VERSION"
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return "Luxriot SA 0.0.0"


APP_VERSION = _read_app_version()

APP_DB_PATH = Path(os.getenv("LUXRIOT_APP_DB_PATH", str(REPO_ROOT / "backend" / "data" / "app.sqlite")))

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234").rstrip("/")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL") or None
