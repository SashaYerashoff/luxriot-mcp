from __future__ import annotations

import os
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return list(default)
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items or list(default)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _repo_root()

DOCS_DIR = Path(os.getenv("LUXRIOT_DOCS_DIR", str(REPO_ROOT / "docs"))).expanduser()
DATASTORE_DIR = Path(os.getenv("LUXRIOT_DATASTORE_DIR", str(REPO_ROOT / "datastore"))).expanduser()

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

CORS_ORIGINS = _env_list("LUXRIOT_CORS_ORIGINS", ["*"])
CORS_ALLOW_CREDENTIALS = _env_bool("LUXRIOT_CORS_ALLOW_CREDENTIALS", False)
TRUSTED_HOSTS = _env_list("LUXRIOT_TRUSTED_HOSTS", ["*"])

COOKIE_SECURE = _env_bool("LUXRIOT_COOKIE_SECURE", False)
COOKIE_SAMESITE = (os.getenv("LUXRIOT_COOKIE_SAMESITE", "lax") or "lax").strip().lower()
if COOKIE_SAMESITE not in {"lax", "strict", "none"}:
    COOKIE_SAMESITE = "lax"
COOKIE_DOMAIN = (os.getenv("LUXRIOT_COOKIE_DOMAIN") or "").strip() or None

RAWDOCS_REQUIRE_AUTH = _env_bool("LUXRIOT_RAWDOCS_REQUIRE_AUTH", True)
ASSETS_REQUIRE_AUTH = _env_bool("LUXRIOT_ASSETS_REQUIRE_AUTH", True)
