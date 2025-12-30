from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import app_db
from .config import REPO_ROOT
from .logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_SETTINGS_PATH = REPO_ROOT / "backend" / "default_settings.json"


class SettingsError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SettingsError(f"Failed to read settings file {path}: {e}") from e


def load_defaults() -> dict[str, Any]:
    if not DEFAULT_SETTINGS_PATH.exists():
        raise SettingsError(f"Default settings file not found: {DEFAULT_SETTINGS_PATH}")

    raw = _read_json(DEFAULT_SETTINGS_PATH)

    prompt_file = raw.get("system_prompt_template_file")
    if not isinstance(prompt_file, str) or not prompt_file:
        raise SettingsError("default_settings.json missing 'system_prompt_template_file'")

    prompt_path = (REPO_ROOT / prompt_file).resolve()
    if not prompt_path.exists():
        raise SettingsError(f"System prompt template file not found: {prompt_path}")

    system_prompt_template = prompt_path.read_text(encoding="utf-8")
    defaults = dict(raw)
    defaults["system_prompt_template"] = system_prompt_template

    prompt_files = raw.get("system_prompt_templates_files")
    if isinstance(prompt_files, dict):
        templates: dict[str, str] = {}
        for role, rel_path in prompt_files.items():
            if not isinstance(role, str) or not role.strip():
                continue
            if not isinstance(rel_path, str) or not rel_path.strip():
                continue
            p = (REPO_ROOT / rel_path).resolve()
            if not p.exists():
                raise SettingsError(f"System prompt template file not found: {p}")
            templates[str(role).strip()] = p.read_text(encoding="utf-8")
        if templates:
            defaults["system_prompt_templates"] = templates
    return defaults


def get_settings_bundle() -> dict[str, Any]:
    defaults = load_defaults()
    db_settings = app_db.list_settings()

    effective: dict[str, Any] = {}
    for k, v in defaults.items():
        effective[k] = v
    for k, v in db_settings.items():
        effective[k] = v

    return {"defaults": defaults, "settings": db_settings, "effective": effective}


def ensure_defaults() -> None:
    bundle = get_settings_bundle()
    defaults: dict[str, Any] = bundle["defaults"]
    settings: dict[str, Any] = bundle["settings"]

    to_set: dict[str, Any] = {}

    # Only seed keys that are missing.
    for key in ("required_placeholders", "system_prompt_template", "system_prompt_templates", "llm", "retrieval"):
        if key not in settings and key in defaults:
            to_set[key] = defaults[key]

    # If we just introduced role-based prompts, preserve any existing single prompt template.
    if "system_prompt_templates" not in settings and "system_prompt_template" in settings:
        base = settings.get("system_prompt_template")
        if isinstance(base, str) and base.strip():
            to_set["system_prompt_templates"] = {
                "admin": base,
                "redactor": base,
                "support": base,
                "client": base,
                "anonymous": base,
            }

    def merge_missing(dst: Any, src: Any) -> tuple[Any, bool]:
        if not isinstance(dst, dict) or not isinstance(src, dict):
            return dst, False
        changed = False
        out = dict(dst)
        for k, v in src.items():
            if k not in out:
                out[k] = v
                changed = True
            else:
                merged, did = merge_missing(out[k], v)
                if did:
                    out[k] = merged
                    changed = True
        return out, changed

    # Backfill newly added nested defaults without overwriting user values.
    for key in ("llm", "retrieval"):
        if key in settings and key in defaults:
            merged, changed = merge_missing(settings.get(key), defaults.get(key))
            if changed:
                to_set[key] = merged

    if to_set:
        log.info("Seeding default settings keys: %s", ", ".join(sorted(to_set.keys())))
        app_db.set_settings(to_set)


def update_settings(new_values: dict[str, Any]) -> dict[str, Any]:
    app_db.set_settings(new_values)
    return get_settings_bundle()
