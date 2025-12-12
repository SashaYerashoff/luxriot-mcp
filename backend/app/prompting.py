from __future__ import annotations

from typing import Any


class PromptTemplateError(RuntimeError):
    pass


def render_template(template: str, variables: dict[str, Any], required_placeholders: list[str] | None = None) -> str:
    required_placeholders = required_placeholders or []

    missing = [p for p in required_placeholders if f"{{{{{p}}}}}" not in template]
    if missing:
        raise PromptTemplateError(
            "System prompt template missing required placeholders: " + ", ".join(f"{{{{{m}}}}}" for m in missing)
        )

    out = template
    for key, value in variables.items():
        out = out.replace(f"{{{{{key}}}}}", str(value))
    return out

