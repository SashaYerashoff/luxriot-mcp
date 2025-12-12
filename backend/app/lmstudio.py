from __future__ import annotations

import asyncio
from typing import Any

import httpx

from .config import LMSTUDIO_BASE_URL, LMSTUDIO_MODEL
from .logging_utils import get_logger

log = get_logger(__name__)


class LMStudioError(RuntimeError):
    pass


_model_cache_lock = asyncio.Lock()
_cached_model_id: str | None = None


async def _detect_model_id(client: httpx.AsyncClient, base_url: str) -> str:
    resp = await client.get(f"{base_url}/v1/models")
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data") or []
    if not models:
        raise LMStudioError("LM Studio returned no models; load a model in LM Studio.")
    model_id = models[0].get("id")
    if not model_id:
        raise LMStudioError("LM Studio models list missing 'id'.")
    return str(model_id)


async def get_model_id(base_url: str = LMSTUDIO_BASE_URL) -> str:
    global _cached_model_id

    if LMSTUDIO_MODEL:
        return LMSTUDIO_MODEL

    if _cached_model_id:
        return _cached_model_id

    async with _model_cache_lock:
        if _cached_model_id:
            return _cached_model_id
        async with httpx.AsyncClient(timeout=10.0) as client:
            _cached_model_id = await _detect_model_id(client, base_url)
            return _cached_model_id


async def chat_completion(
    messages: list[dict[str, Any]],
    base_url: str = LMSTUDIO_BASE_URL,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    model_id = await get_model_id(base_url)
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            detail = None
            if hasattr(e, "response") and e.response is not None:
                try:
                    detail = e.response.text
                except Exception:
                    detail = None
            raise LMStudioError(f"LM Studio request failed: {e} {detail or ''}".strip()) from e

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise LMStudioError(f"Unexpected LM Studio response shape: {data}") from e

