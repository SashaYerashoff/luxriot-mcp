from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from typing import Any, AsyncIterator

import httpx

from .config import LMSTUDIO_BASE_URL, LMSTUDIO_MODEL
from .logging_utils import get_logger

log = get_logger(__name__)


class LMStudioError(RuntimeError):
    pass


_model_cache_lock = asyncio.Lock()
_cached_model_id: str | None = None

_EMBED_H3_RE = re.compile(r"(?m)^([ \t]*)###(\s+)")
_CONTROL_RE = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _markdown_tables_to_text(text: str) -> str:
    lines: list[str] = []
    for line in (text or "").splitlines():
        if "|" not in line:
            lines.append(line)
            continue
        stripped = line.strip().strip("|").strip()
        if not stripped:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and all((not c) or (set(c) <= set("-:")) for c in cells):
            continue
        cleaned = []
        for c in cells:
            c = c.strip()
            if not c:
                continue
            c = c.replace("**", "")
            c = re.sub(r"\s+", " ", c).strip()
            if c:
                cleaned.append(c)
        if cleaned:
            lines.append(" - ".join(cleaned))
    return "\n".join(lines)


def _prepare_embedding_text(text: str, max_chars: int = 4000) -> str:
    t = (text or "").replace("\x00", " ")
    t = unicodedata.normalize("NFKC", t).replace("\u00a0", " ").replace("\\*", "*")
    t = (
        t.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2026", "...")
    )
    # Match ingester workaround: downgrade markdown H3 headings ("### ") to H2 to avoid LM Studio embedding errors.
    t = _EMBED_H3_RE.sub(r"\1##\2", t)
    t = _markdown_tables_to_text(t).replace("**", "").replace("*", "")
    t = _CONTROL_RE.sub(" ", t).strip()
    if max_chars > 0 and len(t) > max_chars:
        t = t[:max_chars]
    return t


async def _detect_model_id(client: httpx.AsyncClient, base_url: str) -> str:
    resp = await client.get(f"{base_url}/v1/models")
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data") or []
    if not models:
        raise LMStudioError("LM Studio returned no models; load a model in LM Studio.")
    for m in models:
        model_id = str(m.get("id") or "").strip()
        if not model_id:
            continue
        low = model_id.lower()
        if "embedding" in low or low.startswith("text-embedding"):
            continue
        return model_id
    raise LMStudioError("No chat/LLM model found in LM Studio /v1/models. Load a non-embedding model, or set LMSTUDIO_MODEL.")


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
    model: str | None = None,
    timeout_s: float = 60.0,
) -> str:
    model_id = str(model).strip() if isinstance(model, str) and model.strip() else None
    if not model_id:
        model_id = await get_model_id(base_url)
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
        try:
            resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise LMStudioError(f"LM Studio request timed out after {timeout_s:.1f}s ({type(e).__name__}).") from e
        except httpx.HTTPError as e:
            detail = None
            if hasattr(e, "response") and e.response is not None:
                try:
                    detail = e.response.text
                except Exception:
                    detail = None
            msg = str(e).strip() or repr(e)
            raise LMStudioError(
                f"LM Studio request failed ({type(e).__name__}): {msg} {detail or ''}".strip()
            ) from e

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise LMStudioError(f"Unexpected LM Studio response shape: {data}") from e


async def chat_completion_stream(
    messages: list[dict[str, Any]],
    base_url: str = LMSTUDIO_BASE_URL,
    temperature: float = 0.2,
    max_tokens: int = 800,
    model: str | None = None,
    timeout_s: float = 60.0,
) -> AsyncIterator[str]:
    model_id = str(model).strip() if isinstance(model, str) and model.strip() else None
    if not model_id:
        model_id = await get_model_id(base_url)

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    # For streaming, read timeout is per-chunk; keep it generous.
    timeout = httpx.Timeout(timeout_s, connect=10.0, read=timeout_s, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream("POST", f"{base_url}/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    s = line.strip()
                    if not s.startswith("data:"):
                        continue
                    data_str = s[len("data:") :].strip()
                    if not data_str:
                        continue
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except Exception:
                        continue
                    try:
                        choice0 = (data.get("choices") or [{}])[0] or {}
                        delta = choice0.get("delta") or {}
                        content = delta.get("content")
                        if not content:
                            continue
                        yield str(content)
                    except Exception:
                        continue
        except httpx.TimeoutException as e:
            raise LMStudioError(f"LM Studio request timed out after {timeout_s:.1f}s ({type(e).__name__}).") from e
        except httpx.HTTPError as e:
            detail = None
            if hasattr(e, "response") and e.response is not None:
                try:
                    detail = e.response.text
                except Exception:
                    detail = None
            msg = str(e).strip() or repr(e)
            raise LMStudioError(f"LM Studio request failed ({type(e).__name__}): {msg} {detail or ''}".strip()) from e


async def embeddings(
    texts: list[str],
    model: str,
    base_url: str = LMSTUDIO_BASE_URL,
    timeout_s: float = 60.0,
) -> list[list[float]]:
    if not texts:
        return []
    prepared = [_prepare_embedding_text(t) for t in texts]
    payload = {"model": model, "input": prepared}
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        try:
            resp = await client.post(f"{base_url}/v1/embeddings", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            detail = None
            if hasattr(e, "response") and e.response is not None:
                try:
                    detail = e.response.text
                except Exception:
                    detail = None
            raise LMStudioError(f"LM Studio embeddings request failed: {e} {detail or ''}".strip()) from e

        data = resp.json()
        items = data.get("data") or []
        if len(items) != len(texts):
            raise LMStudioError(f"Embedding response size mismatch: got {len(items)} embeddings for {len(texts)} inputs.")
        items = sorted(items, key=lambda x: int(x.get("index", 0)))
        out: list[list[float]] = []
        for it in items:
            emb = it.get("embedding")
            if not isinstance(emb, list) or not emb:
                raise LMStudioError("Embedding response missing 'embedding' list.")
            out.append([float(x) for x in emb])
        return out
