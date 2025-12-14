# Luxriot MCP – Development Notes (Evo 1.32 docs foundation)

Date: 2025-12-12  
Goal of this sprint: **working end-to-end grounding on Luxriot EVO 1.32 documentation** (6 guides), with **citations + relevant screenshots** surfaced to the user and available to the model.

## Current status (repo reality)
- **Backend (FastAPI)**: implemented and working (`/health`, `/docs/search`, `/chat`, sessions/messages, assets).
- **Ingestion CLI**: implemented (`backend/cli/ingest_evo_1_32.py`) → builds `datastore/evo_1_32/` incl. **embeddings**.
- **Admin re-indexing**: UI triggers ingestion in the background and shows status/logs (`/admin/reindex`).
- **Frontend**: wired (`index.html`) to backend APIs; renders answers, citations, screenshots; shows raw errors.
- **Streaming**: `POST /chat/stream` (SSE) streams phases + model deltas; UI renders incrementally with activity status.
- **Cancel generation**: UI `STOP` button aborts the SSE request and keeps partial output visible.
- **MCP server**: implemented (`mcp-server/`) exposing `luxriot_docs_query`.
- **Web scraping**: optional URL fetch + DuckDuckGo HTML search (settings-driven; transparent in Admin Tools).
- **Prompt transparency**: **no hardcoded prompts in service code**. System prompt is stored in **settings** and editable via Admin Tools panel.
- **Retrieval**: BM25 + embeddings + **hybrid (RRF)**, plus deterministic consistency helpers:
  - **MMR diversification** (reduces redundancy; settings-driven)
  - **Heading match boost** (helps “Add/Configure/…” pages surface)
  - **Neighbor expansion** (stitches adjacent chunks to keep procedures intact)

## Prompt transparency contract
- Prompts are **data**, not code:
  - Defaults live in `backend/prompts/system_prompt.md` (and are seeded into SQLite settings).
  - Current values are stored in SQLite `settings` table and editable via `GET/POST /admin/settings`.
- Backend must not ship “hidden” system prompts or hidden guard-strings in code. If something is missing/misconfigured, return an explicit error.

---

## Definition of Done (end-of-day working)
1. **Frontend**: `index.html` is wired to backend APIs (no mocked successes; show honest errors).
2. **Docs ingestion**: a CLI ingests **Help+Manual HTML export + relative images** for **6 guides**, converts to **Markdown**, builds an index.
3. **Retrieval**: query returns a **context pack**: top chunks + citations + a capped set of image refs (e.g., 3–6).
4. **Model call**: LM Studio at `http://localhost:1234` is used as the LLM endpoint; answers are produced using retrieved context and include citations.
5. **Screenshots**: UI renders thumbnails/links for screenshots referenced by retrieved chunks.
6. **MCP**: Luxriot MCP server can be added to LM Studio via `mcp.json` and exposes at least one tool: `luxriot_docs_query` (query → context pack).

---

## Why this plan works (foundation-first)
- The product is a **documentation engine with a chat surface**, not a “chat app with docs bolted on.”
- We index **HTML + relative images** directly (best structural fidelity).
- We treat screenshots as **first-class assets**: preserved, referenced, served—vision models can use them when needed.

---

## High-level architecture
### A) Backend (Python 3, FastAPI recommended)
**Responsibilities**
- Ingest docs (HTML → MD), extract metadata (titles, headings, images), chunk sections.
- Build/search index (BM25; hybrid with embeddings).
- Serve assets (images) via HTTP.
- Orchestrate chat:
  - retrieve context pack
  - call LM Studio model
  - return answer + citations + image URLs
- Store sessions/messages (SQLite for sprint).

### B) MCP server (Node.js)
**Responsibilities**
- Expose MCP tools to LM Studio (stdio or http as supported by LM Studio).
- Primary tool:
  - `luxriot_docs_query({ query, k })` → calls backend `/docs/search` → returns chunks + citations + image URLs.
- Web tools: `duckduckgo_search({ query, k })`, `fetch_url({ url, ... })` (public web scraping + fetch).

### C) Frontend (HTML now, can migrate later)
- Keep your current mock.
- Replace mock data with API calls:
  - send message → `/chat`
  - load history → `/sessions`, `/sessions/:id/messages`
  - render citations + screenshots

---

## Folder layout proposal (inside `luxriot-mcp/`)
```
luxriot-mcp/
  index.html
  docs/                         # input: Help+Manual export(s) for Evo 1.32 (6 guide folders)
  datastore/
    evo_1_32/
      pages.jsonl               # per-page metadata (title, headings, images, source path)
      pages/                    # normalized markdown pages
      assets/                   # copied/normalized images
      index.sqlite              # BM25 (and later embeddings)
  backend/
    requirements.txt
    default_settings.json
    prompts/
      system_prompt.md
    app/
      main.py
      app_db.py
      settings.py
    cli/
      ingest_evo_1_32.py
  mcp-server/
    package.json
    src/
      index.ts
  README.md
  dev-notes.md                  # (this file, or similar)
```

---

## Data model (minimal, sprint-safe)
SQLite tables (or SQLModel):
- `sessions(id, title, created_at, updated_at)`
- `messages(id, session_id, role, content, created_at, meta_json)`
- `settings(scope, key, value_json)` *(scope: global / user / role)*

Docs store:
- `pages.jsonl` contains:
  - `doc_id`, `page_id`, `title`, `heading_path`, `source_path`, `md_path`
  - `images[]`: `{ asset_path, url_path, near_heading, alt }`

---

## APIs (minimal set)
### Health / config
- `GET /health`
- `GET /admin/settings` *(includes defaults + current + effective)*
- `POST /admin/settings` *(updates settings; applies immediately)*

### Docs
- `POST /docs/ingest` *(optional; CLI is preferred)*
- `POST /docs/search` → `{ query, k }` → context pack:
  - `chunks[]`: `{ doc_id, page_id, heading_path, text, score }`
  - `citations[]`: `{ title, doc_id, page_id, anchor, source_path }`
  - `images[]`: `{ url, doc_id, page_id, near_heading }`

### Chat
- `POST /chat` → `{ session_id?, message }` → `{ session_id, answer, citations, images }`
- `GET /sessions`
- `POST /sessions`
- `GET /sessions/{id}/messages`

### Assets
- `GET /assets/{doc_version}/{...path}`
- `GET /rawdocs/{doc_version}/{...path}` *(serves original HTML for clickable citations; next: markdown viewer)*

---

## Ingestion specifics (Help+Manual HTML export)
We convert HTML → Markdown with:
- heading extraction (H1/H2/H3)
- ordered list preservation (step-by-step procedures stay intact)
- note/warning blocks preserved
- images preserved and remapped to backend-served URLs

Image preservation rule:
- copy images into `datastore/evo_1_32/assets/`
- keep stable URL mapping under `/assets/evo_1_32/...`
- store “near heading” context for better selection at answer time

---

## Retrieval strategy (Day-1)
- BM25 keyword index is enough to ship and is robust for product terms.
- Embeddings + hybrid scoring (RRF) are implemented for better recall.

Notes:
- LM Studio `/v1/embeddings` can return intermittent `400 {"error":"Model has unloaded or crashed.."}` for some doc strings.
- The ingester uses deterministic text normalization (H3/table markdown cleanup) plus bounded truncation / bag-of-words fallback to ensure ingestion completes; it logs when fallbacks are used.
- Default reindex knobs are conservative for stability: `embedding_max_chars=448`, `embedding_batch_size=8` (adjustable in Admin Tools → Docs).

Image selection policy:
- cap per answer (3–6)
- prioritize images that are:
  - referenced inside top chunks
  - closest to step lists
  - nearest heading match

---

## MCP in LM Studio (how we’ll integrate)
LM Studio supports MCP servers and can be configured via `mcp.json` (Cursor-style notation).

Plan:
- MCP server implemented in Node.js.
- Add it to LM Studio via `mcp.json` so LM Studio can call `luxriot_docs_query` and inject the returned context into the model run.

---

## Next steps checklist (practical sequence)
1. **Backend skeleton**: FastAPI app + `/health` + static assets serving.
2. **Ingest CLI**: HTML → MD + `pages.jsonl` + BM25 index build.
3. **Search endpoint**: return context packs with citations + image URLs.
4. **Chat endpoint**: retrieval → LM Studio call → answer + citations + images.
5. **UI wiring**: connect mock to `/chat`, show citations + thumbnails.
6. **MCP server**: implement `luxriot_docs_query` tool calling `/docs/search`.
7. **LM Studio config**: register server in `mcp.json`, test tool invocation.
8. **Done** Embeddings + hybrid retrieval (RRF) with transparent settings.
9. **Next** Consider MMR/diversification + snippet shaping for higher answer consistency.
10. **Next** Replace raw HTML citations with a mkdocs-like Markdown viewer route.

---

## Non-goals for this sprint (explicitly deferred)
- Full multi-version routing (1.31/1.30 etc). We only do **latest: 1.32** now.
- Full screenshot understanding (captioning/OCR). We only **serve + reference** them.
- Fine-tuning pipeline beyond basic export (nice-to-have later).
