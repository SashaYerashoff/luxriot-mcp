# luxriot-mcp

Docs-grounded Luxriot EVO assistant (Evo 1.32).

## Quickstart

1) Ingest docs (writes `datastore/evo_1_32/`):

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt
python3 backend/cli/ingest_evo_1_32.py --docs-dir docs --out-dir datastore/evo_1_32
```

2) Run backend (FastAPI):

```bash
uvicorn backend.app.main:app --reload --port 8000
```

3) Open `http://localhost:8000/` in your browser.

To serve on your LAN, run:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --reload --port 8000
```

## Re-index from UI

Open `Administrator tools → Docs` and click `RE-INDEX` to rebuild `datastore/evo_1_32/` from the docs folder.

Notes:
- If embeddings ingestion is unstable (LM Studio returns `400 {"error":"Model has unloaded or crashed.."}`), lower `Emb max chars` (try `448` or `384`) and/or lower `Emb batch` (try `4`).

## Streaming chat (SSE)

- `POST /chat/stream` streams server status + model output deltas (SSE over `fetch`).
- The UI uses streaming by default and falls back to `POST /chat` if streaming is unavailable.

## Web scraping (optional)

In `Administrator tools → Web`, enable web tools, then:

- Paste a URL in chat (e.g. `fetch https://www.luxriot.com/`) to fetch+summarize it.
- Use `search: <query>` / `web: <query>` to include DuckDuckGo HTML search results in context.
- The UI shows a “Web sources” section under answers when web context is used.

## Environment variables

- `LMSTUDIO_BASE_URL` (default `http://localhost:1234`)
- `LMSTUDIO_MODEL` (optional; auto-detected if unset)
- `LUXRIOT_DOCS_VERSION` (default `evo_1_32`)
- `LUXRIOT_APP_DB_PATH` (default `backend/data/app.sqlite`)

## MCP server

See `mcp-server/README.md` and `mcp-server/mcp.sample.json`.
