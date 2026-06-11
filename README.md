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

## Versioning

- App version is stored in `VERSION` (default format: `Luxriot SA 0.x.y`).
- Release notes live in `CHANGELOG.md`.
- `/health` now returns `app_version` so the UI can display it.
- You can override the version with `LUXRIOT_APP_VERSION`.

## Re-index from UI

Open `Administrator tools → Docs` and click `RE-INDEX` to rebuild `datastore/evo_1_32/` from the docs folder.
Use `REFRESH` to reindex from the current datastore (ingested docs + published edits only).

Notes:
- If embeddings ingestion is unstable (LM Studio returns `400 {"error":"Model has unloaded or crashed.."}`), lower `Emb max chars` (try `448` or `384`) and/or lower `Emb batch` (try `4`).

## Streaming chat (SSE)

- `POST /chat/stream` streams server status + model output deltas (SSE over `fetch`).
- The UI uses streaming by default and falls back to `POST /chat` if streaming is unavailable.
- While streaming, the UI shows a `STOP` button to cancel generation (keeps partial output in the chat; no assistant message is saved server-side).

## Web scraping (optional)

In `Administrator tools → Web`, enable web tools, then:

- Paste a URL in chat (e.g. `fetch https://www.luxriot.com/`) to fetch+summarize it.
- Use `search: <query>` / `web: <query>` to include DuckDuckGo HTML search results in context.
- The UI shows a “Web sources” section under answers when web context is used.

## Authentication & roles

- Default: `anonymous` (docs reader + chat; no chat history).
- Bootstrap admin: if the DB has zero users, an `admin` user is created on startup. Set `LUXRIOT_ADMIN_USERNAME` / `LUXRIOT_ADMIN_PASSWORD` to control credentials; otherwise the password is generated and printed to server logs.
- Reset/create admin without deleting the DB: set `LUXRIOT_ADMIN_PASSWORD_RESET=1` and `LUXRIOT_ADMIN_PASSWORD=...` for one restart.
- Roles: `admin` (full access), `redactor` (redactor + debug + history), `support` (debug + history), `client` (history), `anonymous` (no history).
- Login supports username or email.
- Role-based system prompts are editable in `Administrator tools → System prompt` and are applied per logged-in user role.

## Docs editor (alpha)

- Open `Documentation` → select a page → click the pencil icon to enter edit mode.
- Access is permission-based: `docs_edit` enables editing, `docs_publish` enables publishing. Admin can toggle these per user.
- If a user has `docs_edit` but not `docs_publish`, they can submit a publish request for admin approval.
- Upload screenshots in edit mode (Image button) to store under `datastore/<version>/assets/user/...`.

Auth endpoints:
- `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`
- Admin: `GET/POST/PATCH /auth/users` (supports `disabled: true/false`), `POST /auth/users/{id}/password/reset`
- Self: `POST /auth/password/change`
- Publish requests: `GET /admin/publish-requests`, `POST /docs/page/{doc_id}/{page_id}/publish/request`, `POST /admin/publish-requests/{doc_id}/{page_id}/approve|reject`

## Preview deployment: VPS gateway + office backend

Recommended preview topology:

- Put the public DNS name and TLS certificate on the VPS.
- Serve `index.html` from the VPS or from the FastAPI app behind the VPS.
- Reverse-proxy backend paths to the office backend through a private tunnel.
- Keep LM Studio, `datastore/`, `docs/`, and `backend/data/app.sqlite` on the office server.
- Add a customer IP allowlist on the VPS/firewall until OAuth is added.

For the lowest-risk setup, keep the browser and API same-origin:

```text
https://assistant.example.com/          -> static UI
https://assistant.example.com/auth/...  -> office backend through tunnel
https://assistant.example.com/docs/...  -> office backend through tunnel
https://assistant.example.com/chat/...  -> office backend through tunnel
https://assistant.example.com/assets/... -> office backend through tunnel
```

If the static UI and API are intentionally split, define this before loading `index.html`:

```html
<script>
  window.LUXRIOT_CONFIG = {
    apiBase: "https://api.assistant.example.com",
    apiCredentials: "include"
  };
</script>
```

The URL parameter `?api=https://...` still overrides the API base for quick debugging.

## Environment variables

- `LMSTUDIO_BASE_URL` (default `http://localhost:1234`)
- `LMSTUDIO_MODEL` (optional; auto-detected if unset)
- `LUXRIOT_APP_VERSION` (overrides `VERSION`)
- `LUXRIOT_DOCS_VERSION` (default `evo_1_32`)
- `LUXRIOT_DOCS_DIR` (default `docs`)
- `LUXRIOT_DATASTORE_DIR` (default `datastore`)
- `LUXRIOT_APP_DB_PATH` (default `backend/data/app.sqlite`)
- `LUXRIOT_AUTH_SECRET` (required for stable production sessions)
- `LUXRIOT_ADMIN_USERNAME` / `LUXRIOT_ADMIN_PASSWORD` (bootstrap admin credentials)
- `LUXRIOT_TRUSTED_HOSTS` (comma-separated hosts accepted by FastAPI, default `*`)
- `LUXRIOT_CORS_ORIGINS` (comma-separated allowed origins, default `*`)
- `LUXRIOT_CORS_ALLOW_CREDENTIALS` (`1` only for explicit cross-origin cookie use)
- `LUXRIOT_COOKIE_SECURE` (`1` behind HTTPS)
- `LUXRIOT_COOKIE_SAMESITE` (`lax`, `strict`, or `none`; use `none` only with HTTPS cross-origin)
- `LUXRIOT_COOKIE_DOMAIN` (optional cookie domain)
- `LUXRIOT_RAWDOCS_REQUIRE_AUTH` (default `1`)
- `LUXRIOT_ASSETS_REQUIRE_AUTH` (default `1`)

Example same-origin HTTPS preview:

```bash
export LUXRIOT_AUTH_SECRET='replace-with-long-random-value'
export LUXRIOT_TRUSTED_HOSTS='assistant.example.com,127.0.0.1,localhost'
export LUXRIOT_COOKIE_SECURE=1
export LUXRIOT_COOKIE_SAMESITE=lax
export LUXRIOT_RAWDOCS_REQUIRE_AUTH=1
export LUXRIOT_ASSETS_REQUIRE_AUTH=1
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Example split-origin preview:

```bash
export LUXRIOT_AUTH_SECRET='replace-with-long-random-value'
export LUXRIOT_TRUSTED_HOSTS='api.assistant.example.com,127.0.0.1,localhost'
export LUXRIOT_CORS_ORIGINS='https://assistant.example.com'
export LUXRIOT_CORS_ALLOW_CREDENTIALS=1
export LUXRIOT_COOKIE_SECURE=1
export LUXRIOT_COOKIE_SAMESITE=none
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

## MCP server

See `mcp-server/README.md` and `mcp-server/mcp.sample.json`.

## Docker preview deployment

Two-container preview deployment files are available:

- `docker-compose.cloud.yml` - cloud face container with Caddy, static UI, HTTPS, and reverse proxy.
- `docker-compose.cloud.selfsigned.yml` - self-signed/internal TLS override for lab demos.
- `docker-compose.office.yml` - office backend container with mounted docs/datastore/app DB.

See `deploy/README.md` for tunnel and startup commands.
