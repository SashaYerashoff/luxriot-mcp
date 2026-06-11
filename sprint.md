# Sprint: CLOUD 01 - VPS Gateway + Office Backend Preview

## Objective
Prepare the assistant for a controlled customer preview where the public entrypoint lives on a small VPS, while the model, documentation, datastore, and app DB stay on the office server.

## Target Architecture
- Customer browser connects to `https://assistant.example.com`.
- VPS terminates TLS and serves the static UI.
- VPS reverse-proxies backend paths (`/chat`, `/docs`, `/auth`, `/assets`, `/rawdocs`, etc.) to the office backend through a private tunnel.
- Office backend talks to LM Studio and local documentation only from inside the office network.
- Access is limited by source IP allowlist at the VPS/firewall layer for this pre-OAuth preview.

## Scope
1. Deployment configuration
   - Environment-driven CORS, trusted hosts, cookie security, and raw file access behavior.
   - Container-friendly docs/datastore/app DB paths.
   - Defaults remain local-dev friendly.
2. Same-origin gateway readiness
   - Static frontend can use default same-origin API routing behind a reverse proxy.
   - Frontend can also accept `window.LUXRIOT_CONFIG.apiBase` for split deployments.
3. File access boundaries
   - `/assets` and `/rawdocs` are no longer unguarded static file leaks by default.
   - Raw docs and assets are checked against the same role-level documentation access rules as catalog/page endpoints.
4. Deployment notes
   - Document recommended VPS + tunnel topology and production environment variables.
5. Preview containers
   - Cloud `face` container serves the UI, terminates TLS, and proxies backend paths.
   - Office `backend` container mounts docs, datastore, app DB, and talks to LM Studio.
   - Self-signed TLS override is available for lab runs.
6. Verification
   - Compile backend modules.
   - Check that role-filtered docs, raw docs, assets, and frontend API calls still work.

## Acceptance Criteria
- A logged-in user can use chat, docs reader, editor, screenshots, and PDF export through the VPS URL.
- Anonymous/client roles cannot directly load blocked raw docs or assets from denied documentation.
- Admin/support/redactor workflows keep working with protected file routes.
- The office backend does not need to expose LM Studio or the docs datastore directly to the Internet.
- The deployment can be configured without code edits.

## Out of Scope
- OAuth and public self-service signup.
- Multi-tenant customer isolation.
- Cloud-side model inference.
- Full Docker/Ansible production packaging.
- Long-term audit logging and rate limiting beyond basic preview hardening.

## Implementation Plan
1. Finish config hardening and runtime API wiring.
2. Protect raw docs and asset routes with role-aware checks.
3. Add README deployment notes and environment variable reference.
4. Add Docker preview scaffolding for cloud face and office backend.
5. Run focused compile/lint checks.
6. Smoke-test local server paths if the quick checks pass.

## Risks
- Direct cross-origin deployments require explicit CORS origins plus `SameSite=None; Secure` cookies; same-origin reverse proxy is lower risk for the demo.
- Raw source files are mapped through the ingested catalog. If a source file is not indexed, direct `/rawdocs` access will return 404 under protected mode.
- Asset access uses the asset path convention (`assets/<doc_id>/...` or `assets/user/<doc_id>/...`). Orphaned assets are staff-only.
