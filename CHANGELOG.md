# Changelog

## Luxriot SA 0.2.6

- Retrieval:
  - Tuned documentation search intent routing for camera, folders, Modbus, storage, license, and upgrade scenarios.
  - Added session-aware retrieval query expansion for follow-up questions.
  - Set demo-safe defaults to BM25 without summary/MMR routing.
- Documentation workflow:
  - Improved related screenshot placement and filtering in assistant answers.
  - Added support-oriented ingest/reporting improvements for broken and repaired images.
  - Added partner preview one-page HTML presentation.
- UI:
  - Restored the assistant UI toward the stable dark layout while keeping docs-reader controls and screenshot fixes.

## Luxriot SA 0.2.4

- Markdown conventions (global):
  - Global conventions file and prompt injection (`{{markdown_conventions}}`) so LLM output and ingested docs follow the same rules.
  - Conventions hash recorded per ingested page (`md_conventions_hash`) for traceability.
  - New PDF/layout directives supported end-to-end: `:::luxriot-page-break` and `:::luxriot-blank-line`.
- HTML -> Markdown ingestion:
  - Help+Manual note boxes are converted into admonitions without breaking inline formatting (prevents word-by-word splitting inside callouts).
  - Added CAUTION support end-to-end (ingestion, viewer rendering, PDF export).
- Retrieval:
  - Fixed chat streaming crash when reranker config is present (`reranker_enabled` undefined).
  - Embedding model id is configurable; default updated to `text-embedding-qwen3-embedding-0.6b`.
- Dev tooling:
  - Added a lightweight retrieval evaluation harness and dataset under `backend/eval/` and `backend/cli/`.

## Luxriot SA 0.2.3

- Home cards:
  - Truncated card titles show a tooltip with the full title.

## Luxriot SA 0.2.2

- Home cards:
  - Shows recent chats in the home view.

## Luxriot SA 0.2.1

- Home cards (anonymous/client UX):
  - Home button behavior and role-targeted quick actions.
  - Card dismissals persisted per user.

## Luxriot SA 0.2.0

- Baseline production build.
