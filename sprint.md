# Sprint: DOC 02 — PDF Parity + Doc Composer

## Objective
Make PDF exports visually match the docs viewer and enable the LLM to generate clean, publish‑ready draft guides (with cover metadata) that open directly in the editor.

## Principles
- One markdown source of truth for viewer + PDF.
- Preserve structure (headings, lists, tables, admonitions) exactly.
- No hidden prompts; composer behavior is transparent and configurable.
- Drafts are editable and reviewable before publish.

## Scope
1. **PDF layout parity**
   - Match viewer styles for headings, body text, lists, tables, and admonitions.
   - Add robust inline emphasis inside lists and paragraphs.
   - Unicode coverage (Cyrillic + arrows) with font fallback.
   - Image sizing rules: block vs inline, plus explicit width overrides in markdown.
   - Cover + headers derived from editor fields only (no auto titles).
2. **Markdown normalization (non‑AI)**
   - Normalize input markdown into our conventions (heading levels, list spacing, admonitions).
   - Prepare for external import without “format drift.”
3. **Doc Composer flow**
   - New action: “Compose guide from chat answer” → opens editor with draft markdown.
   - Composer output includes cover metadata (type, title, image, custom text, version).
   - Preserve citations as links to the doc viewer anchors.
   - Allow save as draft → request publish → admin approval.
4. **Observability & UX**
   - Show composer metadata (model, prompt version, timestamp) inside editor (not in PDF).
   - Clear errors when PDF export skips unsupported elements.

## Deliverables
- PDF rendering that visually matches viewer for all supported markdown elements.
- Font fallback pipeline that renders non‑Latin text correctly.
- Markdown normalization utility used on import + composer outputs.
- Draft‑from‑chat flow for redactor/admin, with cover fields prefilled.

## Acceptance Criteria
- Lists + nested lists render identically in viewer and PDF.
- Tables and admonitions match viewer styling in PDF.
- Cyrillic + arrow symbols render correctly in PDF using fallback fonts.
- Composer output opens in editor with correct cover metadata and citations.

## Open Questions
- Preferred markdown syntax for explicit image widths (e.g., `{width=60%}` vs `![alt](url "w=60")`)?
- Should composer drafts include a “Sources” section or inline citations only?
- Which roles can invoke the composer action (admin + redactor only?)
