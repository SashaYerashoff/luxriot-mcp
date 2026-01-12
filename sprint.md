# Sprint: RAG 01 — Topological Screenshots + Two‑Pass Retrieval

## Objective
Improve retrieval consistency and enforce topological screenshot placement, so images appear only next to the text that cites their source chunks. Add an optional two‑pass retrieval flow using summary routing.

## Principles
- No unrelated screenshots (only images attached to cited chunks).
- Evidence-first: answers must cite chunks for each factual step.
- Keep UI behavior deterministic (no heuristic screenshot dumping).

## Scope
1. **Chunk → image contract**
   - Expose chunk-level image lists alongside retrieval results.
   - Preserve chunk IDs through to the answer pipeline.
2. **Topological screenshot placement**
   - Insert images only after paragraphs/steps containing citations for those chunks.
   - Remove/disable heuristic “Screenshots” block if not tied to citations.
3. **Summary routing (optional two‑pass retrieval)**
   - Build a summary index (per page or per section).
   - Pass 1: BM25 over summaries to select candidate pages/sections.
   - Pass 2: BM25/embedding/hybrid over chunks within selected candidates.
4. **Controls + observability**
   - Admin toggles for summary routing and image placement limits.
   - Debug output to inspect candidate selection and image attachment.

## Non‑Goals (this sprint)
- Automated screenshot selection via vision.
- Full multi‑turn tool‑calling RAG loop in `/chat`.

## Deliverables
- Deterministic screenshot placement tied to cited chunks.
- Summary index + two‑pass retrieval (behind a toggle).
- Updated prompts to encourage per‑step citations.
- Multi‑granularity chunking (topic/section/part) to improve recall and step‑level grounding.

## Acceptance Criteria
- If a chunk has no images, no images are inserted for it.
- Images never appear without a matching chunk citation.
- Two‑pass retrieval falls back to full search if routing finds nothing.
- Debug view can show which chunks contributed to each inserted image.

## Decisions (confirmed)
1. **Summary generation:** LLM‑based. Default model: `qwen3-vl-4b` (configurable per deployment).
2. **Granularity:** dynamic heading level based on token budget (H1 → H2 → H3 … until sections fit).
3. **Two‑pass retrieval:** ON by default.
4. **Screenshot placement:** no preference; pick most reliable implementation.
5. **Answer format:** prompt‑dependent (keep markdown + `[n]` citations by default).
6. **Image limits:** no artificial limit beyond topological relevance.
