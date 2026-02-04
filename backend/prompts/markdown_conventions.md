# Luxriot SA Markdown Conventions (Global)

These conventions apply to all documentation output and LLM responses.
Manual edits override auto-ingestion output.

## Allowed blocks
- Headings: H1-H5 only.
- Paragraphs.
- Ordered and unordered lists (with nesting).
- Tables with a header separator line.
- Admonitions (TIP/INFO/WARNING/NOTE).
- Code blocks (fenced).
- Images and links.
- Horizontal rule (`---`).
- Fenced directives (Luxriot-specific).

## Headings
- Use **one** H1 per page (the page title).
- Sections start at H2.
- Do not use H6 or deeper.

## Lists
- Bullets: `-` only.
- Numbered steps: `1.` format (markdown auto-numbers).
- Nested lists: indent by 2 spaces.

## Tables
- Must include a header row and a separator row.
- Example:
  | Column | Column |
  | --- | --- |
  | Value | Value |

## Admonitions
Use the blockquote syntax:
> [!TIP] Optional title
> Body line 1
> Body line 2

Supported kinds: TIP, INFO, WARNING, CAUTION, NOTE.

## Images
- Standard: `![alt text](url)`
- Optional width override:
  - `![alt](url|width=60%)`
  - `![alt](url?width=480px)`
  - `![alt](url#width=60%)`

## Links
- Standard: `[label](https://example.com)`

## Code blocks
```
code here
```

## Luxriot directives (fenced)
These are safe, non-standard directives used for PDF layout.
They must be written exactly as shown:

**Page break:**
:::luxriot-page-break
:::

**Blank line (spacer):**
:::luxriot-blank-line
:::

## Normalization rules (ingestion)
During HTML ingestion we normalize to match these conventions:
- Additional H1 headings are downscaled to H2.
- Headings deeper than H5 are downscaled to H5.
- Bullet markers are normalized to `-`.
- Excess blank lines are collapsed.
- Help+Manual note boxes become TIP/INFO/WARNING/CAUTION/NOTE admonitions.
