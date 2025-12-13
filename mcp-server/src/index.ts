import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

type DocsSearchResponse = {
  chunks: Array<{
    doc_id: string;
    page_id: string;
    heading_path: string[];
    text: string;
    score: number;
  }>;
  citations: Array<{
    title: string;
    doc_id: string;
    page_id: string;
    anchor?: string | null;
    source_path: string;
  }>;
  images: Array<{
    url: string;
    doc_id: string;
    page_id: string;
    near_heading?: string | null;
  }>;
};

const backendUrl = (process.env.LUXRIOT_BACKEND_URL || "http://localhost:8000").replace(/\/$/, "");
const DEFAULT_TIMEOUT_MS = 15_000;
const DEFAULT_MAX_BYTES = 1_000_000; // 1MB
const DEFAULT_MAX_CHARS = 20_000;

function isHttpUrl(input: string): boolean {
  try {
    const u = new URL(input);
    return u.protocol === "http:" || u.protocol === "https:";
  } catch {
    return false;
  }
}

function decodeHtmlEntities(input: string): string {
  return input
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => String.fromCodePoint(parseInt(hex, 16)))
    .replace(/&#([0-9]+);/g, (_, dec) => String.fromCodePoint(parseInt(dec, 10)));
}

function stripTagsKeepText(html: string): string {
  let s = html;
  s = s.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, "");
  s = s.replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, "");
  s = s.replace(/<noscript\b[^>]*>[\s\S]*?<\/noscript>/gi, "");
  s = s.replace(/<br\s*\/?>/gi, "\n");
  s = s.replace(/<\/(p|div|li|tr|h[1-6])\s*>/gi, "\n");
  s = s.replace(/<li\b[^>]*>/gi, "- ");
  s = s.replace(/<[^>]+>/g, "");
  s = decodeHtmlEntities(s);
  s = s.replace(/[ \t]+\n/g, "\n");
  s = s.replace(/\n{3,}/g, "\n\n");
  return s.trim();
}

function extractHtmlTitle(html: string): string | null {
  const m = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
  if (!m) return null;
  const title = stripTagsKeepText(m[1] || "");
  return title || null;
}

async function readTextLimited(resp: Response, maxBytes: number): Promise<{ text: string; truncated: boolean; bytes: number }> {
  if (!resp.body) return { text: "", truncated: false, bytes: 0 };
  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  let truncated = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    const next = received + value.byteLength;
    if (next > maxBytes) {
      const take = Math.max(0, maxBytes - received);
      if (take > 0) chunks.push(value.slice(0, take));
      received = maxBytes;
      truncated = true;
      try {
        await reader.cancel();
      } catch {
        // ignore
      }
      break;
    }
    chunks.push(value);
    received = next;
  }

  const buf = Buffer.concat(chunks.map((c) => Buffer.from(c)));
  return { text: buf.toString("utf8"), truncated, bytes: received };
}

async function fetchWithTimeout(url: string, opts: RequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...opts, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

function normalizeDdgUrl(raw: string): string {
  try {
    const u = new URL(raw, "https://duckduckgo.com");
    const uddg = u.searchParams.get("uddg");
    if (uddg) return decodeURIComponent(uddg);
    return u.toString();
  } catch {
    return raw;
  }
}

function extractDdgResults(html: string, k: number): Array<{ title: string; url: string; snippet: string }> {
  const out: Array<{ title: string; url: string; snippet: string }> = [];
  const seen = new Set<string>();

  const linkRe = /<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
  let m: RegExpExecArray | null;
  while ((m = linkRe.exec(html)) !== null) {
    const href = normalizeDdgUrl(m[1] || "");
    const title = stripTagsKeepText(m[2] || "");
    if (!href || !title) continue;
    if (seen.has(href)) continue;
    seen.add(href);

    const after = html.slice(linkRe.lastIndex, linkRe.lastIndex + 2500);
    let snippet = "";
    const snipM =
      after.match(/<a[^>]*class="result__snippet"[^>]*>([\s\S]*?)<\/a>/i) ||
      after.match(/<div[^>]*class="result__snippet"[^>]*>([\s\S]*?)<\/div>/i);
    if (snipM && snipM[1]) snippet = stripTagsKeepText(snipM[1]);

    out.push({ title, url: href, snippet });
    if (out.length >= k) break;
  }
  return out;
}

const server = new Server(
  { name: "luxriot-mcp", version: "0.1.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "luxriot_docs_query",
        description: "Search Luxriot EVO 1.32 docs (hybrid keyword+semantic) and return a context pack.",
        inputSchema: {
          type: "object",
          properties: {
            query: { type: "string", description: "Search query" },
            k: { type: "integer", description: "Number of chunks to return (1-25)", minimum: 1, maximum: 25 }
          },
          required: ["query"]
        }
      },
      {
        name: "duckduckgo_search",
        description: "Search the public web via DuckDuckGo HTML (scrape) and return top results.",
        inputSchema: {
          type: "object",
          properties: {
            query: { type: "string", description: "Search query" },
            k: { type: "integer", description: "Number of results to return (1-10)", minimum: 1, maximum: 10 }
          },
          required: ["query"]
        }
      },
      {
        name: "fetch_url",
        description: "Fetch a URL (http/https) and return readable text (HTML is stripped).",
        inputSchema: {
          type: "object",
          properties: {
            url: { type: "string", description: "URL to fetch (http/https only)" },
            max_chars: { type: "integer", description: "Max characters in returned text (1000-50000)", minimum: 1000, maximum: 50000 },
            timeout_ms: { type: "integer", description: "Timeout in ms (1000-60000)", minimum: 1000, maximum: 60000 }
          },
          required: ["url"]
        }
      }
    ]
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const name = request.params.name;

  if (name === "luxriot_docs_query") {
    const args = (request.params.arguments || {}) as { query?: unknown; k?: unknown };
    const query = typeof args.query === "string" ? args.query : "";
    const k = typeof args.k === "number" ? args.k : 8;

    if (!query.trim()) {
      return {
        isError: true,
        content: [{ type: "text", text: "Invalid arguments: 'query' must be a non-empty string." }]
      };
    }

    const resp = await fetch(`${backendUrl}/docs/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query, k })
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      return {
        isError: true,
        content: [{ type: "text", text: `Backend /docs/search failed (${resp.status}): ${text}` }]
      };
    }

    const data = (await resp.json()) as DocsSearchResponse;
    return {
      content: [{ type: "text", text: JSON.stringify(data, null, 2) }]
    };
  }

  if (name === "duckduckgo_search") {
    const args = (request.params.arguments || {}) as { query?: unknown; k?: unknown };
    const query = typeof args.query === "string" ? args.query : "";
    const k = typeof args.k === "number" ? Math.max(1, Math.min(10, Math.floor(args.k))) : 5;

    if (!query.trim()) {
      return {
        isError: true,
        content: [{ type: "text", text: "Invalid arguments: 'query' must be a non-empty string." }]
      };
    }

    try {
      const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
      const resp = await fetchWithTimeout(
        url,
        {
          method: "GET",
          headers: { "user-agent": "luxriot-mcp/0.1 (+https://localhost)", accept: "text/html" }
        },
        DEFAULT_TIMEOUT_MS
      );

      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        return {
          isError: true,
          content: [{ type: "text", text: `DuckDuckGo request failed (${resp.status}): ${text}` }]
        };
      }

      const { text: html, truncated } = await readTextLimited(resp, DEFAULT_MAX_BYTES);
      const results = extractDdgResults(html, k);
      const payload = { query, k, results, source: "duckduckgo_html", truncated_html: truncated };

      return {
        content: [{ type: "text", text: JSON.stringify(payload, null, 2) }]
      };
    } catch (e: any) {
      return {
        isError: true,
        content: [{ type: "text", text: `DuckDuckGo request failed: ${e?.message || String(e)}` }]
      };
    }
  }

  if (name === "fetch_url") {
    const args = (request.params.arguments || {}) as { url?: unknown; max_chars?: unknown; timeout_ms?: unknown };
    const url = typeof args.url === "string" ? args.url : "";
    const maxChars =
      typeof args.max_chars === "number" ? Math.max(1000, Math.min(50_000, Math.floor(args.max_chars))) : DEFAULT_MAX_CHARS;
    const timeoutMs =
      typeof args.timeout_ms === "number" ? Math.max(1000, Math.min(60_000, Math.floor(args.timeout_ms))) : DEFAULT_TIMEOUT_MS;

    if (!url.trim() || !isHttpUrl(url)) {
      return {
        isError: true,
        content: [{ type: "text", text: "Invalid arguments: 'url' must be a valid http/https URL." }]
      };
    }

    try {
      const resp = await fetchWithTimeout(
        url,
        {
          method: "GET",
          headers: {
            "user-agent": "luxriot-mcp/0.1 (+https://localhost)",
            accept: "text/html,text/plain,application/json;q=0.9,*/*;q=0.1"
          }
        },
        timeoutMs
      );

      const contentType = resp.headers.get("content-type") || "";
      const isText = /^text\//i.test(contentType) || /json/i.test(contentType) || contentType === "";

      if (!resp.ok) {
        const { text } = await readTextLimited(resp, DEFAULT_MAX_BYTES);
        return {
          isError: true,
          content: [{ type: "text", text: `Fetch failed (${resp.status}): ${text}` }]
        };
      }

      if (!isText) {
        return {
          isError: true,
          content: [{ type: "text", text: `Unsupported content-type: ${contentType || "(unknown)"}` }]
        };
      }

      const { text: raw, truncated: truncatedBytes, bytes } = await readTextLimited(resp, DEFAULT_MAX_BYTES);
      let textOut = raw;
      let title: string | null = null;
      if (/html/i.test(contentType)) {
        title = extractHtmlTitle(raw);
        textOut = stripTagsKeepText(raw);
      }

      const truncatedChars = textOut.length > maxChars;
      if (truncatedChars) textOut = textOut.slice(0, maxChars);

      const payload = {
        url,
        final_url: resp.url || url,
        status: resp.status,
        content_type: contentType,
        title,
        bytes,
        truncated: truncatedBytes || truncatedChars,
        text: textOut
      };

      return {
        content: [{ type: "text", text: JSON.stringify(payload, null, 2) }]
      };
    } catch (e: any) {
      return {
        isError: true,
        content: [{ type: "text", text: `Fetch failed: ${e?.message || String(e)}` }]
      };
    }
  }

  return {
    isError: true,
    content: [{ type: "text", text: `Unknown tool: ${name}` }]
  };
});

const transport = new StdioServerTransport();
await server.connect(transport);
