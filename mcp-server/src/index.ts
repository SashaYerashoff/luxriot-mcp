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

const server = new Server(
  { name: "luxriot-mcp", version: "0.1.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "luxriot_docs_query",
        description: "Search Luxriot EVO 1.32 docs (BM25 keyword search) and return a context pack.",
        inputSchema: {
          type: "object",
          properties: {
            query: { type: "string", description: "Search query" },
            k: { type: "integer", description: "Number of chunks to return (1-25)", minimum: 1, maximum: 25 }
          },
          required: ["query"]
        }
      }
    ]
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name !== "luxriot_docs_query") {
    return {
      isError: true,
      content: [{ type: "text", text: `Unknown tool: ${request.params.name}` }]
    };
  }

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
});

const transport = new StdioServerTransport();
await server.connect(transport);

