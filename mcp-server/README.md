# luxriot-mcp server

This MCP server exposes tools:

- `luxriot_docs_query({ query, k })` → calls backend `POST /docs/search`
- `duckduckgo_search({ query, k })` → DuckDuckGo HTML scrape
- `fetch_url({ url, max_chars?, timeout_ms? })` → fetch a URL and return readable text

## Build & run

```bash
cd mcp-server
npm install
npm run build
npm start
```

## Config snippet

See `mcp-server/mcp.sample.json`.
