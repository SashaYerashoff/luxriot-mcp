# luxriot-mcp server

This MCP server exposes a single tool:

- `luxriot_docs_query({ query, k })` → calls backend `POST /docs/search`

## Build & run

```bash
cd mcp-server
npm install
npm run build
npm start
```

## Config snippet

See `mcp-server/mcp.sample.json`.

