# Agentic code memory

A small **retrieval layer for coding agents** — semantic search over your own snippets. Backend service and tooling so you can **store, embed, and recall** code via natural language. You store snippets in a local **ChromaDB** vector database; each snippet is summarized (via OpenAI) and embedded so you can query them in natural language and get matching source back.

## What’s in this repo

- **FastAPI app** (`code_search/main.py`) — HTTP API to add snippets and search them.
- **ChromaDB** — persistent vectors under `./chroma_db` (created on first use; not committed to git).
- **OpenAI** — embeddings (and description generation for indexing) using your `OPENAI_API_KEY`.
- **MCP server** (`mcp/code_search_mcp.py`) — optional [Model Context Protocol](https://modelcontextprotocol.io/) tool that calls your API’s `/search_code` endpoint from any MCP-capable client.

## API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/add_code` | Body: `code`, optional `user_description`. Indexes a new snippet. |
| `POST` | `/search_code` | Body: `query`, optional `top_k`. Returns a list of matching code strings. |

Default server URL in the MCP client: `http://127.0.0.1:8000` (override with `CODE_BACKEND_URL`).

## Setup

1. **Python 3.11+** recommended (project has been used with 3.13).

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set your OpenAI key:

   ```bash
   cp .env.example .env
   # edit .env — set OPENAI_API_KEY
   ```

## Run the API

From the repository root:

```bash
uvicorn code_search.main:app --reload --host 0.0.0.0 --port 8000
```

## MCP clients

Copy `.mcp.json.example` to `.mcp.json` (or merge into your editor config) and point the `args` path at `mcp/code_search_mcp.py` on your machine. Ensure the FastAPI app is running if the MCP tool should hit your local backend.

## Layout

```
code_search/     # FastAPI app, Chroma + OpenAI indexing (`handle_db.py`), optional `embedding_service.py`
mcp/             # STDIO MCP server that proxies search to the API
scripts/         # Ad-hoc scripts (e.g. test indexing)
```

## Security notes

- Never commit `.env`; only `.env.example` belongs in git.
- Treat API keys as secrets; rotate them if they are exposed.
