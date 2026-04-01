# code_search_mcp.py

import os
from typing import List

import httpx
from fastmcp import FastMCP

# Where your FastAPI backend lives.
# Can be overridden with env var if you want.
BACKEND_URL = os.environ.get("CODE_BACKEND_URL", "http://127.0.0.1:8000")

mcp = FastMCP("code-search-mcp")


@mcp.tool
def search_code(query: str, top_k: int = 5) -> List[str]:
    """
    Search your local code-snippet database.

    Parameters
    ----------
    query : str
        Natural language description of the code you want.
    top_k : int
        Max number of snippets to return.
    """
    payload = {"query": query, "top_k": top_k}

    try:
        resp = httpx.post(
            f"{BACKEND_URL}/search_code",
            json=payload,
            timeout=20.0,
        )
        resp.raise_for_status()
    except Exception as e:
        # The MCP host surfaces this string in the tool result
        return [f"Error calling backend: {e!r}"]

    # Assuming your FastAPI endpoint returns `list[str]`
    data = resp.json()
    if not isinstance(data, list):
        return [f"Unexpected response from backend: {data!r}"]

    # You can also join them into one string if you prefer, but list[str] is fine.
    return data


if __name__ == "__main__":
    # STDIO transport for local MCP hosts.
    mcp.run()
