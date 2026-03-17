"""MCP server — exposes Alexandria search to AI agents via stdio transport.

Tools:
  - search_code: Search within a specific context
  - search_all: Search across all contexts
  - list_contexts: List all indexed contexts with stats
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from alexandria.config import Config
from alexandria.embedder import BaseEmbedder, create_embedder
from alexandria.store import SearchResult, Store

mcp = FastMCP(
    name="alexandria",
    instructions=(
        "Alexandria provides semantic code search over indexed codebases. "
        "Use search_code to search within a specific codebase context, "
        "search_all to search across all indexed codebases, "
        "and list_contexts to see what's available."
    ),
)

# These are initialized lazily on first use
_config: Config | None = None
_store: Store | None = None
_embedder: BaseEmbedder | None = None


def _get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config


def _get_store() -> Store:
    global _store
    if _store is None:
        _store = Store(_get_config())
    return _store


def _get_embedder() -> BaseEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = create_embedder(_get_config())
    return _embedder


def _format_results(results: list[SearchResult]) -> str:
    """Format search results into a readable string for the agent."""
    if not results:
        return "No results found."

    parts = []
    for i, r in enumerate(results, 1):
        header = f"### Result {i} (score: {r.score:.4f})"
        location = f"**{r.file}** L{r.start_line}-{r.end_line}"
        if r.symbol:
            location += f" (`{r.symbol}`)"
        location += f" [{r.language}]"

        lines = [header, location, ""]

        if r.context_before:
            lines.append("```")
            lines.append(f"... (context before)")
            lines.append(r.context_before)
            lines.append("```")
            lines.append("")

        lines.append(f"```{r.language}")
        lines.append(r.text)
        lines.append("```")

        if r.context_after:
            lines.append("")
            lines.append("```")
            lines.append(r.context_after)
            lines.append(f"... (context after)")
            lines.append("```")

        parts.append("\n".join(lines))

    return "\n\n---\n\n".join(parts)


@mcp.tool()
def search_code(
    query: str,
    context: str,
    limit: int = 10,
    language: str | None = None,
) -> str:
    """Search for code within a specific indexed codebase context.

    Args:
        query: Natural language description of what you're looking for.
               e.g. "authentication handler", "database connection setup"
        context: Name of the indexed codebase context to search in.
                 Use list_contexts to see available contexts.
        limit: Maximum number of results to return (default: 10).
        language: Optional language filter (e.g. "python", "typescript").

    Returns:
        Matching code chunks with file locations and surrounding context.
    """
    embedder = _get_embedder()
    store = _get_store()

    query_vector = embedder.embed(query)
    results = store.search(
        context=context,
        query_vector=query_vector,
        limit=limit,
        language_filter=language,
    )

    return _format_results(results)


@mcp.tool()
def search_all(
    query: str,
    limit: int = 10,
) -> str:
    """Search for code across ALL indexed codebase contexts.

    Useful when you don't know which codebase contains the relevant code,
    or when searching across multiple projects.

    Args:
        query: Natural language description of what you're looking for.
        limit: Maximum number of results to return (default: 10).

    Returns:
        Matching code chunks from any context, ranked by relevance.
    """
    embedder = _get_embedder()
    store = _get_store()

    query_vector = embedder.embed(query)
    results = store.search_all(query_vector=query_vector, limit=limit)

    return _format_results(results)


@mcp.tool()
def list_contexts() -> str:
    """List all indexed codebase contexts and their statistics.

    Returns:
        A list of context names with point counts and status.
    """
    store = _get_store()
    contexts = store.list_contexts()

    if not contexts:
        return (
            "No contexts indexed. Use `alexandria index --context NAME PATH` to index a codebase."
        )

    lines = ["## Indexed Contexts", ""]
    for ctx in sorted(contexts):
        stats = store.get_context_stats(ctx)
        lines.append(f"- **{ctx}**: {stats['points']} chunks indexed (status: {stats['status']})")

    return "\n".join(lines)


def run_stdio() -> None:
    """Run the MCP server with stdio transport."""
    mcp.run(transport="stdio")
