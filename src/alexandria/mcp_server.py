"""MCP server — exposes Alexandria search to AI agents via stdio transport.

Tools:
  - search_code: Search within a specific context
  - search_all: Search across all contexts
  - list_contexts: List all indexed contexts with stats

At query time the server reads each collection's stored embedding metadata
(model, backend, dimension) and creates the matching embedder automatically.
This prevents the "vector dimension mismatch" error that occurs when the
server's default environment variables differ from the model used at index
time.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from alexandria.config import DIM_TO_DEFAULT_MODEL, CollectionEmbedInfo, Config
from alexandria.embedder import BaseEmbedder, create_embedder
from alexandria.store import SearchResult, Store

log = logging.getLogger(__name__)

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
# Default embedder (used when no per-context override is needed).
_embedder: BaseEmbedder | None = None
# Per-context embedder cache keyed by (backend, model).
_embedder_cache: dict[tuple[str, str], BaseEmbedder] = {}


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
    """Return the default embedder (based on env-var config)."""
    global _embedder
    if _embedder is None:
        _embedder = create_embedder(_get_config())
    return _embedder


def _get_embedder_for_context(context: str) -> BaseEmbedder:
    """Return an embedder whose output dimension matches the collection.

    1. Read the collection's stored metadata (backend + model).
    2. If it matches the default embedder, reuse it.
    3. Otherwise build a context-specific embedder and cache it.
    4. If no metadata is stored (legacy index), fall back to
       :data:`DIM_TO_DEFAULT_MODEL` for auto-detection by dimension.
    5. As a last resort, return the default embedder and let the
       dimension validation in :meth:`Store.search` produce a clear error.

    Args:
        context: The context name to look up.

    Returns:
        A :class:`BaseEmbedder` that produces vectors matching the collection.
    """
    store = _get_store()
    embed_info: CollectionEmbedInfo | None = store.get_collection_embed_info(context)

    default = _get_embedder()
    if embed_info is None:
        return default

    backend = embed_info.embed_backend
    model = embed_info.embed_model

    # If metadata is missing (legacy collection), try to infer from dimension.
    if not backend or not model:
        dim_defaults = DIM_TO_DEFAULT_MODEL.get(embed_info.embed_dim)
        if dim_defaults is not None:
            backend, model = dim_defaults
            log.info(
                "Context '%s' has no stored model metadata; inferred %s/%s from dimension %d",
                context,
                backend,
                model,
                embed_info.embed_dim,
            )
        else:
            # Cannot determine — fall back to default and let validation catch it.
            return default

    # If the default embedder already matches, reuse it.
    default_cfg = _get_config()
    if backend == default_cfg.embed_backend and model == default_cfg.embed_model:
        return default

    # Check the cache.
    cache_key = (backend, model)
    if cache_key in _embedder_cache:
        return _embedder_cache[cache_key]

    # Build a new config + embedder for this backend/model.
    log.info(
        "Creating embedder for context '%s': backend=%s model=%s (dim=%d)",
        context,
        backend,
        model,
        embed_info.embed_dim,
    )
    ctx_config = Config(
        embed_backend=backend,
        embed_model=model,
        embed_dim=embed_info.embed_dim,
        # Carry over connection settings from the default config so the user
        # only needs to set these once via env vars.
        qdrant_url=default_cfg.qdrant_url,
        ollama_host=default_cfg.ollama_host,
        embed_api_url=default_cfg.embed_api_url,
        embed_api_key=default_cfg.embed_api_key,
    )
    embedder = create_embedder(ctx_config)
    _embedder_cache[cache_key] = embedder
    return embedder


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
            lines.append("... (context before)")
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
            lines.append("... (context after)")
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
    embedder = _get_embedder_for_context(context)
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

    If different contexts were indexed with different embedding models the
    query is embedded separately for each context using the matching model.

    Args:
        query: Natural language description of what you're looking for.
        limit: Maximum number of results to return (default: 10).

    Returns:
        Matching code chunks from any context, ranked by relevance.
    """
    store = _get_store()
    contexts = store.list_contexts()

    all_results: list[SearchResult] = []
    # Group contexts by (backend, model) so we only embed once per model.
    model_contexts: dict[tuple[str, str], list[str]] = {}
    for ctx in contexts:
        embedder = _get_embedder_for_context(ctx)
        key = (embedder.config.embed_backend, embedder.model)
        model_contexts.setdefault(key, []).append(ctx)

    for (_backend, _model), ctx_list in model_contexts.items():
        # Pick the embedder for this group (all use the same model).
        embedder = _get_embedder_for_context(ctx_list[0])
        query_vector = embedder.embed(query)
        for ctx in ctx_list:
            try:
                ctx_results = store.search(ctx, query_vector, limit=limit)
                all_results.extend(ctx_results)
            except Exception as exc:
                log.warning("Skipping context '%s' in search_all: %s", ctx, exc)

    all_results.sort(key=lambda r: r.score, reverse=True)
    n = limit or store.config.search_limit
    return _format_results(all_results[:n])


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
        detail = f"{stats['points']} chunks indexed (status: {stats['status']})"
        embed_model = stats.get("embed_model", "")
        embed_dim = stats.get("embed_dim", "")
        if embed_model and embed_model != "—":
            detail += f", model: {embed_model} ({embed_dim}-dim)"
        lines.append(f"- **{ctx}**: {detail}")

    return "\n".join(lines)


def run_stdio() -> None:
    """Run the MCP server with stdio transport."""
    mcp.run(transport="stdio")
