# AGENTS.md — Alexandria

Alexandria is a semantic code search tool that indexes codebases into a Qdrant
vector database using tree-sitter AST chunking and Ollama embeddings, and
exposes search to AI agents via MCP (Model Context Protocol) stdio transport.

Python 3.11+ | src-layout | Nix flake for builds and dev shell.

## Build & Run

```bash
# Enter the dev shell (provides Python, Qdrant, Ollama, fd, ruff, mypy, pytest)
nix develop

# Install in editable mode (from within dev shell)
pip install -e .

# Build the Nix package
nix build

# Build Docker image
nix build .#docker
```

CLI entry points: `alexandria` or `alex` (alias).

```bash
alex setup              # Pull Ollama model, verify Qdrant
alex index -c CTX PATH  # Index a codebase into context CTX
alex serve              # Start MCP server (stdio)
alex list               # Show indexed contexts
alex drop -c CTX        # Delete a context
```

## Lint & Type-Check

```bash
ruff check .                    # Lint (E, F, W, I, N, UP, B, A, SIM rules)
ruff check --fix .              # Lint with auto-fix
ruff format .                   # Format (line-length 100)
mypy src/                       # Type-check (strict mode)
```

Ruff is the primary linter AND formatter. Line length is 100. Mypy runs in
strict mode (`warn_return_any`, `warn_unused_configs` enabled).

## Test

```bash
pytest                                  # Run all tests
pytest tests/test_chunker.py            # Run one test file
pytest tests/test_chunker.py::test_foo  # Run one test function
pytest -k "keyword"                     # Run tests matching keyword
pytest -x                               # Stop on first failure
```

Test config is in `pyproject.toml` under `[tool.pytest.ini_options]`.
Test directory: `tests/`. Async tests use `pytest-asyncio`.

## Code Style

### Imports

- Every module starts with `from __future__ import annotations` (first import).
- Import order enforced by ruff's isort rule (`I`):
  1. `__future__`
  2. Standard library
  3. Third-party packages
  4. Local (`alexandria.*`)
- Heavy local imports are lazy (inside functions) in `cli.py` to keep CLI
  startup fast. Only do this in CLI command functions, not in library code.

### Formatting

- Line length: **100** characters (configured in `[tool.ruff]`).
- Formatter: **ruff format** (not black — ruff is authoritative).
- Trailing commas in multi-line function args and collections.
- Spaces around `=` in keyword arguments only in type annotations.

### Type Annotations

- Full type annotations on **all** function signatures (args + return type).
- Modern syntax — use `str | None` not `Optional[str]`, use `list[str]` not
  `List[str]`, use `dict[str, str]` not `Dict[str, str]`.
- Mypy strict mode is enforced. Do not use `# type: ignore` without a
  specific error code (e.g. `# type: ignore[arg-type]`).

### Naming Conventions

- `snake_case` — functions, variables, module names.
- `PascalCase` — classes (`Config`, `Chunk`, `Store`, `SearchResult`).
- `UPPER_SNAKE_CASE` — module-level constants (`EXTENSION_MAP`, `CHUNK_NODE_TYPES`).
- `_leading_underscore` — private/internal functions and module-level globals
  (`_get_config`, `_store`, `_ensure_collection`).

### Data Structures

- Use `@dataclass` for value objects / data containers (e.g. `Config`, `Chunk`,
  `SearchResult`).
- Use classes for stateful service objects (e.g. `Store`, `Embedder`).
- Config values read from `os.environ.get()` with defaults in dataclass fields.

### Error Handling

- Graceful degradation: try preferred path, fall back on failure
  (e.g. `fd` -> pure-Python walk in `discovery.py`).
- Check service availability before operations with `is_available()` methods;
  print user-friendly messages and `sys.exit(1)` on failure.
- Use specific exception types (`OSError`, `PermissionError`, `ValueError`,
  `KeyError`) — only use bare `except Exception` for non-critical checks
  (availability probes, stats lookups).
- Prefer early returns to avoid deep nesting.

### Documentation

- Module-level docstrings on every file describing its purpose.
- Class and public method docstrings in Google style (`Args:`, `Returns:`).
- Inline comments for non-obvious logic.

### Other Patterns

- Deterministic IDs: `Chunk.id` uses SHA-256 hash -> UUID for idempotent
  Qdrant upserts.
- Rich library for all terminal output (colored text, progress bars, tables).
- Lazy global singletons in `mcp_server.py` — `_config`, `_store`,
  `_embedder` initialized on first use.
- Batch processing with configurable sizes (embedding: 64, Qdrant upsert: 100).
- Binary file detection via null-byte heuristic in first 8KB.

## Project Structure

```
src/alexandria/
  __init__.py       # Package init, version
  cli.py            # Click CLI commands (index, serve, list, drop, setup)
  config.py         # Config dataclass, extension/language maps, chunk node types
  chunker.py        # Tree-sitter AST chunking + sliding-window fallback
  discovery.py      # File discovery via fd (with pure-Python fallback)
  embedder.py       # Ollama embedding client
  mcp_server.py     # FastMCP server (search_code, search_all, list_contexts)
  store.py          # Qdrant vector store wrapper
tests/              # pytest test directory
pyproject.toml      # Project metadata, dependencies, tool config
flake.nix           # Nix flake (package, devShell, NixOS module, Docker)
```

## Dependencies & Runtime

- **Qdrant** — vector database (must be running at `QDRANT_URL`)
- **Ollama** — embedding model server (must be running at `OLLAMA_HOST`)
- **fd** — fast file discovery (wrapped onto PATH by Nix; pure-Python fallback)
- **tree-sitter** + **tree-sitter-language-pack** — AST parsing

Environment variables (with defaults):
- `QDRANT_URL` = `http://localhost:6333`
- `OLLAMA_HOST` = `http://localhost:11434`
- `ALEXANDRIA_EMBED_MODEL` = `nomic-embed-text`
- `ALEXANDRIA_DEV` = set to `1` in dev shell

## Ruff Rule Set Reference

| Code | Scope |
|------|-------|
| E | pycodestyle errors |
| F | pyflakes |
| W | pycodestyle warnings |
| I | isort (import sorting) |
| N | pep8-naming |
| UP | pyupgrade (modern syntax) |
| B | flake8-bugbear |
| A | flake8-builtins |
| SIM | flake8-simplify |
