# Alexandria

Semantic code search for AI agents. Alexandria indexes codebases into a
[Qdrant](https://qdrant.tech/) vector database using
[tree-sitter](https://tree-sitter.github.io/) AST chunking and
[Ollama](https://ollama.com/) embeddings, then exposes search via
[MCP](https://modelcontextprotocol.io/) (Model Context Protocol) so that AI
coding agents can find relevant code by meaning rather than exact text match.

## How It Works

1. **Discovery** -- finds source files using `fd` (respects `.gitignore`),
   with a pure-Python fallback.
2. **Chunking** -- parses each file with tree-sitter to extract semantically
   meaningful chunks at function/class boundaries. Falls back to a
   sliding-window approach for unsupported languages.
3. **Embedding** -- sends chunks to Ollama (`nomic-embed-text` by default) to
   produce 768-dimensional vectors.
4. **Storage** -- upserts vectors into Qdrant with metadata (file path, line
   range, symbol name, language, file hash for change detection).
5. **Search** -- an MCP server exposes `search_code`, `search_all`, and
   `list_contexts` tools over stdio, so any MCP-compatible agent can query
   indexed codebases with natural language.

## Requirements

- Python 3.11+
- [Qdrant](https://qdrant.tech/) running locally (default `http://localhost:6333`)
- [Ollama](https://ollama.com/) running locally (default `http://localhost:11434`)
- [fd](https://github.com/sharkdp/fd) for fast file discovery (optional; pure-Python fallback available)

All of these are provided automatically by the Nix dev shell.

## Quick Start

### With Nix (recommended)

```bash
# Enter the dev shell -- brings Python, Qdrant, Ollama, fd, and dev tools
nix develop

# Start Qdrant and Ollama in separate terminals
qdrant
ollama serve

# Pull the embedding model and verify connectivity
alex setup

# Index a codebase
alex index --context myproject /path/to/code

# Start the MCP server
alex serve
```

### Without Nix

```bash
pip install -e .

# Ensure Qdrant and Ollama are running, then:
alex setup
alex index --context myproject /path/to/code
alex serve
```

## CLI Reference

Alexandria installs two equivalent entry points: `alexandria` and `alex`.

| Command | Description |
|---------|-------------|
| `alex setup` | Pull the Ollama embedding model and verify Qdrant connectivity |
| `alex index -c CTX PATH` | Index a codebase directory into context `CTX` |
| `alex serve` | Start the MCP server (stdio transport) |
| `alex list` | Show all indexed contexts with chunk counts |
| `alex drop -c CTX` | Delete a context from the vector database |

### Indexing Options

```bash
alex index --context myproject \
  --chunk-lines 50 \       # Lines per sliding-window chunk (default: 50)
  --chunk-overlap 10 \     # Overlap between chunks (default: 10)
  --follow-symlinks \      # Follow symbolic links
  /path/to/code
```

Re-running `index` on the same context is incremental -- only changed files
(detected via SHA-256 hash) are re-embedded.

## MCP Integration

Alexandria exposes three tools over MCP stdio transport:

- **`search_code`** -- search within a specific context by natural language
  query, with optional language filter
- **`search_all`** -- search across all indexed contexts, results merged and
  ranked by relevance
- **`list_contexts`** -- list available contexts with stats

### Configuration Examples

**OpenCode** (`opencode.jsonc`):
```jsonc
{
  "mcp": {
    "alexandria": {
      "type": "local",
      "command": ["alex", "serve"],
      "enabled": true
    }
  }
}
```

**Claude Desktop** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "alexandria": {
      "command": "alex",
      "args": ["serve"]
    }
  }
}
```

## Supported Languages

Alexandria uses tree-sitter for AST-aware chunking in 40+ languages including
Python, JavaScript, TypeScript, Go, Rust, C/C++, Java, Ruby, Lua, Nix, Haskell,
Elixir, Scala, Kotlin, C#, Swift, Zig, and more. Files in unsupported languages
are chunked with a sliding-window fallback.

## Configuration

All configuration is via environment variables with sensible defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP API endpoint |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `ALEXANDRIA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model name |

## NixOS Module

The flake includes a NixOS module for production deployment with systemd
services for Qdrant, Ollama, model setup, and periodic re-indexing:

```nix
{
  services.alexandria = {
    enable = true;
    reindexSchedule = "daily";  # systemd calendar expression
    indexes = {
      myproject = { path = /home/user/src/myproject; };
      another  = { path = /home/user/src/another; };
    };
  };
}
```

This configures Qdrant, Ollama, pulls the embedding model on boot, runs
initial indexing, and sets up timers for periodic re-indexing.

## Docker

```bash
nix build .#docker
docker load < result
docker run --rm alexandria --help
```

## Development

```bash
nix develop              # Enter dev shell
pip install -e .         # Editable install

ruff check .             # Lint
ruff format .            # Format
mypy src/                # Type-check (strict)
pytest                   # Run tests
```

See [AGENTS.md](AGENTS.md) for detailed code style guidelines and conventions.

## License

MIT
