"""Configuration constants and defaults for Alexandria."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Project-level config file name (placed at repo root).
PROJECT_CONFIG_FILE = ".alexandria.yml"


@dataclass
class Config:
    """Runtime configuration, populated from env vars and CLI flags."""

    # Qdrant
    qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")

    # Embedding backend: "ollama" (default) or "openai" (any OpenAI-compatible API)
    embed_backend: str = os.environ.get("ALEXANDRIA_EMBED_BACKEND", "ollama")

    # Ollama
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    embed_model: str = os.environ.get("ALEXANDRIA_EMBED_MODEL", "nomic-embed-text")

    # OpenAI-compatible API settings (used when embed_backend == "openai")
    embed_api_url: str = os.environ.get(
        "ALEXANDRIA_EMBED_API_URL", "https://models.github.ai/inference"
    )
    embed_api_key: str = os.environ.get(
        "ALEXANDRIA_EMBED_API_KEY",
        os.environ.get("GITHUB_TOKEN", ""),
    )

    # Embedding dimensions — 768 for nomic-embed-text, 1536 for text-embedding-3-small,
    # 3072 for text-embedding-3-large. Set to 0 to auto-detect on first embed call.
    embed_dim: int = int(os.environ.get("ALEXANDRIA_EMBED_DIM", "0"))

    # Chunking
    chunk_lines: int = 50
    chunk_overlap: int = 10
    # max_chunk_chars: 0 means "auto-select based on embed backend" (see create_embedder).
    #   - ollama  -> 3000 (~2000 tokens, safe for nomic-embed-text's 2048 token ctx)
    #   - openai  -> 6000 (~4000 tokens, text-embedding-3-* supports 8191 tokens)
    max_chunk_chars: int = int(os.environ.get("ALEXANDRIA_MAX_CHUNK_CHARS", "0"))
    context_lines: int = 5  # lines of context around search results

    # Maximum tokens per embedding API request.  0 means "no limit" (Ollama is local
    # so there is no external cap).  For GitHub Models / OpenAI the free tier allows
    # 64 000 tokens per request — set via create_embedder() when embed_backend == "openai".
    max_tokens_per_request: int = int(os.environ.get("ALEXANDRIA_MAX_TOKENS_PER_REQUEST", "0"))

    # Search
    search_limit: int = 10

    # File discovery
    follow_symlinks: bool = False
    ignore_patterns: list[str] = field(default_factory=list)

    # Collection naming
    collection_prefix: str = "alexandria_"

    def collection_name(self, context: str) -> str:
        """Return the Qdrant collection name for a context."""
        return f"{self.collection_prefix}{context}"

    def resolve_embed_dim(self) -> int:
        """Return the embedding dimension, falling back to known model defaults.

        If ``embed_dim`` is already set (> 0) it is returned as-is.
        Otherwise we look up the model name in ``KNOWN_EMBED_DIMS``.
        If the model is unknown we return 0 (meaning "detect at runtime").
        """
        if self.embed_dim > 0:
            return self.embed_dim
        return KNOWN_EMBED_DIMS.get(self.embed_model, 0)


# Well-known embedding dimensions keyed by model name.
KNOWN_EMBED_DIMS: dict[str, int] = {
    "nomic-embed-text": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "cohere-embed-v4": 1024,
    # GitHub Models uses {publisher}/{model} format.
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
    "openai/text-embedding-ada-002": 1536,
}

# Reverse lookup: dimension -> (backend, model).  Used to auto-detect the
# correct embedder from a collection's vector size when no explicit metadata
# is stored.  Only includes the *primary* (most common) model per dimension.
DIM_TO_DEFAULT_MODEL: dict[int, tuple[str, str]] = {
    768: ("ollama", "nomic-embed-text"),
    1536: ("openai", "text-embedding-3-small"),
    3072: ("openai", "text-embedding-3-large"),
    1024: ("openai", "cohere-embed-v4"),
}


@dataclass
class CollectionEmbedInfo:
    """Embedding model metadata stored on a Qdrant collection.

    This is persisted in the collection's ``metadata`` dict so that
    Alexandria can auto-select the correct embedder at query time —
    even if the server's environment variables point at a different
    backend than the one used to build the index.

    Args:
        embed_backend: ``"ollama"`` or ``"openai"``.
        embed_model: Model name (e.g. ``"nomic-embed-text"``).
        embed_dim: Dimensionality of the stored vectors.
    """

    embed_backend: str
    embed_model: str
    embed_dim: int


def load_project_config(repo_root: Path, config: Config) -> Config:
    """Load ``.alexandria.yml`` from *repo_root* and merge into *config*.

    Supported keys:

    - ``ignore`` — list of glob patterns to exclude from indexing.
      Uses the same syntax as ``fd --exclude`` / ``.gitignore``.

    If the file does not exist or is empty the config is returned unchanged.

    Args:
        repo_root: The repository root directory.
        config: The existing Config to merge into.

    Returns:
        The (possibly updated) Config.
    """
    config_path = repo_root / PROJECT_CONFIG_FILE
    if not config_path.is_file():
        return config

    import yaml  # type: ignore[import-untyped]  # lazy — only needed when the file exists

    try:
        raw = yaml.safe_load(config_path.read_text())
    except Exception as exc:
        log.warning("Failed to parse %s: %s", config_path, exc)
        return config

    if not isinstance(raw, dict):
        return config

    # ignore patterns
    ignore = raw.get("ignore")
    if isinstance(ignore, list):
        config.ignore_patterns = [str(p) for p in ignore if p]

    return config


# Extension -> tree-sitter language name mapping
EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".lua": "lua",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".hh": "cpp",
    ".rb": "ruby",
    ".java": "java",
    ".scala": "scala",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cs": "csharp",
    ".swift": "swift",
    ".zig": "zig",
    ".nim": "nim",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml_interface",
    ".clj": "clojure",
    ".cljc": "clojure",
    ".cljs": "clojure",
    ".r": "r",
    ".R": "r",
    ".jl": "julia",
    ".php": "php",
    ".pl": "perl",
    ".pm": "perl",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".fish": "fish",
    ".vim": "vim",
    ".el": "elisp",
    ".dart": "dart",
    ".v": "v",
    ".d": "d",
    ".f90": "fortran",
    ".f95": "fortran",
    ".f03": "fortran",
    ".sql": "sql",
    ".proto": "proto",
    ".nix": "nix",
    ".vue": "vue",
    ".svelte": "svelte",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".xml": "xml",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".tf": "terraform",
    ".hcl": "hcl",
    ".dockerfile": "dockerfile",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".make": "make",
    ".cmake": "cmake",
    ".groovy": "groovy",
    ".gradle": "groovy",
    ".glsl": "glsl",
    ".wgsl": "wgsl",
}

# File names (no extension) -> language
FILENAME_MAP: dict[str, str] = {
    "Makefile": "make",
    "Dockerfile": "dockerfile",
    "CMakeLists.txt": "cmake",
    "Jenkinsfile": "groovy",
    "BUILD": "starlark",
    "BUILD.bazel": "starlark",
    "WORKSPACE": "starlark",
}

# Node types that represent top-level code blocks we want to extract as chunks.
# Maps language name -> set of node types.
CHUNK_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
        "expression_statement",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
        "expression_statement",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
    },
    "tsx": {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
        "expression_statement",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
    },
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "mod_item",
        "type_item",
    },
    "c": {
        "function_definition",
        "struct_specifier",
        "enum_specifier",
        "type_definition",
        "declaration",
    },
    "cpp": {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "enum_specifier",
        "type_definition",
        "declaration",
        "namespace_definition",
        "template_declaration",
    },
    "java": {
        "class_declaration",
        "method_declaration",
        "interface_declaration",
        "enum_declaration",
    },
    "ruby": {"method", "class", "module", "singleton_method"},
    "lua": {"function_declaration", "function_definition", "local_function"},
    "nix": {"binding", "function_expression"},
    "bash": {"function_definition"},
    "elixir": {"call"},  # defmodule, def, defp are all `call` nodes
    "haskell": {"function", "type_alias", "newtype", "data"},
    "scala": {"function_definition", "class_definition", "object_definition", "trait_definition"},
    "kotlin": {"function_declaration", "class_declaration", "object_declaration"},
    "csharp": {
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "struct_declaration",
        "enum_declaration",
        "namespace_declaration",
    },
    "swift": {
        "function_declaration",
        "class_declaration",
        "struct_declaration",
        "enum_declaration",
        "protocol_declaration",
    },
    "zig": {"function_declaration", "container_declaration"},
}

# Default fallback: extract any named top-level children
DEFAULT_CHUNK_NODE_TYPES: set[str] = {
    "function_definition",
    "function_declaration",
    "class_definition",
    "class_declaration",
    "method_declaration",
    "method_definition",
}

# Symbol name extraction: which child field holds the name
NAME_FIELDS: list[str] = ["name", "declarator"]
