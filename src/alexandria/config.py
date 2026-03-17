"""Configuration constants and defaults for Alexandria."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Runtime configuration, populated from env vars and CLI flags."""

    # Qdrant
    qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")

    # Ollama
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    embed_model: str = os.environ.get("ALEXANDRIA_EMBED_MODEL", "nomic-embed-text")

    # Embedding dimensions for nomic-embed-text
    embed_dim: int = 768

    # Chunking
    chunk_lines: int = 50
    chunk_overlap: int = 10
    max_chunk_chars: int = 6000  # ~2500 tokens at code tokenization rates, safe for nomic-embed-text's 8192 token window
    context_lines: int = 5  # lines of context around search results

    # Search
    search_limit: int = 10

    # File discovery
    follow_symlinks: bool = False

    # Collection naming
    collection_prefix: str = "alexandria_"

    def collection_name(self, context: str) -> str:
        """Return the Qdrant collection name for a context."""
        return f"{self.collection_prefix}{context}"


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
    "javascript": {"function_declaration", "class_declaration", "export_statement",
                    "lexical_declaration", "expression_statement"},
    "typescript": {"function_declaration", "class_declaration", "export_statement",
                   "lexical_declaration", "expression_statement", "interface_declaration",
                   "type_alias_declaration", "enum_declaration"},
    "tsx": {"function_declaration", "class_declaration", "export_statement",
            "lexical_declaration", "expression_statement", "interface_declaration",
            "type_alias_declaration", "enum_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item",
             "trait_item", "mod_item", "type_item"},
    "c": {"function_definition", "struct_specifier", "enum_specifier",
          "type_definition", "declaration"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier",
            "enum_specifier", "type_definition", "declaration",
            "namespace_definition", "template_declaration"},
    "java": {"class_declaration", "method_declaration", "interface_declaration",
             "enum_declaration"},
    "ruby": {"method", "class", "module", "singleton_method"},
    "lua": {"function_declaration", "function_definition", "local_function"},
    "nix": {"binding", "function_expression"},
    "bash": {"function_definition"},
    "elixir": {"call"},  # defmodule, def, defp are all `call` nodes
    "haskell": {"function", "type_alias", "newtype", "data"},
    "scala": {"function_definition", "class_definition", "object_definition",
              "trait_definition"},
    "kotlin": {"function_declaration", "class_declaration", "object_declaration"},
    "csharp": {"method_declaration", "class_declaration", "interface_declaration",
               "struct_declaration", "enum_declaration", "namespace_declaration"},
    "swift": {"function_declaration", "class_declaration", "struct_declaration",
              "enum_declaration", "protocol_declaration"},
    "zig": {"function_declaration", "container_declaration"},
}

# Default fallback: extract any named top-level children
DEFAULT_CHUNK_NODE_TYPES: set[str] = {
    "function_definition", "function_declaration",
    "class_definition", "class_declaration",
    "method_declaration", "method_definition",
}

# Symbol name extraction: which child field holds the name
NAME_FIELDS: list[str] = ["name", "declarator"]
