"""Chunker — extracts semantically meaningful code chunks from source files.

Uses tree-sitter for AST-aware chunking (function/class boundaries) and
falls back to a sliding-window line-based approach for unsupported files.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

from tree_sitter_language_pack import get_parser

from alexandria.config import (
    CHUNK_NODE_TYPES,
    DEFAULT_CHUNK_NODE_TYPES,
    Config,
    EXTENSION_MAP,
    FILENAME_MAP,
    NAME_FIELDS,
)


@dataclass
class Chunk:
    """A single chunk of source code with metadata."""

    text: str
    file: str  # relative path from repo root
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed, inclusive
    symbol: str | None  # function/class name, if applicable
    language: str  # detected language or "text"
    file_hash: str  # sha256 of the whole file (for change detection)

    @property
    def id(self) -> str:
        """Deterministic UUID for this chunk (for Qdrant compatibility)."""
        raw = f"{self.file}:{self.start_line}:{self.end_line}:{self.file_hash}"
        hex_digest = hashlib.sha256(raw.encode()).hexdigest()
        # Use first 32 hex chars to construct a valid UUID
        return str(uuid.UUID(hex_digest[:32]))


def detect_language(filepath: Path) -> str | None:
    """Detect the tree-sitter language for a file, or None if unsupported."""
    # Check filename first (e.g. Makefile, Dockerfile)
    name = filepath.name
    if name in FILENAME_MAP:
        return FILENAME_MAP[name]

    # Check extension
    ext = filepath.suffix.lower()
    return EXTENSION_MAP.get(ext)


def _get_symbol_name(node: object) -> str | None:
    """Extract the symbol name from a tree-sitter node.

    Handles decorated definitions by recursing into the inner definition.
    """
    # For decorated definitions, recurse into the actual definition
    node_type = getattr(node, "type", "")
    if node_type == "decorated_definition":
        named_children = getattr(node, "named_children", [])
        for child in named_children:
            child_type = getattr(child, "type", "")
            if child_type in (
                "function_definition", "class_definition",
                "function_declaration", "class_declaration",
            ):
                return _get_symbol_name(child)

    # Try standard name fields
    for field_name in NAME_FIELDS:
        child = getattr(node, "child_by_field_name", lambda _: None)(field_name)
        if child is not None:
            text = child.text
            if isinstance(text, bytes):
                return text.decode("utf-8", errors="replace")
            return str(text) if text else None

    # Try first identifier child as a fallback
    named_children = getattr(node, "named_children", [])
    for child in named_children:
        child_type = getattr(child, "type", "")
        if child_type in ("identifier", "name"):
            text = child.text
            if isinstance(text, bytes):
                return text.decode("utf-8", errors="replace")
            return str(text) if text else None
    return None


def _extract_preceding_comments(
    source_lines: list[str],
    start_line_0idx: int,
) -> int:
    """Walk backwards from start_line to include preceding comment/docstring lines.

    Returns the new start line (0-indexed).
    """
    idx = start_line_0idx - 1
    while idx >= 0:
        stripped = source_lines[idx].strip()
        if stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("*"):
            idx -= 1
        elif stripped.startswith("/*") or stripped.endswith("*/"):
            idx -= 1
        elif stripped.startswith('"""') or stripped.startswith("'''"):
            idx -= 1
        elif stripped == "":
            # Allow one blank line between comment block and definition
            if idx > 0 and source_lines[idx - 1].strip().startswith(("#", "//", "*")):
                idx -= 1
            else:
                break
        else:
            break
    return idx + 1


def _split_oversized_chunk(chunk: Chunk, max_chars: int, window: int = 50, overlap: int = 10) -> list[Chunk]:
    """Split a chunk that exceeds max_chars into smaller sub-chunks.

    Uses the sliding-window approach on the chunk's lines, preserving
    symbol name and other metadata.  Returns a list of one or more chunks.
    If the chunk is within limits, returns it unchanged as a single-element list.
    """
    if len(chunk.text) <= max_chars:
        return [chunk]

    lines = chunk.text.splitlines()

    # If the window itself would produce chunks that are still too large,
    # shrink the window until it fits (but never below 5 lines).
    effective_window = window
    while effective_window > 5:
        sample = "\n".join(lines[:effective_window])
        if len(sample) <= max_chars:
            break
        effective_window = effective_window // 2

    step = max(1, effective_window - overlap)
    sub_chunks: list[Chunk] = []
    i = 0
    part = 0

    while i < len(lines):
        end = min(i + effective_window, len(lines))
        text = "\n".join(lines[i:end])

        if text.strip():
            # Compute sub-chunk line numbers relative to the file
            sub_start_line = chunk.start_line + i
            sub_end_line = chunk.start_line + end - 1

            # Annotate symbol with part number so we know it's a fragment
            symbol = f"{chunk.symbol}[part {part}]" if chunk.symbol else None

            sub_chunks.append(
                Chunk(
                    text=text,
                    file=chunk.file,
                    start_line=sub_start_line,
                    end_line=sub_end_line,
                    symbol=symbol,
                    language=chunk.language,
                    file_hash=chunk.file_hash,
                )
            )
            part += 1

        if end >= len(lines):
            break
        i += step

    return sub_chunks if sub_chunks else [chunk]


def chunk_file_treesitter(
    filepath: Path,
    source: bytes,
    language: str,
    config: Config,
    repo_root: Path,
) -> list[Chunk]:
    """Chunk a file using tree-sitter AST parsing."""
    try:
        parser = get_parser(language)  # type: ignore[arg-type]
    except (ValueError, KeyError):
        return []

    tree = parser.parse(source)
    root = tree.root_node

    source_text = source.decode("utf-8", errors="replace")
    source_lines = source_text.splitlines()
    file_hash = hashlib.sha256(source).hexdigest()
    rel_path = str(filepath.relative_to(repo_root))

    node_types = CHUNK_NODE_TYPES.get(language, DEFAULT_CHUNK_NODE_TYPES)
    chunks: list[Chunk] = []

    for child in root.children:
        if child.type not in node_types:
            continue

        start_line_0 = child.start_point[0]
        end_line_0 = child.end_point[0]

        # Include preceding comments/docstrings
        start_line_0 = _extract_preceding_comments(source_lines, start_line_0)

        # Extract chunk text
        chunk_lines = source_lines[start_line_0 : end_line_0 + 1]
        text = "\n".join(chunk_lines)

        if not text.strip():
            continue

        symbol = _get_symbol_name(child)

        chunks.append(
            Chunk(
                text=text,
                file=rel_path,
                start_line=start_line_0 + 1,
                end_line=end_line_0 + 1,
                symbol=symbol,
                language=language,
                file_hash=file_hash,
            )
        )

    # If tree-sitter found nothing (e.g. file is all top-level code), fall back
    if not chunks:
        return chunk_file_sliding_window(filepath, source, language, config, repo_root)

    # Split any chunks that exceed the embedding model's context window
    if config.max_chunk_chars > 0:
        final: list[Chunk] = []
        for c in chunks:
            final.extend(
                _split_oversized_chunk(
                    c, config.max_chunk_chars, config.chunk_lines, config.chunk_overlap,
                )
            )
        return final

    return chunks


def chunk_file_sliding_window(
    filepath: Path,
    source: bytes,
    language: str,
    config: Config,
    repo_root: Path,
) -> list[Chunk]:
    """Chunk a file using a sliding window of lines."""
    source_text = source.decode("utf-8", errors="replace")
    lines = source_text.splitlines()
    file_hash = hashlib.sha256(source).hexdigest()
    rel_path = str(filepath.relative_to(repo_root))

    if not lines:
        return []

    chunks: list[Chunk] = []
    window = config.chunk_lines
    overlap = config.chunk_overlap
    step = max(1, window - overlap)

    i = 0
    while i < len(lines):
        end = min(i + window, len(lines))
        chunk_lines = lines[i:end]
        text = "\n".join(chunk_lines)

        if text.strip():
            chunks.append(
                Chunk(
                    text=text,
                    file=rel_path,
                    start_line=i + 1,
                    end_line=end,
                    symbol=None,
                    language=language,
                    file_hash=file_hash,
                )
            )

        if end >= len(lines):
            break
        i += step

    # Split any sliding-window chunks that still exceed the limit
    if config.max_chunk_chars > 0:
        final: list[Chunk] = []
        for c in chunks:
            final.extend(
                _split_oversized_chunk(
                    c, config.max_chunk_chars, config.chunk_lines, config.chunk_overlap,
                )
            )
        return final

    return chunks


def chunk_file(
    filepath: Path,
    config: Config,
    repo_root: Path,
) -> list[Chunk]:
    """Chunk a single file. Dispatches to tree-sitter or sliding-window."""
    try:
        source = filepath.read_bytes()
    except (OSError, PermissionError):
        return []

    # Skip empty files and binary files (heuristic: null bytes)
    if not source or b"\x00" in source[:8192]:
        return []

    language = detect_language(filepath)

    if language is not None:
        chunks = chunk_file_treesitter(filepath, source, language, config, repo_root)
    else:
        # Fallback: sliding window with "text" as language
        chunks = chunk_file_sliding_window(filepath, source, "text", config, repo_root)

    return chunks
