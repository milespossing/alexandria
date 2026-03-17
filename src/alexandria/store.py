"""Store — Qdrant vector database wrapper.

Each context gets its own collection named `alexandria_{context}`.
Metadata stored per point: file, start_line, end_line, symbol, language, file_hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from alexandria.chunker import Chunk
from alexandria.config import Config


@dataclass
class SearchResult:
    """A single search result with score and expanded context."""

    text: str
    file: str
    start_line: int
    end_line: int
    symbol: str | None
    language: str
    score: float
    context_before: str  # lines before the chunk
    context_after: str  # lines after the chunk


class Store:
    """Qdrant-backed vector store with per-context collections."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = QdrantClient(url=config.qdrant_url)

    def _ensure_collection(self, context: str) -> None:
        """Create the collection if it doesn't exist."""
        name = self.config.collection_name(context)
        collections = self.client.get_collections().collections
        existing = {c.name for c in collections}
        if name not in existing:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.config.embed_dim,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_chunks(
        self,
        context: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> int:
        """Upsert chunks with their embeddings into the collection.

        Returns the number of points upserted.
        """
        self._ensure_collection(context)
        name = self.config.collection_name(context)

        points = []
        for chunk, vector in zip(chunks, vectors):
            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        "file": chunk.file,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "symbol": chunk.symbol,
                        "language": chunk.language,
                        "file_hash": chunk.file_hash,
                    },
                )
            )

        # Qdrant supports batch upsert; send in chunks of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=name, points=batch)

        return len(points)

    def get_indexed_file_hashes(self, context: str) -> dict[str, str]:
        """Return a mapping of file -> file_hash for all indexed points.

        Used for change detection: if a file's hash hasn't changed, skip re-indexing.
        """
        name = self.config.collection_name(context)
        collections = self.client.get_collections().collections
        if name not in {c.name for c in collections}:
            return {}

        file_hashes: dict[str, str] = {}
        offset = None
        while True:
            results = self.client.scroll(
                collection_name=name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = results
            for point in points:
                payload = point.payload or {}
                f = payload.get("file", "")
                h = payload.get("file_hash", "")
                if f and h:
                    file_hashes[f] = h
            if next_offset is None:
                break
            offset = next_offset

        return file_hashes

    def delete_file_points(self, context: str, file_path: str) -> None:
        """Delete all points for a specific file (before re-indexing it)."""
        name = self.config.collection_name(context)
        self.client.delete(
            collection_name=name,
            points_selector=Filter(
                must=[FieldCondition(key="file", match=MatchValue(value=file_path))]
            ),
        )

    def search(
        self,
        context: str,
        query_vector: list[float],
        limit: int | None = None,
        file_filter: str | None = None,
        language_filter: str | None = None,
    ) -> list[SearchResult]:
        """Search a context for similar code chunks.

        Args:
            context: The context name to search in.
            query_vector: The embedding of the search query.
            limit: Max results (defaults to config.search_limit).
            file_filter: Optional glob/suffix filter on file paths.
            language_filter: Optional language filter.
        """
        name = self.config.collection_name(context)
        n = limit or self.config.search_limit

        # Build optional filters
        must_conditions = []
        if language_filter:
            must_conditions.append(
                FieldCondition(key="language", match=MatchValue(value=language_filter))
            )
        if file_filter:
            must_conditions.append(
                FieldCondition(key="file", match=MatchValue(value=file_filter))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        results = self.client.query_points(
            collection_name=name,
            query=query_vector,
            limit=n,
            query_filter=query_filter,
            with_payload=True,
        )

        search_results = []
        for point in results.points:
            payload = point.payload or {}
            text = payload.get("text", "")
            file_path = payload.get("file", "")
            start_line = payload.get("start_line", 0)
            end_line = payload.get("end_line", 0)

            # Expand context: read surrounding lines from disk if possible
            context_before, context_after = self._get_surrounding_context(
                file_path, start_line, end_line
            )

            search_results.append(
                SearchResult(
                    text=text,
                    file=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    symbol=payload.get("symbol"),
                    language=payload.get("language", "text"),
                    score=point.score,
                    context_before=context_before,
                    context_after=context_after,
                )
            )

        return search_results

    def search_all(
        self,
        query_vector: list[float],
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search across all contexts, merging and re-ranking results."""
        all_results: list[SearchResult] = []
        for context in self.list_contexts():
            ctx_results = self.search(context, query_vector, limit=limit)
            all_results.extend(ctx_results)

        # Sort by score descending, take top N
        all_results.sort(key=lambda r: r.score, reverse=True)
        n = limit or self.config.search_limit
        return all_results[:n]

    def list_contexts(self) -> list[str]:
        """List all context names (strips the collection prefix)."""
        collections = self.client.get_collections().collections
        prefix = self.config.collection_prefix
        return [
            c.name[len(prefix) :]
            for c in collections
            if c.name.startswith(prefix)
        ]

    def get_context_stats(self, context: str) -> dict[str, int | str]:
        """Get stats for a context: point count, file count, etc."""
        name = self.config.collection_name(context)
        try:
            info = self.client.get_collection(name)
            return {
                "context": context,
                "points": info.points_count or 0,
                "status": str(info.status),
            }
        except Exception:
            return {"context": context, "points": 0, "status": "not found"}

    def drop_context(self, context: str) -> bool:
        """Delete an entire context's collection. Returns True if it existed."""
        name = self.config.collection_name(context)
        try:
            self.client.delete_collection(name)
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def _get_surrounding_context(
        self, file_path: str, start_line: int, end_line: int
    ) -> tuple[str, str]:
        """Read surrounding lines from disk for expanded context.

        Returns (context_before, context_after) as strings.
        Falls back to empty strings if the file can't be read.
        """
        n = self.config.context_lines
        try:
            path = Path(file_path)
            if not path.is_absolute():
                # Try relative to cwd — won't always work for MCP but best effort
                path = Path.cwd() / path
            if not path.exists():
                return ("", "")

            lines = path.read_text(errors="replace").splitlines()
            before_start = max(0, start_line - 1 - n)
            before_end = max(0, start_line - 1)
            after_start = min(len(lines), end_line)
            after_end = min(len(lines), end_line + n)

            context_before = "\n".join(lines[before_start:before_end])
            context_after = "\n".join(lines[after_start:after_end])
            return (context_before, context_after)
        except (OSError, PermissionError):
            return ("", "")
