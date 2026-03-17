"""Embedder — generates vector embeddings from text chunks.

Supports two backends:
  - **ollama** (default): Local Ollama server with models like nomic-embed-text.
  - **openai**: Any OpenAI-compatible API (GitHub Models, Azure OpenAI, TEI, vLLM, etc.)

Use ``create_embedder(config)`` to get the right implementation based on
``config.embed_backend``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

import ollama as ollama_client

from alexandria.config import Config

log = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract interface shared by all embedding backends."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = config.embed_model

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a vector."""

    @abstractmethod
    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        labels: list[str] | None = None,
    ) -> list[list[float]]:
        """Embed a batch of texts. Returns list of vectors."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the embedding service is reachable and ready."""

    def _zero_vector(self) -> list[float]:
        """Return a zero vector matching the configured dimension."""
        dim = self.config.resolve_embed_dim()
        if dim <= 0:
            raise RuntimeError(
                "Cannot create zero-vector fallback: embed_dim is unknown. "
                "Set ALEXANDRIA_EMBED_DIM or use a known model."
            )
        return [0.0] * dim


class Embedder(BaseEmbedder):
    """Generates embeddings using Ollama's embedding API."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.client = ollama_client.Client(host=config.ollama_host)

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a vector."""
        response = self.client.embed(model=self.model, input=text, truncate=True)
        # ollama returns {"embeddings": [[...]]} for embed()
        embeddings = response.get("embeddings", [])
        if embeddings:
            return embeddings[0]
        raise RuntimeError(f"Ollama returned no embeddings for model {self.model}")

    def _embed_batch_request(self, batch: list[str]) -> list[list[float]]:
        """Send a single batch embed request to Ollama.

        Args:
            batch: List of texts to embed in one request.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            ollama_client.ResponseError: If Ollama rejects the request.
            RuntimeError: If the response has the wrong number of embeddings.
        """
        response = self.client.embed(model=self.model, input=batch, truncate=True)
        embeddings: list[list[float]] = response.get("embeddings", [])
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Ollama returned {len(embeddings)} embeddings for a batch of {len(batch)} texts"
            )
        return embeddings

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        labels: list[str] | None = None,
    ) -> list[list[float]]:
        """Embed a batch of texts. Returns list of vectors.

        Sends texts to Ollama in sub-batches using the native batch API
        (``input=list[str]``).  This is dramatically faster than
        one-at-a-time because each HTTP round-trip embeds up to
        *batch_size* texts in a single model invocation.

        If a batch request fails (e.g. due to context-length limits on
        older Ollama versions that ignore ``truncate``), the batch is
        retried one text at a time so that a single oversized chunk
        cannot crash the entire indexing run.

        Args:
            texts: The texts to embed.
            batch_size: Max texts per Ollama request.  Defaults to 64.
            labels: Optional human-readable labels (e.g. file paths) for
                each text, used in warning messages when embedding fails.
                Must be the same length as *texts* if provided.
        """
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                all_embeddings.extend(self._embed_batch_request(batch))
            except (ollama_client.ResponseError, RuntimeError) as exc:
                # Fallback: embed texts one at a time so one bad chunk
                # doesn't poison the whole batch.
                log.warning(
                    "Batch embed failed (offset %d, size %d): %s  — retrying individually",
                    i,
                    len(batch),
                    exc,
                )
                for j, text in enumerate(batch):
                    idx = i + j
                    label = labels[idx] if labels else f"offset {idx}"
                    try:
                        vec = self.embed(text)
                        all_embeddings.append(vec)
                    except (ollama_client.ResponseError, RuntimeError) as inner:
                        log.warning(
                            "Failed to embed chunk (%s, %d chars): %s",
                            label,
                            len(text),
                            inner,
                        )
                        # Return a zero vector so index positions stay aligned
                        all_embeddings.append(self._zero_vector())
        return all_embeddings

    def is_available(self) -> bool:
        """Check if Ollama is reachable and the model is pulled."""
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models]
            # Model names may include tags like ":latest"
            return any(
                self.model in name or name.startswith(f"{self.model}:") for name in model_names
            )
        except Exception:
            return False

    def pull_model(self) -> None:
        """Pull the embedding model if not already available."""
        self.client.pull(self.model)


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------

# Default rate-limit retry settings.
_MAX_RETRIES = 5
_INITIAL_BACKOFF_S = 1.0


class OpenAIEmbedError(Exception):
    """Raised when an OpenAI-compatible embedding request fails."""


class OpenAIEmbedder(BaseEmbedder):
    """Generates embeddings via any OpenAI-compatible embeddings API.

    Works with GitHub Models, Azure OpenAI, text-embeddings-inference (TEI),
    vLLM, LiteLLM, and any other service that implements the OpenAI embedding
    endpoint.

    GitHub Models uses ``POST /inference/embeddings`` (no ``/v1/`` prefix) and
    requires extra headers (``Accept``, ``X-GitHub-Api-Version``).  These are
    auto-detected when ``embed_api_url`` contains ``models.github.ai``.

    Args:
        config: Alexandria Config.  Uses ``embed_api_url``, ``embed_api_key``,
            and ``embed_model`` to construct requests.
    """

    # GitHub Models API version header value.
    _GITHUB_API_VERSION = "2026-03-10"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.api_url = config.embed_api_url.rstrip("/")
        self.api_key = config.embed_api_key

        # Detect GitHub Models and adjust endpoint path accordingly.
        self._is_github_models = "models.github.ai" in self.api_url
        if self._is_github_models:
            self.endpoint = f"{self.api_url}/embeddings"
        else:
            self.endpoint = f"{self.api_url}/v1/embeddings"

    # -- internal helpers ---------------------------------------------------

    def _request(self, texts: list[str]) -> list[list[float]]:
        """Send a single ``POST /v1/embeddings`` request with retry on 429.

        Args:
            texts: Input texts to embed in one request.

        Returns:
            Embedding vectors in the same order as *texts*.

        Raises:
            OpenAIEmbedError: On non-retryable HTTP errors or exhausted retries.
        """
        body = json.dumps({"model": self.model, "input": texts}).encode()
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # GitHub Models requires these extra headers.
        if self._is_github_models:
            headers["Accept"] = "application/vnd.github+json"
            headers["X-GitHub-Api-Version"] = self._GITHUB_API_VERSION

        backoff = _INITIAL_BACKOFF_S
        for attempt in range(_MAX_RETRIES):
            req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data: dict[str, object] = json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    # Rate limited — honour Retry-After if present.
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    wait = float(retry_after) if retry_after else backoff
                    log.warning(
                        "Rate limited (429), attempt %d/%d — waiting %.1fs",
                        attempt + 1,
                        _MAX_RETRIES,
                        wait,
                    )
                    time.sleep(wait)
                    backoff = min(backoff * 2, 60.0)
                    continue

                # Read the error body for a useful message.
                err_body = ""
                with contextlib.suppress(Exception):
                    err_body = exc.read().decode(errors="replace")[:500]
                raise OpenAIEmbedError(f"HTTP {exc.code} from {self.endpoint}: {err_body}") from exc
            except urllib.error.URLError as exc:
                raise OpenAIEmbedError(f"Cannot reach {self.endpoint}: {exc.reason}") from exc

            # Parse response — {"data": [{"embedding": [...], "index": N}, ...]}
            data_items = data.get("data")
            if not isinstance(data_items, list) or len(data_items) != len(texts):
                raise OpenAIEmbedError(
                    f"Expected {len(texts)} embeddings, got "
                    f"{len(data_items) if isinstance(data_items, list) else 'invalid response'}"
                )

            # Sort by index to guarantee order (spec allows unordered).
            sorted_items = sorted(data_items, key=lambda d: d.get("index", 0))
            embeddings: list[list[float]] = []
            for item in sorted_items:
                if not isinstance(item, dict):
                    raise OpenAIEmbedError(f"Unexpected item in response data: {item!r}")
                vec = item.get("embedding")
                if not isinstance(vec, list):
                    raise OpenAIEmbedError(f"Missing 'embedding' in response item: {item!r}")
                embeddings.append(vec)

            # Auto-detect embed_dim from the first successful response.
            if self.config.embed_dim <= 0 and embeddings:
                detected_dim = len(embeddings[0])
                log.info("Auto-detected embedding dimension: %d", detected_dim)
                self.config.embed_dim = detected_dim

            return embeddings

        raise OpenAIEmbedError(f"Exhausted {_MAX_RETRIES} retries due to rate limiting")

    # -- public interface ---------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a vector."""
        results = self._request([text])
        return results[0]

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        labels: list[str] | None = None,
    ) -> list[list[float]]:
        """Embed a batch of texts via the OpenAI-compatible API.

        Sends texts in sub-batches of *batch_size*.  If a batch fails, retries
        one text at a time (same resilience pattern as the Ollama backend).

        Args:
            texts: The texts to embed.
            batch_size: Max texts per API request.  Defaults to 64.
            labels: Optional human-readable labels for error messages.
        """
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                all_embeddings.extend(self._request(batch))
            except OpenAIEmbedError as exc:
                log.warning(
                    "Batch embed failed (offset %d, size %d): %s  — retrying individually",
                    i,
                    len(batch),
                    exc,
                )
                for j, text in enumerate(batch):
                    idx = i + j
                    label = labels[idx] if labels else f"offset {idx}"
                    try:
                        vec = self.embed(text)
                        all_embeddings.append(vec)
                    except OpenAIEmbedError as inner:
                        log.warning(
                            "Failed to embed chunk (%s, %d chars): %s",
                            label,
                            len(text),
                            inner,
                        )
                        all_embeddings.append(self._zero_vector())
        return all_embeddings

    def is_available(self) -> bool:
        """Check if the API endpoint is reachable by sending a tiny test embed."""
        try:
            self._request(["test"])
            return True
        except (OpenAIEmbedError, Exception):
            return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Default max_chunk_chars per backend.  Applied when config.max_chunk_chars == 0.
_OLLAMA_MAX_CHUNK_CHARS = 3000  # ~2000 tokens — safe for nomic-embed-text (2048 tok ctx)
_OPENAI_MAX_CHUNK_CHARS = 6000  # ~4000 tokens — text-embedding-3-* supports 8191 tok ctx


def create_embedder(config: Config) -> BaseEmbedder:
    """Create the appropriate embedder for the configured backend.

    Args:
        config: Alexandria Config with ``embed_backend`` set to
            ``"ollama"`` or ``"openai"``.

    Returns:
        An embedder instance ready for use.

    Raises:
        ValueError: If ``embed_backend`` is not recognized.
    """
    backend = config.embed_backend.lower()
    if backend == "ollama":
        # Default Ollama embed_dim to 768 (nomic-embed-text) if not set.
        if config.embed_dim <= 0:
            resolved = config.resolve_embed_dim()
            if resolved > 0:
                config.embed_dim = resolved
            else:
                config.embed_dim = 768  # safe default for Ollama
        if config.max_chunk_chars <= 0:
            config.max_chunk_chars = _OLLAMA_MAX_CHUNK_CHARS
        return Embedder(config)

    if backend == "openai":
        # Default model for GitHub Models if still set to the Ollama default.
        if config.embed_model == "nomic-embed-text":
            is_github = "models.github.ai" in config.embed_api_url
            config.embed_model = (
                "openai/text-embedding-3-small" if is_github else "text-embedding-3-small"
            )
            log.info(
                "OpenAI backend: defaulting embed_model to '%s'",
                config.embed_model,
            )
        # Auto-detect defaults for known OpenAI models.
        if config.embed_dim <= 0:
            resolved = config.resolve_embed_dim()
            if resolved > 0:
                config.embed_dim = resolved
            # else: will be auto-detected on first embed call
        if config.max_chunk_chars <= 0:
            config.max_chunk_chars = _OPENAI_MAX_CHUNK_CHARS
        return OpenAIEmbedder(config)

    raise ValueError(
        f"Unknown embed_backend '{config.embed_backend}'. Supported values: 'ollama', 'openai'."
    )
