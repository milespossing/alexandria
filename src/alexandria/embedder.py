"""Embedder — generates vector embeddings from text chunks via Ollama."""

from __future__ import annotations

import logging

import ollama as ollama_client

from alexandria.config import Config

log = logging.getLogger(__name__)


class Embedder:
    """Generates embeddings using Ollama's embedding API."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = ollama_client.Client(host=config.ollama_host)
        self.model = config.embed_model

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
                        all_embeddings.append([0.0] * self.config.embed_dim)
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
