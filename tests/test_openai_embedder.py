"""Tests for the OpenAI-compatible embedding backend and create_embedder factory.

These are unit tests — they mock HTTP responses and do not require any
external services (no Qdrant, no Ollama, no OpenAI API).
"""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from alexandria.config import Config
from alexandria.embedder import (
    BaseEmbedder,
    Embedder,
    OpenAIEmbedder,
    OpenAIEmbedError,
    create_embedder,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _openai_config(**overrides: Any) -> Config:
    """Return a Config wired for the OpenAI backend."""
    defaults: dict[str, Any] = {
        "embed_backend": "openai",
        "embed_api_url": "https://models.example.com",
        "embed_api_key": "test-key-123",
        "embed_model": "text-embedding-3-small",
        "embed_dim": 4,  # tiny dim for tests
    }
    defaults.update(overrides)
    return Config(**defaults)


def _make_response(embeddings: list[list[float]]) -> bytes:
    """Build a JSON response body matching the OpenAI /v1/embeddings format."""
    data = [
        {"embedding": vec, "index": i, "object": "embedding"} for i, vec in enumerate(embeddings)
    ]
    return json.dumps(
        {
            "object": "list",
            "data": data,
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
    ).encode()


def _mock_urlopen(response_bytes: bytes) -> MagicMock:
    """Create a mock for urllib.request.urlopen that returns *response_bytes*."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_bytes
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# create_embedder factory
# ---------------------------------------------------------------------------


class TestCreateEmbedder:
    """Verify the factory dispatches to the right backend."""

    def test_ollama_backend(self) -> None:
        config = Config(embed_backend="ollama")
        embedder = create_embedder(config)
        assert isinstance(embedder, Embedder)

    def test_openai_backend(self) -> None:
        config = _openai_config()
        embedder = create_embedder(config)
        assert isinstance(embedder, OpenAIEmbedder)

    def test_unknown_backend_raises(self) -> None:
        config = Config(embed_backend="magic")
        with pytest.raises(ValueError, match="Unknown embed_backend"):
            create_embedder(config)

    def test_ollama_resolves_embed_dim(self) -> None:
        config = Config(embed_backend="ollama", embed_dim=0)
        create_embedder(config)
        assert config.embed_dim == 768  # nomic-embed-text default

    def test_ollama_defaults_max_chunk_chars(self) -> None:
        """Ollama backend defaults max_chunk_chars to 3000."""
        config = Config(embed_backend="ollama", max_chunk_chars=0)
        create_embedder(config)
        assert config.max_chunk_chars == 3000

    def test_openai_defaults_max_chunk_chars(self) -> None:
        """OpenAI backend defaults max_chunk_chars to 6000."""
        config = _openai_config(max_chunk_chars=0)
        create_embedder(config)
        assert config.max_chunk_chars == 6000

    def test_explicit_max_chunk_chars_preserved(self) -> None:
        """User-set max_chunk_chars is not overridden."""
        config = _openai_config(max_chunk_chars=4000)
        create_embedder(config)
        assert config.max_chunk_chars == 4000

    def test_openai_defaults_model(self) -> None:
        """If embed_model is still the Ollama default, switch to text-embedding-3-small."""
        config = Config(
            embed_backend="openai",
            embed_model="nomic-embed-text",
            embed_api_url="https://example.com",
            embed_api_key="key",
            embed_dim=1536,
        )
        create_embedder(config)
        assert config.embed_model == "text-embedding-3-small"

    def test_openai_defaults_model_github(self) -> None:
        """If targeting GitHub Models, default model gets the vendor prefix."""
        config = Config(
            embed_backend="openai",
            embed_model="nomic-embed-text",
            embed_api_url="https://models.github.ai/inference",
            embed_api_key="key",
        )
        create_embedder(config)
        assert config.embed_model == "openai/text-embedding-3-small"
        # Vendor-prefixed name should also resolve embed_dim
        assert config.embed_dim == 1536

    def test_openai_keeps_explicit_model(self) -> None:
        """If the user set a custom model, don't override it."""
        config = Config(
            embed_backend="openai",
            embed_model="my-custom-model",
            embed_api_url="https://example.com",
            embed_api_key="key",
            embed_dim=256,
        )
        create_embedder(config)
        assert config.embed_model == "my-custom-model"

    def test_returns_base_embedder(self) -> None:
        """Both backends satisfy the BaseEmbedder interface."""
        ollama = create_embedder(Config(embed_backend="ollama"))
        openai = create_embedder(_openai_config())
        assert isinstance(ollama, BaseEmbedder)
        assert isinstance(openai, BaseEmbedder)


# ---------------------------------------------------------------------------
# OpenAIEmbedder endpoint construction & GitHub Models detection
# ---------------------------------------------------------------------------


class TestOpenAIEndpoint:
    """Verify endpoint URL and header logic for GitHub Models vs generic APIs."""

    def test_generic_endpoint(self) -> None:
        config = _openai_config(embed_api_url="https://api.openai.com")
        embedder = OpenAIEmbedder(config)
        assert embedder.endpoint == "https://api.openai.com/v1/embeddings"
        assert embedder._is_github_models is False

    def test_github_models_endpoint(self) -> None:
        config = _openai_config(embed_api_url="https://models.github.ai/inference")
        embedder = OpenAIEmbedder(config)
        assert embedder.endpoint == "https://models.github.ai/inference/embeddings"
        assert embedder._is_github_models is True

    def test_github_models_endpoint_trailing_slash(self) -> None:
        config = _openai_config(embed_api_url="https://models.github.ai/inference/")
        embedder = OpenAIEmbedder(config)
        assert embedder.endpoint == "https://models.github.ai/inference/embeddings"

    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_github_models_headers(self, mock_urlopen_fn: MagicMock) -> None:
        """GitHub Models requests include Accept and X-GitHub-Api-Version headers."""
        vec = [0.1, 0.2, 0.3, 0.4]
        mock_urlopen_fn.return_value = _mock_urlopen(_make_response([vec]))

        config = _openai_config(embed_api_url="https://models.github.ai/inference")
        embedder = OpenAIEmbedder(config)
        embedder.embed("test")

        req = mock_urlopen_fn.call_args[0][0]
        assert req.get_header("Accept") == "application/vnd.github+json"
        assert req.get_header("X-github-api-version") == "2026-03-10"

    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_generic_no_github_headers(self, mock_urlopen_fn: MagicMock) -> None:
        """Generic OpenAI API requests do NOT include GitHub-specific headers."""
        vec = [0.1, 0.2, 0.3, 0.4]
        mock_urlopen_fn.return_value = _mock_urlopen(_make_response([vec]))

        config = _openai_config(embed_api_url="https://api.openai.com")
        embedder = OpenAIEmbedder(config)
        embedder.embed("test")

        req = mock_urlopen_fn.call_args[0][0]
        assert req.get_header("Accept") is None
        assert req.get_header("X-github-api-version") is None


# ---------------------------------------------------------------------------
# OpenAIEmbedder.embed
# ---------------------------------------------------------------------------


class TestOpenAIEmbed:
    """Test single-text embedding via the OpenAI backend."""

    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_embed_single_text(self, mock_urlopen_fn: MagicMock) -> None:
        vec = [0.1, 0.2, 0.3, 0.4]
        mock_urlopen_fn.return_value = _mock_urlopen(_make_response([vec]))

        embedder = OpenAIEmbedder(_openai_config())
        result = embedder.embed("hello world")

        assert result == vec
        mock_urlopen_fn.assert_called_once()

        # Verify the request body
        call_args = mock_urlopen_fn.call_args
        req = call_args[0][0]
        body = json.loads(req.data)
        assert body["model"] == "text-embedding-3-small"
        assert body["input"] == ["hello world"]
        assert req.get_header("Authorization") == "Bearer test-key-123"
        assert req.get_header("Content-type") == "application/json"

    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_embed_no_api_key(self, mock_urlopen_fn: MagicMock) -> None:
        """When api_key is empty, no Authorization header is sent."""
        vec = [0.1, 0.2, 0.3, 0.4]
        mock_urlopen_fn.return_value = _mock_urlopen(_make_response([vec]))

        config = _openai_config(embed_api_key="")
        embedder = OpenAIEmbedder(config)
        embedder.embed("test")

        req = mock_urlopen_fn.call_args[0][0]
        assert req.get_header("Authorization") is None

    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_embed_http_error(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.side_effect = urllib.error.HTTPError(
            url="https://example.com",
            code=500,
            msg="Internal Server Error",
            hdrs=None,  # type: ignore[arg-type]
            fp=BytesIO(b"server broke"),
        )

        embedder = OpenAIEmbedder(_openai_config())
        with pytest.raises(OpenAIEmbedError, match="HTTP 500"):
            embedder.embed("fail")

    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_embed_url_error(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.side_effect = urllib.error.URLError("connection refused")

        embedder = OpenAIEmbedder(_openai_config())
        with pytest.raises(OpenAIEmbedError, match="Cannot reach"):
            embedder.embed("fail")

    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_embed_wrong_count(self, mock_urlopen_fn: MagicMock) -> None:
        """Server returns wrong number of embeddings."""
        # Send 1 text, get 2 embeddings back
        mock_urlopen_fn.return_value = _mock_urlopen(
            _make_response([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        )
        embedder = OpenAIEmbedder(_openai_config())
        with pytest.raises(OpenAIEmbedError, match="Expected 1 embeddings"):
            embedder.embed("test")


# ---------------------------------------------------------------------------
# OpenAIEmbedder.embed_batch
# ---------------------------------------------------------------------------


class TestOpenAIEmbedBatch:
    """Test batch embedding with fallback behaviour."""

    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_batch_success(self, mock_urlopen_fn: MagicMock) -> None:
        vecs = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        mock_urlopen_fn.return_value = _mock_urlopen(_make_response(vecs))

        embedder = OpenAIEmbedder(_openai_config())
        result = embedder.embed_batch(["hello", "world"], batch_size=64)

        assert len(result) == 2
        assert result[0] == vecs[0]
        assert result[1] == vecs[1]

    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_batch_splits_into_sub_batches(self, mock_urlopen_fn: MagicMock) -> None:
        """With batch_size=2 and 3 texts, should make 2 API calls."""
        call_count = 0

        def side_effect(req: Any, timeout: int = 120) -> MagicMock:
            nonlocal call_count
            body = json.loads(req.data)
            n = len(body["input"])
            vecs = [[float(call_count + i)] * 4 for i in range(n)]
            call_count += 1
            return _mock_urlopen(_make_response(vecs))

        mock_urlopen_fn.side_effect = side_effect

        embedder = OpenAIEmbedder(_openai_config())
        result = embedder.embed_batch(["a", "b", "c"], batch_size=2)

        assert len(result) == 3
        assert mock_urlopen_fn.call_count == 2  # 2 texts + 1 text

    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_batch_fallback_on_error(self, mock_urlopen_fn: MagicMock) -> None:
        """If the batch request fails, retry one-at-a-time."""
        call_count = 0

        def side_effect(req: Any, timeout: int = 120) -> MagicMock:
            nonlocal call_count
            call_count += 1
            body = json.loads(req.data)
            if len(body["input"]) > 1:
                # Fail the batch request
                raise urllib.error.HTTPError(
                    url="https://example.com",
                    code=500,
                    msg="fail",
                    hdrs=None,  # type: ignore[arg-type]
                    fp=BytesIO(b"batch error"),
                )
            # Succeed for individual texts
            return _mock_urlopen(_make_response([[0.1, 0.2, 0.3, 0.4]]))

        mock_urlopen_fn.side_effect = side_effect

        embedder = OpenAIEmbedder(_openai_config())
        result = embedder.embed_batch(["a", "b"], batch_size=64, labels=["file:1", "file:2"])

        # 1 batch attempt (failed) + 2 individual retries = 3 calls
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3, 0.4]

    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_batch_zero_vector_on_total_failure(self, mock_urlopen_fn: MagicMock) -> None:
        """If even individual embedding fails, insert a zero vector."""
        mock_urlopen_fn.side_effect = urllib.error.HTTPError(
            url="https://example.com",
            code=500,
            msg="fail",
            hdrs=None,  # type: ignore[arg-type]
            fp=BytesIO(b"always fails"),
        )

        config = _openai_config(embed_dim=4)
        embedder = OpenAIEmbedder(config)
        result = embedder.embed_batch(["a"], batch_size=64)

        assert len(result) == 1
        assert result[0] == [0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Token-budget batching
# ---------------------------------------------------------------------------


class TestTokenBudgetBatching:
    """Test _split_by_token_budget and token-aware embed_batch."""

    def test_estimate_tokens(self) -> None:
        embedder = OpenAIEmbedder(_openai_config())
        # 400 chars -> ~100 tokens
        assert embedder._estimate_tokens("a" * 400) == 100
        # Empty -> minimum of 1
        assert embedder._estimate_tokens("") == 1

    def test_split_all_fit_in_one_batch(self) -> None:
        config = _openai_config(max_tokens_per_request=1000)
        embedder = OpenAIEmbedder(config)
        # 3 texts of 100 chars each -> ~25 tokens each -> 75 total, fits in 1000
        texts = ["a" * 100, "b" * 100, "c" * 100]
        batches = embedder._split_by_token_budget(texts)
        assert len(batches) == 1
        assert batches[0] == [0, 1, 2]

    def test_split_into_multiple_batches(self) -> None:
        config = _openai_config(max_tokens_per_request=100)
        embedder = OpenAIEmbedder(config)
        # 3 texts of 200 chars each -> ~50 tokens each
        # Budget=100 -> 2 fit per batch, then 1 in the last
        texts = ["a" * 200, "b" * 200, "c" * 200]
        batches = embedder._split_by_token_budget(texts)
        assert len(batches) == 2
        assert batches[0] == [0, 1]
        assert batches[1] == [2]

    def test_split_oversized_single_text(self) -> None:
        config = _openai_config(max_tokens_per_request=10)
        embedder = OpenAIEmbedder(config)
        # 1 text of 200 chars -> ~50 tokens, way over budget of 10
        # Should still get its own batch (API will handle the error)
        texts = ["a" * 200]
        batches = embedder._split_by_token_budget(texts)
        assert len(batches) == 1
        assert batches[0] == [0]

    def test_split_no_budget(self) -> None:
        config = _openai_config(max_tokens_per_request=0)
        embedder = OpenAIEmbedder(config)
        texts = ["a" * 200] * 100
        batches = embedder._split_by_token_budget(texts)
        assert len(batches) == 1
        assert len(batches[0]) == 100

    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_embed_batch_uses_token_budget(self, mock_urlopen_fn: MagicMock) -> None:
        """With token budget, texts are grouped by estimated tokens, not count."""
        call_count = 0

        def side_effect(req: Any, timeout: int = 120) -> MagicMock:
            nonlocal call_count
            body = json.loads(req.data)
            n = len(body["input"])
            vecs = [[float(call_count)] * 4 for _ in range(n)]
            call_count += 1
            return _mock_urlopen(_make_response(vecs))

        mock_urlopen_fn.side_effect = side_effect

        # Budget of 100 tokens, 3 texts of ~50 tokens each -> 2 batches
        config = _openai_config(max_tokens_per_request=100)
        embedder = OpenAIEmbedder(config)
        result = embedder.embed_batch(["a" * 200, "b" * 200, "c" * 200])

        assert len(result) == 3
        assert mock_urlopen_fn.call_count == 2

    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_embed_batch_fallback_without_budget(self, mock_urlopen_fn: MagicMock) -> None:
        """Without token budget, falls back to fixed batch_size splitting."""
        call_count = 0

        def side_effect(req: Any, timeout: int = 120) -> MagicMock:
            nonlocal call_count
            body = json.loads(req.data)
            n = len(body["input"])
            vecs = [[float(call_count)] * 4 for _ in range(n)]
            call_count += 1
            return _mock_urlopen(_make_response(vecs))

        mock_urlopen_fn.side_effect = side_effect

        # No token budget, batch_size=2, 3 texts -> 2 calls
        config = _openai_config(max_tokens_per_request=0)
        embedder = OpenAIEmbedder(config)
        result = embedder.embed_batch(["a", "b", "c"], batch_size=2)

        assert len(result) == 3
        assert mock_urlopen_fn.call_count == 2

    def test_factory_sets_max_tokens(self) -> None:
        """create_embedder sets max_tokens_per_request for the OpenAI backend."""
        config = Config(
            embed_backend="openai",
            embed_api_url="https://example.com",
            embed_api_key="key",
            embed_model="text-embedding-3-small",
            embed_dim=1536,
            max_tokens_per_request=0,
        )
        create_embedder(config)
        assert config.max_tokens_per_request == 64_000

    def test_factory_preserves_explicit_max_tokens(self) -> None:
        """User-set max_tokens_per_request is not overridden."""
        config = Config(
            embed_backend="openai",
            embed_api_url="https://example.com",
            embed_api_key="key",
            embed_model="text-embedding-3-small",
            embed_dim=1536,
            max_tokens_per_request=32_000,
        )
        create_embedder(config)
        assert config.max_tokens_per_request == 32_000


# ---------------------------------------------------------------------------
# OpenAIEmbedder.is_available
# ---------------------------------------------------------------------------


class TestOpenAIIsAvailable:
    """Test availability check."""

    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_available(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(_make_response([[0.1, 0.2, 0.3, 0.4]]))
        embedder = OpenAIEmbedder(_openai_config())
        assert embedder.is_available() is True

    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_not_available(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.side_effect = urllib.error.URLError("refused")
        embedder = OpenAIEmbedder(_openai_config())
        assert embedder.is_available() is False


# ---------------------------------------------------------------------------
# Rate-limit retry
# ---------------------------------------------------------------------------


class TestOpenAIRateLimit:
    """Test 429 retry with exponential backoff."""

    @patch("alexandria.embedder.time.sleep")
    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_retry_on_429(self, mock_urlopen_fn: MagicMock, mock_sleep: MagicMock) -> None:
        """Should retry after 429 and succeed on the second attempt."""
        vec = [0.1, 0.2, 0.3, 0.4]

        headers = MagicMock()
        headers.get.return_value = "2"  # Retry-After: 2 seconds

        call_count = 0

        def side_effect(req: Any, timeout: int = 120) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise urllib.error.HTTPError(
                    url="https://example.com",
                    code=429,
                    msg="Too Many Requests",
                    hdrs=headers,
                    fp=BytesIO(b"rate limited"),
                )
            return _mock_urlopen(_make_response([vec]))

        mock_urlopen_fn.side_effect = side_effect

        embedder = OpenAIEmbedder(_openai_config())
        result = embedder.embed("test")

        assert result == vec
        assert call_count == 2
        mock_sleep.assert_called_once_with(2.0)  # Retry-After value

    @patch("alexandria.embedder.time.sleep")
    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_exhausted_retries(self, mock_urlopen_fn: MagicMock, mock_sleep: MagicMock) -> None:
        """After max retries on 429, should raise."""
        headers = MagicMock()
        headers.get.return_value = None  # no Retry-After

        mock_urlopen_fn.side_effect = urllib.error.HTTPError(
            url="https://example.com",
            code=429,
            msg="Too Many Requests",
            hdrs=headers,
            fp=BytesIO(b"rate limited"),
        )

        embedder = OpenAIEmbedder(_openai_config())
        with pytest.raises(OpenAIEmbedError, match="Exhausted .* retries"):
            embedder.embed("test")


# ---------------------------------------------------------------------------
# Auto-detect embed_dim
# ---------------------------------------------------------------------------


class TestAutoDetectDim:
    """Verify embed_dim is auto-detected from the first response."""

    @patch("alexandria.embedder.urllib.request.urlopen")
    def test_auto_detect(self, mock_urlopen_fn: MagicMock) -> None:
        vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        mock_urlopen_fn.return_value = _mock_urlopen(_make_response([vec]))

        config = _openai_config(embed_dim=0)
        embedder = OpenAIEmbedder(config)
        embedder.embed("detect me")

        assert config.embed_dim == 8


# ---------------------------------------------------------------------------
# Config.resolve_embed_dim
# ---------------------------------------------------------------------------


class TestResolveEmbedDim:
    """Verify the dimension resolution helper."""

    def test_explicit_dim_returned(self) -> None:
        config = Config(embed_dim=256)
        assert config.resolve_embed_dim() == 256

    def test_known_model_lookup(self) -> None:
        config = Config(embed_dim=0, embed_model="text-embedding-3-small")
        assert config.resolve_embed_dim() == 1536

    def test_vendor_prefixed_model_lookup(self) -> None:
        """GitHub Models vendor-prefixed names are also recognized."""
        config = Config(embed_dim=0, embed_model="openai/text-embedding-3-small")
        assert config.resolve_embed_dim() == 1536

    def test_unknown_model_returns_zero(self) -> None:
        config = Config(embed_dim=0, embed_model="some-unknown-model")
        assert config.resolve_embed_dim() == 0
