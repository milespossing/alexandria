"""Pytest configuration and fixtures for Alexandria integration tests.

Fixtures connect to isolated Qdrant and Ollama instances started by
``scripts/integration-test.sh`` on non-default ports (16333 / 14434).
This avoids any collision with a locally-installed Alexandria.

Override via environment variables:
    ALEXANDRIA_TEST_QDRANT_URL   (default: http://localhost:16333)
    ALEXANDRIA_TEST_OLLAMA_HOST  (default: http://localhost:14434)
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest

from alexandria.config import Config
from alexandria.embedder import Embedder
from alexandria.store import Store

# Isolated test ports — different from the dev defaults (6333 / 11434).
TEST_QDRANT_URL = os.environ.get("ALEXANDRIA_TEST_QDRANT_URL", "http://localhost:16333")
TEST_OLLAMA_HOST = os.environ.get("ALEXANDRIA_TEST_OLLAMA_HOST", "http://localhost:14434")
TEST_EMBED_MODEL = os.environ.get("ALEXANDRIA_TEST_EMBED_MODEL", "nomic-embed-text")
TEST_CONTEXT = "alexandria_test"


@pytest.fixture(scope="session")
def test_config() -> Config:
    """Return a Config pointing at the isolated test services."""
    return Config(
        qdrant_url=TEST_QDRANT_URL,
        ollama_host=TEST_OLLAMA_HOST,
        embed_model=TEST_EMBED_MODEL,
    )


@pytest.fixture(scope="session")
def test_store(test_config: Config) -> Store:
    """Return a Store connected to the test Qdrant.

    The integration-test.sh script ensures Qdrant is already running.
    """
    store = Store(test_config)
    if not store.is_available():
        pytest.skip("Qdrant not reachable — run scripts/integration-test.sh")
    return store


@pytest.fixture(scope="session")
def test_embedder(test_config: Config) -> Embedder:
    """Return an Embedder connected to the test Ollama.

    The integration-test.sh script ensures Ollama is running and the
    embedding model has been pulled.
    """
    embedder = Embedder(test_config)
    if not embedder.is_available():
        pytest.skip("Ollama / embedding model not available — run scripts/integration-test.sh")
    return embedder


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root (the directory containing this test suite)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _cleanup_test_context(request: pytest.FixtureRequest) -> Generator[None]:
    """Drop the test context before/after tests that use the store.

    Only runs when the test actually depends on ``test_store`` (directly or
    transitively).  Unit tests that don't need Qdrant are left alone.
    """
    # Only clean up if the test (or its class) uses the test_store fixture.
    fixture_names = request.fixturenames
    if "test_store" not in fixture_names:
        yield
        return

    store: Store = request.getfixturevalue("test_store")
    store.drop_context(TEST_CONTEXT)
    yield
    store.drop_context(TEST_CONTEXT)
