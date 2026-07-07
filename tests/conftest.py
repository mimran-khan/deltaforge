"""Shared test fixtures — ensure runtime directories exist in CI."""

import pytest

from config import settings


@pytest.fixture(autouse=True, scope="session")
def ensure_runtime_dirs():
    """Create data/ and logs/ dirs so tests that write HALT files or open DBs succeed."""
    settings.ensure_dirs()
