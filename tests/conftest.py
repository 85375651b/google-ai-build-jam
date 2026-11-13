"""Keep tests isolated from author data and online research."""
import os
import tempfile
from pathlib import Path

import pytest

_test_directory = tempfile.TemporaryDirectory(prefix="worldbuilding-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_test_directory.name) / 'test.db'}"
os.environ["TAVILY_API_KEY"] = ""


@pytest.fixture(scope="session", autouse=True)
def initialize_test_database():
    from backend.database import SessionLocal, engine, init_db
    from backend.services.book_ingestion_service import book_ingestion_service

    init_db()
    with SessionLocal() as db:
        book_ingestion_service.ensure_primary_book_ingested(db)
    yield
    engine.dispose()
    _test_directory.cleanup()
