"""Database connection, session management, and SQLite FTS5 initialization."""

from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from backend.config import settings, PROJECT_ROOT


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all relational tables."""
    pass


def _resolve_sqlite_url(db_url: str) -> str:
    """Ensure parent directory exists for SQLite database files."""
    if db_url.startswith("sqlite:///"):
        raw_path = db_url.replace("sqlite:///", "", 1)
        db_path = Path(raw_path)
        if not db_path.is_absolute():
            db_path = (PROJECT_ROOT / db_path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"
    return db_url


RESOLVED_DATABASE_URL = _resolve_sqlite_url(settings.database_url)

engine = create_engine(
    RESOLVED_DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:
    """Enable WAL mode and foreign key constraints on every SQLite connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def init_db() -> None:
    """Initialize all relational tables and SQLite FTS5 virtual tables."""
    import backend.models  # noqa: F401 - ensure models are registered
    # Back up legacy data before any new DDL. Empty databases need base tables first.
    from sqlalchemy import inspect
    from backend.migrations import migrate
    if 'worlds' in inspect(engine).get_table_names():
        migrate(engine)
    Base.metadata.create_all(bind=engine)
    migrate(engine)

    with engine.begin() as conn:
        # FTS5 index for reference book chunks (BM25 keyword retrieval)
        conn.execute(
            text(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS reference_chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    chapter_title,
                    section_title,
                    content
                );
                """
            )
        )
        # FTS5 index for established world facts (hybrid canon lookup)
        conn.execute(
            text(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                    fact_id UNINDEXED,
                    world_id UNINDEXED,
                    title,
                    content
                );
                """
            )
        )


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
