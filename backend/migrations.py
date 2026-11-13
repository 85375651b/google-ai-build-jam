"""Repeatable additive migration. Legacy prose/IDs are deliberately not inferred."""
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from sqlalchemy import inspect, text

VERSION = '20261113_structured_canon_v1'
TABLES = ('entities', 'entity_properties', 'entity_names', 'entity_relationships',
          'entity_constraints', 'entity_revisions', 'canon_operation_batches',
          'naming_policies', 'canon_sync_issues')


def migrate(engine, backup=True):
    from backend.database import Base
    import backend.models  # register tables
    existing = inspect(engine).get_table_names()
    if 'schema_migrations' in existing:
        with engine.connect() as conn:
            if conn.execute(text('SELECT 1 FROM schema_migrations WHERE version=:v'), {'v': VERSION}).first():
                return None
    backup_path = None
    if backup and 'worlds' in existing and engine.dialect.name == 'sqlite' and engine.url.database != ':memory:':
        path = Path(engine.url.database)
        directory = path.parent / 'backups'
        directory.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup_path = directory / f'{path.stem}-before-structured-{stamp}.db'
        source = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        target = sqlite3.connect(str(backup_path))
        try:
            source.backup(target)
        finally:
            source.close(); target.close()
    with engine.begin() as conn:
        Base.metadata.create_all(conn, tables=[Base.metadata.tables[name] for name in TABLES])
        conn.execute(text('CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)'))
        conn.execute(text('INSERT OR IGNORE INTO schema_migrations VALUES (:v, :d)'),
                     {'v': VERSION, 'd': datetime.now(timezone.utc).isoformat()})
    return str(backup_path) if backup_path else None
