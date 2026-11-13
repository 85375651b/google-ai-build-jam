"""Apply the additive entity migration to an existing SQLite database, with backup."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import create_engine, inspect
from backend.migrations import migrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    args = parser.parse_args()
    path = args.database.resolve()
    if not path.is_file():
        parser.error('The existing database file was not found.')
    engine = create_engine('sqlite:///' + str(path))
    try:
        if 'worlds' not in inspect(engine).get_table_names():
            parser.error('This file is not an existing World Bible database.')
        backup = migrate(engine)
        print('Structured canon schema is current.')
        print('Backup: ' + (backup or 'migration already applied; no additional backup needed'))
    finally:
        engine.dispose()


if __name__ == '__main__':
    main()
