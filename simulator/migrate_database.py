from pathlib import Path
from sqlite3 import Connection


def migrate_database(connection: Connection):
    migrations = get_migration_paths()
    current_version = connection.execute("PRAGMA user_version").fetchone()[0]
    migrations = [(v, p) for v, p in migrations if v > current_version]
    for version, path in migrations:
        print(f"Applying migration {version}: {path}")
        script = f"""
            BEGIN;
            {path.read_text()};
            PRAGMA user_version = {version};
            COMMIT;
        """
        connection.executescript(script)


def get_migration_paths() -> list[tuple[int, Path]]:
    MIGRATIONS_PATH = Path("simulator/migrations")

    return list(
        enumerate(sorted(MIGRATIONS_PATH.glob("*.sql"), key=lambda p: p.name), start=1)
    )
