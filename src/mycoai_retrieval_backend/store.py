import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class DataStore:
    def __init__(self, path: Path | str = ":memory:") -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self.migrate()

    def close(self) -> None:
        self._connection.close()

    def migrate(self) -> None:
        self._connection.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS species (
                species_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                name_key TEXT NOT NULL UNIQUE,
                description TEXT,
                taxonomic_info TEXT,
                reference_images TEXT NOT NULL DEFAULT '[]',
                archived_alias_of TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                is_archived INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS strains (
                strain_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                species_id TEXT NOT NULL REFERENCES species(species_id),
                created_at TEXT NOT NULL,
                is_archived INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL CHECK (
                    source IN ('curated_primary', 'incoming_low_quality', 'user_upload')
                )
            );

            CREATE TABLE IF NOT EXISTS images (
                image_id TEXT PRIMARY KEY,
                strain_id TEXT NOT NULL REFERENCES strains(strain_id),
                media TEXT NOT NULL,
                file_path TEXT NOT NULL,
                segments TEXT NOT NULL DEFAULT '[]',
                indexed_in_qdrant INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                is_archived INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                detail TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._connection.commit()

    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection:
            yield self._connection

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
        cursor = self._connection.execute(sql, params)
        self._connection.commit()
        return cursor

    def query(self, sql: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
        return list(self._connection.execute(sql, params))

    def query_one(
        self, sql: str, params: tuple[object, ...] = ()
    ) -> sqlite3.Row | None:
        cursor = self._connection.execute(sql, params)
        result = cursor.fetchone()
        return result if result is not None else None

    def log(self, action: str, entity_type: str, entity_id: str, detail: str) -> None:
        self.execute(
            "INSERT INTO audit_log VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid4()), action, entity_type, entity_id, detail, utc_now()),
        )
