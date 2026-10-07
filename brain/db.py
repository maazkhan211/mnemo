"""SQLite storage for note metadata and the connections between notes."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    text      TEXT NOT NULL,
    timestamp TEXT NOT NULL,              -- local time, ISO 8601
    tags      TEXT NOT NULL DEFAULT '',   -- comma-separated, lowercase
    type      TEXT NOT NULL DEFAULT 'note'
);

CREATE TABLE IF NOT EXISTS connections (
    note_id          INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    related_note_id  INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    similarity_score REAL NOT NULL,
    PRIMARY KEY (note_id, related_note_id)
);

CREATE INDEX IF NOT EXISTS idx_notes_timestamp ON notes(timestamp);
CREATE INDEX IF NOT EXISTS idx_connections_related ON connections(related_note_id);
"""


@dataclass(frozen=True)
class Note:
    id: int
    text: str
    timestamp: str
    tags: tuple[str, ...]
    type: str

    @property
    def created(self) -> datetime:
        return datetime.fromisoformat(self.timestamp)

    @property
    def title(self) -> str:
        """First line of the note, truncated: used as a label in lists and the graph."""
        first_line = self.text.strip().splitlines()[0] if self.text.strip() else ""
        return first_line if len(first_line) <= 60 else first_line[:57] + "..."

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Note:
        tags = tuple(t for t in row["tags"].split(",") if t)
        return cls(row["id"], row["text"], row["timestamp"], tags, row["type"])


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        # A fresh connection per operation keeps this safe to use from
        # Streamlit's per-session threads.
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- notes -------------------------------------------------------------

    def add_note(self, text: str, type: str, tags: list[str], timestamp: datetime | None = None) -> Note:
        ts = (timestamp or datetime.now()).isoformat(timespec="seconds")
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO notes (text, timestamp, tags, type) VALUES (?, ?, ?, ?)",
                (text, ts, ",".join(tags), type),
            )
            note_id = cur.lastrowid
        return Note(note_id, text, ts, tuple(tags), type)

    def get_note(self, note_id: int) -> Note | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return Note.from_row(row) if row else None

    def get_notes(self, ids: list[int]) -> dict[int, Note]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._conn() as conn:
            rows = conn.execute(f"SELECT * FROM notes WHERE id IN ({placeholders})", ids).fetchall()
        return {row["id"]: Note.from_row(row) for row in rows}

    def all_notes(self, newest_first: bool = True) -> list[Note]:
        order = "DESC" if newest_first else "ASC"
        with self._conn() as conn:
            rows = conn.execute(f"SELECT * FROM notes ORDER BY timestamp {order}, id {order}").fetchall()
        return [Note.from_row(r) for r in rows]

    def notes_before(self, cutoff: datetime) -> list[Note]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM notes WHERE timestamp < ?", (cutoff.isoformat(timespec="seconds"),)
            ).fetchall()
        return [Note.from_row(r) for r in rows]

    def update_note(self, note_id: int, text: str, type: str, tags: list[str]) -> Note | None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE notes SET text = ?, type = ?, tags = ? WHERE id = ?",
                (text, type, ",".join(tags), note_id),
            )
        return self.get_note(note_id)

    def set_tags(self, note_id: int, tags: list[str]) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE notes SET tags = ? WHERE id = ?", (",".join(tags), note_id))

    def delete_note(self, note_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))

    def count_notes(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]

    def tag_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for note in self.all_notes():
            for tag in note.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    # --- connections -------------------------------------------------------

    def add_connections(self, note_id: int, related: list[tuple[int, float]]) -> None:
        """Store connections in both directions, so older notes also link forward to newer ones."""
        rows = []
        for related_id, score in related:
            rows.append((note_id, related_id, score))
            rows.append((related_id, note_id, score))
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO connections (note_id, related_note_id, similarity_score) VALUES (?, ?, ?)",
                rows,
            )

    def get_connections(self, note_id: int) -> list[tuple[Note, float]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT n.*, c.similarity_score
                FROM connections c JOIN notes n ON n.id = c.related_note_id
                WHERE c.note_id = ?
                ORDER BY c.similarity_score DESC
                """,
                (note_id,),
            ).fetchall()
        return [(Note.from_row(r), r["similarity_score"]) for r in rows]

    def all_connections(self, min_score: float = 0.0) -> list[tuple[int, int, float]]:
        """Each undirected edge once, as (lower_id, higher_id, score)."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT note_id, related_note_id, similarity_score FROM connections
                WHERE note_id < related_note_id AND similarity_score >= ?
                """,
                (min_score,),
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def delete_connections(self, note_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM connections WHERE note_id = ? OR related_note_id = ?", (note_id, note_id))

    def clear_connections(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM connections")
