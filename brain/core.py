"""The Brain: saving notes, semantic search, auto-connections, resurfacing and the note graph."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Protocol

import networkx as nx

from brain.config import DATA_DIR, EMBED_MODEL, MAX_CONNECTIONS, MIN_CONNECTION_SCORE
from brain.db import Database, Note
from brain.enrich import detect_type, normalize_tags
from brain.vectors import Embedder, VectorStore


class EmbedderLike(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class Resurfaced:
    old: Note
    recent: Note | None
    score: float | None


class Brain:
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        embedder: EmbedderLike | None = None,
        tagger: Callable[[str], list[str]] | None = None,
        link_enricher: Callable[[str], str] | None = None,
        max_connections: int = MAX_CONNECTIONS,
        min_connection_score: float = MIN_CONNECTION_SCORE,
    ):
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(data_dir / "brain.db")
        self.vectors = VectorStore(data_dir / "chroma")
        self.embedder = embedder or Embedder(EMBED_MODEL)
        self.tagger = tagger
        self.link_enricher = link_enricher
        self.max_connections = max_connections
        self.min_connection_score = min_connection_score
        self.last_tag_error: str | None = None

    # --- capture -----------------------------------------------------------

    def save_note(
        self,
        text: str,
        type: str | None = None,
        tags: list[str] | str | None = None,
        auto_tag: bool = True,
        timestamp: datetime | None = None,
    ) -> tuple[Note, list[tuple[Note, float]]]:
        """Save a note, embed it, and connect it to its most similar existing notes.

        Returns the saved note and the connections that were made.
        """
        text = text.strip()
        if not text:
            raise ValueError("Note is empty")

        type = type or detect_type(text)
        if self.link_enricher and type == "link":
            text = self.link_enricher(text)

        tags = normalize_tags(tags)
        self.last_tag_error = None
        if auto_tag and self.tagger:
            try:
                tags = normalize_tags(tags + self.tagger(text))
            except Exception as e:  # tagging is a nice-to-have; never lose a note over it
                self.last_tag_error = str(e)

        note = self.db.add_note(text, type, tags, timestamp)
        embedding = self.embedder.encode([text])[0]

        # Query before inserting the new vector, so the note can't match itself.
        related = self._similar_to(embedding, exclude_id=note.id)
        self.db.add_connections(note.id, related)
        self.vectors.upsert(note.id, embedding, note.created)

        notes = self.db.get_notes([nid for nid, _ in related])
        return note, [(notes[nid], score) for nid, score in related if nid in notes]

    def _similar_to(self, embedding: list[float], exclude_id: int | None) -> list[tuple[int, float]]:
        hits = self.vectors.query(embedding, k=self.max_connections, exclude_id=exclude_id)
        return [(nid, score) for nid, score in hits if score >= self.min_connection_score]

    def update_note(
        self,
        note_id: int,
        text: str | None = None,
        type: str | None = None,
        tags: list[str] | str | None = None,
    ) -> Note:
        """Edit a note in place. Arguments left as None keep their current value.

        If the text changes, the note is re-embedded and its connections are recomputed.
        The original timestamp is kept.
        """
        note = self.db.get_note(note_id)
        if note is None:
            raise KeyError(f"Note #{note_id} not found")

        new_text = note.text if text is None else text.strip()
        if not new_text:
            raise ValueError("Note is empty")
        new_type = type or note.type
        new_tags = list(note.tags) if tags is None else normalize_tags(tags)

        updated = self.db.update_note(note_id, new_text, new_type, new_tags)
        if new_text != note.text:
            embedding = self.embedder.encode([new_text])[0]
            self.db.delete_connections(note_id)
            self.db.add_connections(note_id, self._similar_to(embedding, exclude_id=note_id))
            self.vectors.upsert(note_id, embedding, updated.created)
        return updated

    def delete_note(self, note_id: int) -> None:
        self.db.delete_note(note_id)  # connections are removed by ON DELETE CASCADE
        self.vectors.delete(note_id)

    def set_tags(self, note_id: int, tags: list[str] | str) -> None:
        self.db.set_tags(note_id, normalize_tags(tags))

    # --- retrieval ---------------------------------------------------------

    def search(self, query: str, k: int = 5, min_score: float = 0.0) -> list[tuple[Note, float]]:
        """Semantic search: embed the query, find nearest note vectors, fetch the notes from SQLite."""
        query = query.strip()
        if not query:
            return []
        embedding = self.embedder.encode([query])[0]
        hits = [(nid, s) for nid, s in self.vectors.query(embedding, k=k) if s >= min_score]
        notes = self.db.get_notes([nid for nid, _ in hits])
        return [(notes[nid], score) for nid, score in hits if nid in notes]

    def connections(self, note_id: int) -> list[tuple[Note, float]]:
        return self.db.get_connections(note_id)

    def resurface(
        self,
        min_age_days: float = 7,
        recent_days: float = 7,
        rng: random.Random | None = None,
    ) -> Resurfaced | None:
        """Pick a random older note and pair it with its most similar recent note ("remember this?").

        Falls back to the most similar note of any age when nothing recent is related.
        Returns None when there are no notes older than `min_age_days`.
        """
        rng = rng or random.Random()
        now = datetime.now()
        candidates = self.db.notes_before(now - timedelta(days=min_age_days))
        if not candidates:
            return None
        old = rng.choice(candidates)

        embedding = self.vectors.get_embedding(old.id)
        if embedding is None:
            embedding = self.embedder.encode([old.text])[0]

        hits = self.vectors.query(embedding, k=1, exclude_id=old.id, since=now - timedelta(days=recent_days))
        if not hits:
            hits = self.vectors.query(embedding, k=1, exclude_id=old.id)
        if not hits:
            return Resurfaced(old, None, None)
        recent = self.db.get_note(hits[0][0])
        return Resurfaced(old, recent, hits[0][1] if recent else None)

    # --- overview ----------------------------------------------------------

    def all_notes(self) -> list[Note]:
        return self.db.all_notes()

    def tag_counts(self) -> dict[str, int]:
        return self.db.tag_counts()

    def stats(self) -> dict[str, int]:
        return {
            "notes": self.db.count_notes(),
            "connections": len(self.db.all_connections()),
            "tags": len(self.db.tag_counts()),
        }

    def graph(self, min_score: float = 0.0) -> nx.Graph:
        """Nodes are notes (with the Note in the `note` attribute), edges are stored connections."""
        g = nx.Graph()
        for note in self.db.all_notes():
            g.add_node(note.id, note=note)
        for a, b, score in self.db.all_connections(min_score):
            if a in g and b in g:
                g.add_edge(a, b, weight=score)
        return g

    # --- maintenance -------------------------------------------------------

    def rebuild(self, batch_size: int = 64) -> None:
        """Re-embed every note and recompute all connections.

        Use after changing the embedding model or connection settings, or if the vector store is lost.
        """
        notes = self.db.all_notes(newest_first=False)
        self.vectors.reset()
        for i in range(0, len(notes), batch_size):
            batch = notes[i : i + batch_size]
            embeddings = self.embedder.encode([n.text for n in batch])
            self.vectors.upsert_many([n.id for n in batch], embeddings, [n.created for n in batch])

        self.db.clear_connections()
        for note in notes:
            embedding = self.vectors.get_embedding(note.id)
            if embedding is not None:
                self.db.add_connections(note.id, self._similar_to(embedding, exclude_id=note.id))
