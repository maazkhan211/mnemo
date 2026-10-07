"""Local embeddings (sentence-transformers) and vector storage (ChromaDB)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings


class Embedder:
    """Wraps a sentence-transformers model. Vectors are L2-normalised, so cosine == dot product."""

    def __init__(self, model_name: str):
        # Imported lazily: torch is slow to import and tests may inject a fake embedder.
        from sentence_transformers import SentenceTransformer

        try:
            # Prefer the cached copy so the app starts offline and never stalls on a network check.
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except OSError:
            self.model = SentenceTransformer(model_name)  # first run: download

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


class VectorStore:
    COLLECTION = "notes"

    def __init__(self, path: Path):
        self.client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self.collection = self._get_collection()

    def _get_collection(self):
        return self.client.get_or_create_collection(self.COLLECTION, metadata={"hnsw:space": "cosine"})

    def upsert(self, note_id: int, embedding: list[float], created: datetime) -> None:
        self.upsert_many([note_id], [embedding], [created])

    def upsert_many(self, note_ids: list[int], embeddings: list[list[float]], created: list[datetime]) -> None:
        if not note_ids:
            return
        self.collection.upsert(
            ids=[str(i) for i in note_ids],
            embeddings=embeddings,
            metadatas=[{"note_id": i, "created_at": int(c.timestamp())} for i, c in zip(note_ids, created)],
        )

    def delete(self, note_id: int) -> None:
        self.collection.delete(ids=[str(note_id)])

    def get_embedding(self, note_id: int) -> list[float] | None:
        result = self.collection.get(ids=[str(note_id)], include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return list(embeddings[0])

    def query(
        self,
        embedding: list[float],
        k: int,
        exclude_id: int | None = None,
        since: datetime | None = None,
    ) -> list[tuple[int, float]]:
        """Return up to k (note_id, cosine_similarity) pairs, most similar first."""
        total = self.collection.count()
        if total == 0 or k <= 0:
            return []
        n_results = min(k + (1 if exclude_id is not None else 0), total)
        where = {"created_at": {"$gte": int(since.timestamp())}} if since else None
        result = self.collection.query(
            query_embeddings=[embedding], n_results=n_results, where=where, include=["distances"]
        )
        hits = []
        for id_str, distance in zip(result["ids"][0], result["distances"][0]):
            note_id = int(id_str)
            if note_id == exclude_id:
                continue
            hits.append((note_id, 1.0 - distance))  # cosine distance -> similarity
        return hits[:k]

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.COLLECTION)
        self.collection = self._get_collection()
