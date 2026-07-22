"""Application services: process-wide state and the search logic.

The CLIP model is loaded lazily on the first semantic query. Browsing, text
search and stats therefore work instantly, even on a cold start, and a reviewer
who never touches semantic search never pays the model load.
"""

from __future__ import annotations

import logging
import threading

from .config import Settings
from .db import connect, get_meta
from .embeddings import ClipEmbedder, Embedder, VectorIndex
from .models import ImageSummary
from .repository import ImageRepository

logger = logging.getLogger(__name__)


class DatasetNotIngested(RuntimeError):
    """Raised when the API starts before `python -m backend.ingest` has run."""


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.db_path.exists():
            raise DatasetNotIngested(
                f"No dataset found at {settings.db_path}. "
                "Run `python -m backend.ingest` first (see README)."
            )
        self.conn = connect(settings.db_path, read_only=True)
        self.repo = ImageRepository(self.conn)
        self.index = (
            VectorIndex.load(settings.embeddings_path)
            if settings.embeddings_path.exists()
            else None
        )
        self._embedder: Embedder | None = None
        self._embedder_lock = threading.Lock()

    @property
    def embedding_model_name(self) -> str | None:
        return get_meta(self.conn, "embedding_model")

    @property
    def semantic_available(self) -> bool:
        return self.index is not None

    def embedder(self) -> Embedder:
        """Load the model once, under a lock so concurrent requests don't race."""
        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:
                    logger.info("Loading CLIP model %s ...", self.settings.clip_model)
                    self._embedder = ClipEmbedder(
                        self.settings.clip_model, self.settings.clip_pretrained
                    )
        return self._embedder

    def close(self) -> None:
        self.conn.close()

    # --- search ---------------------------------------------------------

    def semantic_search(self, query: str, *, split: str | None, limit: int) -> list[ImageSummary]:
        if self.index is None:
            raise DatasetNotIngested("Embeddings are missing; re-run the ingestion.")
        vector = self.embedder().embed_texts([query])[0]
        candidates = self.repo.row_indices_for_split(split) if split else None
        ranked = self.index.search(vector, limit, candidates=candidates)
        return self.repo.summaries_by_row_index(ranked)

    def similar_images(self, image_id: str, *, limit: int) -> list[ImageSummary] | None:
        if self.index is None:
            raise DatasetNotIngested("Embeddings are missing; re-run the ingestion.")
        row_index = self.repo.row_index_of(image_id)
        if row_index is None:
            return None
        ranked = self.index.search(
            self.index.vector(row_index), limit, exclude=row_index
        )
        return self.repo.summaries_by_row_index(ranked)

    def text_search(self, query: str, *, split: str | None, limit: int) -> list[ImageSummary]:
        sanitized = _sanitize_fts(query)
        if not sanitized:  # nothing searchable survived, e.g. a query of only "*"
            return []
        return self.repo.search_captions(sanitized, split=split, limit=limit)


def _sanitize_fts(query: str) -> str:
    """Turn free text into a safe FTS5 query.

    User input reaches FTS5 as a query expression, where characters like ``"``,
    ``*`` or ``NEAR`` have syntax meaning and can raise. Each token is stripped to
    its alphanumeric core and quoted, which gives predictable AND-of-terms
    behaviour and removes any chance of a syntax error.
    """
    tokens = ["".join(ch for ch in token if ch.isalnum() or ch == "'") for token in query.split()]
    return " ".join(f'"{token}"' for token in tokens if token)
