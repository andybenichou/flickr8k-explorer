"""Test fixtures: a miniature dataset built on disk, so no network or model is needed."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.db import connect, init_schema, set_meta, transaction

CAPTIONS = [
    "A brown dog runs across a green field",
    "Two children play with a red ball on the beach",
    "A man in a blue jacket climbs a rocky mountain",
    "A white cat sleeps on a wooden chair",
]

DIM = 8


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def ingested(settings: Settings) -> Settings:
    """Create a 4-image dataset with deterministic, well-separated embeddings."""
    settings.images_dir.mkdir(parents=True)
    settings.thumbs_dir.mkdir(parents=True)

    conn = connect(settings.db_path)
    init_schema(conn)

    vectors = np.zeros((len(CAPTIONS), DIM), dtype=np.float32)
    with transaction(conn):
        for row_index, caption in enumerate(CAPTIONS):
            split = "train" if row_index < 2 else "test"
            image_id = f"{split}-{row_index:05d}"
            for directory in (settings.images_dir, settings.thumbs_dir):
                Image.new("RGB", (100, 80), color=(row_index * 40, 100, 200)).save(
                    directory / f"{image_id}.jpg"
                )
            conn.execute(
                "INSERT INTO images(id, split, row_index, width, height, file_size,"
                " umap_x, umap_y, cluster) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (image_id, split, row_index, 100, 80, 1234, row_index / 4, 0.5, row_index % 2),
            )
            for caption_index in range(2):
                text = f"{caption} ({caption_index})"
                conn.execute(
                    "INSERT INTO captions(image_id, caption_index, text, n_words)"
                    " VALUES (?, ?, ?, ?)",
                    (image_id, caption_index, text, len(text.split())),
                )
                conn.execute(
                    "INSERT INTO captions_fts(text, image_id) VALUES (?, ?)", (text, image_id)
                )
            # One-hot vectors: nearest neighbours are then exactly predictable.
            vectors[row_index, row_index] = 1.0

        # Two labelled clusters, matching the `row_index % 2` assignment above.
        conn.executemany(
            "INSERT INTO clusters(id, label, size) VALUES (?, ?, ?)",
            [(0, "even cluster", 2), (1, "odd cluster", 2)],
        )

        set_meta(conn, "dataset", "test/flickr8k")
        set_meta(conn, "embedding_model", "test-embedder")
        set_meta(conn, "embedding_dim", str(DIM))
        set_meta(conn, "has_projection", "1")
        set_meta(conn, "ingested_at", "2026-01-01T00:00:00+00:00")

    conn.close()
    np.save(settings.embeddings_path, vectors)
    return settings


class FakeEmbedder:
    """Maps a query to a one-hot vector via a keyword, avoiding a CLIP download."""

    name = "test-embedder"
    dim = DIM
    KEYWORDS = {"dog": 0, "children": 1, "mountain": 2, "cat": 3}

    def embed_texts(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            index = next((i for word, i in self.KEYWORDS.items() if word in text.lower()), 0)
            out[row, index] = 1.0
        return out

    def embed_images(self, images):  # pragma: no cover - unused in tests
        raise NotImplementedError


@pytest.fixture
def client(ingested: Settings):
    from backend.app.api import create_app

    app = create_app(ingested)

    # The app builds its state in `lifespan`, which needs the real settings; patch
    # the getter the lifespan uses so it picks up the temporary dataset.
    import backend.app.api as api_module

    original = api_module.get_settings
    api_module.get_settings = lambda: ingested
    try:
        with TestClient(app) as test_client:
            test_client.app.state.app_state._embedder = FakeEmbedder()
            yield test_client
    finally:
        api_module.get_settings = original
