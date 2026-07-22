"""One-shot ingestion pipeline: HuggingFace parquet -> local images + SQLite + vectors.

Run with ``python -m backend.ingest``. The pipeline is idempotent per stage and
each stage can be skipped, so a failure late in the run (e.g. UMAP) does not cost
the expensive image embedding pass.

Stages
------
1. download   pull the parquet shards from the HuggingFace hub (~275 MB)
2. decode     write full images + thumbnails to disk, fill SQLite (images/captions/FTS)
3. embed      CLIP image embeddings -> data/embeddings.npy
4. project    UMAP to 2D + KMeans clusters -> stored back in SQLite
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from .app.config import Settings, get_settings
from .app.db import connect, init_schema, set_meta, transaction
from .app.embeddings import ClipEmbedder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ingest")

CAPTION_COLUMNS = [f"caption_{i}" for i in range(5)]
SPLITS = ("train", "validation", "test")
WORD_RE = re.compile(r"[\w']+")


# --------------------------------------------------------------------------
# stage 1: download
# --------------------------------------------------------------------------


def download_shards(settings: Settings) -> dict[str, list[Path]]:
    """Fetch the parquet shards, returning ``{split: [paths]}``.

    ``hf_hub_download`` caches under ~/.cache/huggingface, so re-running is free.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [
        f
        for f in list_repo_files(settings.hf_dataset_id, repo_type="dataset")
        if f.startswith("data/") and f.endswith(".parquet")
    ]
    shards: dict[str, list[Path]] = {split: [] for split in SPLITS}
    for filename in sorted(files):
        split = Path(filename).name.split("-")[0]
        if split not in shards:
            logger.warning("Ignoring shard with unexpected split: %s", filename)
            continue
        logger.info("Downloading %s ...", filename)
        local = hf_hub_download(settings.hf_dataset_id, filename, repo_type="dataset")
        shards[split].append(Path(local))
    return {split: paths for split, paths in shards.items() if paths}


# --------------------------------------------------------------------------
# stage 2: decode images and fill SQLite
# --------------------------------------------------------------------------


def decode_and_store(
    conn: sqlite3.Connection,
    settings: Settings,
    shards: dict[str, list[Path]],
    limit_per_split: int | None,
) -> int:
    """Write images and thumbnails to disk and populate the tables.

    Returns the number of stored images. ``row_index`` is assigned sequentially
    here and is the join key between SQLite and the embedding matrix.
    """
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    settings.thumbs_dir.mkdir(parents=True, exist_ok=True)

    with transaction(conn):
        conn.execute("DELETE FROM captions_fts")
        conn.execute("DELETE FROM captions")
        conn.execute("DELETE FROM images")

    row_index = 0
    for split, paths in shards.items():
        seen = 0
        for path in paths:
            parquet = pq.ParquetFile(path)
            total = parquet.metadata.num_rows
            with tqdm(total=min(total, limit_per_split or total), desc=f"{split}", unit="img") as bar:
                for batch in parquet.iter_batches(batch_size=64):
                    records = batch.to_pylist()
                    rows, caption_rows = [], []
                    for record in records:
                        if limit_per_split is not None and seen >= limit_per_split:
                            break
                        image_id = f"{split}-{seen:05d}"
                        payload = record["image"]
                        raw = payload["bytes"] if isinstance(payload, dict) else payload
                        try:
                            width, height, size = _write_image_files(raw, image_id, settings)
                        except Exception as exc:  # a corrupt row must not kill the run
                            logger.warning("Skipping %s: %s", image_id, exc)
                            continue
                        rows.append((image_id, split, row_index, width, height, size))
                        for i, column in enumerate(CAPTION_COLUMNS):
                            text = (record.get(column) or "").strip()
                            if text:
                                caption_rows.append(
                                    (image_id, i, text, len(WORD_RE.findall(text)))
                                )
                        row_index += 1
                        seen += 1
                        bar.update(1)

                    with transaction(conn):
                        conn.executemany(
                            "INSERT INTO images(id, split, row_index, width, height, file_size)"
                            " VALUES (?, ?, ?, ?, ?, ?)",
                            rows,
                        )
                        conn.executemany(
                            "INSERT INTO captions(image_id, caption_index, text, n_words)"
                            " VALUES (?, ?, ?, ?)",
                            caption_rows,
                        )
                        conn.executemany(
                            "INSERT INTO captions_fts(text, image_id) VALUES (?, ?)",
                            [(text, image_id) for image_id, _, text, _ in caption_rows],
                        )

                    if limit_per_split is not None and seen >= limit_per_split:
                        break
            if limit_per_split is not None and seen >= limit_per_split:
                break

    with transaction(conn):
        conn.execute("INSERT INTO captions_fts(captions_fts) VALUES ('optimize')")
    return row_index


def _write_image_files(raw: bytes, image_id: str, settings: Settings) -> tuple[int, int, int]:
    """Save the full image and a thumbnail; return ``(width, height, bytes)``."""
    image = Image.open(io.BytesIO(raw))
    image.load()
    width, height = image.size

    full_path = settings.images_dir / f"{image_id}.jpg"
    full_path.write_bytes(raw)

    thumb = image.convert("RGB")
    thumb.thumbnail((settings.thumbnail_size, settings.thumbnail_size), Image.LANCZOS)
    thumb.save(settings.thumbs_dir / f"{image_id}.jpg", "JPEG", quality=82, optimize=True)

    return width, height, len(raw)


# --------------------------------------------------------------------------
# stage 3: embeddings
# --------------------------------------------------------------------------


def compute_embeddings(conn: sqlite3.Connection, settings: Settings) -> tuple[str, int]:
    """Embed every stored image with CLIP, in ``row_index`` order."""
    rows = conn.execute("SELECT id FROM images ORDER BY row_index").fetchall()
    image_ids = [r["id"] for r in rows]
    if not image_ids:
        raise RuntimeError("No images in the database; run the decode stage first.")

    embedder = ClipEmbedder(settings.clip_model, settings.clip_pretrained)
    logger.info("Embedding %d images with %s on %s", len(image_ids), embedder.name, embedder.device)

    matrix = np.zeros((len(image_ids), embedder.dim), dtype=np.float32)
    batch_size = settings.embed_batch_size
    started = time.time()
    for start in tqdm(range(0, len(image_ids), batch_size), desc="embedding", unit="batch"):
        chunk = image_ids[start : start + batch_size]
        images = [Image.open(settings.images_dir / f"{image_id}.jpg") for image_id in chunk]
        matrix[start : start + len(chunk)] = embedder.embed_images(images)
        for image in images:
            image.close()

    np.save(settings.embeddings_path, matrix)
    logger.info(
        "Wrote %s (%.1f MB) in %.0fs",
        settings.embeddings_path.name,
        matrix.nbytes / 1e6,
        time.time() - started,
    )
    return embedder.name, embedder.dim


# --------------------------------------------------------------------------
# stage 4: 2D projection + clustering
# --------------------------------------------------------------------------


def compute_projection(conn: sqlite3.Connection, settings: Settings) -> None:
    """Project the embeddings to 2D with UMAP and label clusters with KMeans.

    Both are precomputed at ingestion so the map endpoint is a plain SQL read.
    """
    import umap
    from sklearn.cluster import KMeans

    matrix = np.load(settings.embeddings_path)
    logger.info("Running UMAP on %s ...", matrix.shape)
    reducer = umap.UMAP(
        n_neighbors=settings.umap_neighbors,
        min_dist=settings.umap_min_dist,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(matrix)

    n_clusters = min(settings.n_clusters, len(matrix))
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(matrix)

    # Normalise to [0, 1] so the frontend does not rescale. Percentiles rather
    # than min/max: a handful of UMAP outliers would otherwise squeeze the whole
    # cloud into a corner. Outliers clamp onto the border, which is honest enough
    # for a navigation aid.
    lo, hi = np.percentile(coords, 1, axis=0), np.percentile(coords, 99, axis=0)
    coords = np.clip((coords - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)

    with transaction(conn):
        conn.executemany(
            "UPDATE images SET umap_x = ?, umap_y = ?, cluster = ? WHERE row_index = ?",
            [
                (float(x), float(y), int(label), row)
                for row, ((x, y), label) in enumerate(zip(coords, labels))
            ],
        )
        set_meta(conn, "has_projection", "1")
    logger.info("Stored 2D projection with %d clusters", n_clusters)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only ingest N images per split. Use a small value (e.g. 200) for a quick smoke test.",
    )
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip CLIP embeddings.")
    parser.add_argument("--skip-projection", action="store_true", help="Skip UMAP/KMeans.")
    parser.add_argument(
        "--only",
        choices=["decode", "embed", "project"],
        help="Run a single stage against the existing database.",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(settings.db_path)
    init_schema(conn)

    run_decode = args.only in (None, "decode")
    run_embed = args.only in (None, "embed") and not args.skip_embeddings
    run_project = args.only in (None, "project") and not args.skip_projection

    if run_decode:
        shards = download_shards(settings)
        count = decode_and_store(conn, settings, shards, args.limit)
        with transaction(conn):
            set_meta(conn, "dataset", settings.hf_dataset_id)
            set_meta(conn, "n_images", str(count))
            set_meta(conn, "ingested_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        logger.info("Stored %d images", count)

    if run_embed:
        model_name, dim = compute_embeddings(conn, settings)
        with transaction(conn):
            set_meta(conn, "embedding_model", model_name)
            set_meta(conn, "embedding_dim", str(dim))

    if run_project:
        if not settings.embeddings_path.exists():
            logger.warning("No embeddings on disk, skipping the projection stage.")
        else:
            compute_projection(conn, settings)

    summary = {
        "images": conn.execute("SELECT COUNT(*) FROM images").fetchone()[0],
        "captions": conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0],
        "db": str(settings.db_path),
    }
    conn.close()
    logger.info("Ingestion complete: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
