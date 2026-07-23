"""Data access: every SQL statement in the application lives here.

Keeping SQL out of the route handlers means the storage engine can change (or a
second dataset can be plugged in) without touching the HTTP layer.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter

import numpy as np

from .models import (
    Bucket,
    Caption,
    ClusterInfo,
    DatasetStats,
    ImageDetail,
    ImageSummary,
    ProjectionPoint,
)

# Small, explicit stopword list: enough to make the "top words" panel informative
# without pulling NLTK in for one feature.
STOPWORDS = frozenset(
    """
    a an the and or but of to in on at for with from by as is are was were be been being
    it its this that these those there here he she they them his her their you your i we
    my me our us who whom which what while into over under near up down out off then than
    two three some other another very
    """.split()
)

WORD_RE = re.compile(r"[a-z']+")

CAPTION_LENGTH_BINS = [(0, 8), (8, 11), (11, 14), (14, 17), (17, 21), (21, 10_000)]


class ImageRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._stats: DatasetStats | None = None

    # --- listing -------------------------------------------------------

    def count(self, split: str | None = None) -> int:
        if split:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM images WHERE split = ?", (split,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()
        return int(row["n"])

    def list_images(self, *, split: str | None, offset: int, limit: int) -> list[ImageSummary]:
        sql = """
            SELECT i.id, i.split, i.width, i.height, c.text AS caption
            FROM images i
            JOIN captions c ON c.image_id = i.id AND c.caption_index = 0
            {where}
            ORDER BY i.row_index
            LIMIT ? OFFSET ?
        """
        params: list[object] = []
        where = ""
        if split:
            where = "WHERE i.split = ?"
            params.append(split)
        params += [limit, offset]
        rows = self.conn.execute(sql.format(where=where), params).fetchall()
        return [_to_summary(row) for row in rows]

    def get_detail(self, image_id: str) -> ImageDetail | None:
        row = self.conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        if row is None:
            return None
        captions = [
            Caption(index=c["caption_index"], text=c["text"], n_words=c["n_words"])
            for c in self.conn.execute(
                "SELECT caption_index, text, n_words FROM captions "
                "WHERE image_id = ? ORDER BY caption_index",
                (image_id,),
            )
        ]
        umap = (
            (row["umap_x"], row["umap_y"])
            if row["umap_x"] is not None and row["umap_y"] is not None
            else None
        )
        return ImageDetail(
            id=row["id"],
            split=row["split"],
            width=row["width"],
            height=row["height"],
            file_size=row["file_size"],
            aspect_ratio=round(row["width"] / row["height"], 3),
            image_url=f"/media/images/{row['id']}.jpg",
            thumb_url=f"/media/thumbs/{row['id']}.jpg",
            captions=captions,
            umap=umap,
            cluster=row["cluster"],
        )

    # --- vector-index bridging ------------------------------------------

    def row_index_of(self, image_id: str) -> int | None:
        row = self.conn.execute(
            "SELECT row_index FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        return int(row["row_index"]) if row else None

    def row_indices_for_split(self, split: str) -> np.ndarray:
        rows = self.conn.execute(
            "SELECT row_index FROM images WHERE split = ? ORDER BY row_index", (split,)
        ).fetchall()
        return np.array([r["row_index"] for r in rows], dtype=np.int64)

    def summaries_by_row_index(self, ranked: list[tuple[int, float]]) -> list[ImageSummary]:
        """Resolve ranked ``(row_index, score)`` pairs, preserving the ranking."""
        if not ranked:
            return []
        placeholders = ",".join("?" * len(ranked))
        rows = self.conn.execute(
            f"""
            SELECT i.row_index, i.id, i.split, i.width, i.height, c.text AS caption
            FROM images i
            JOIN captions c ON c.image_id = i.id AND c.caption_index = 0
            WHERE i.row_index IN ({placeholders})
            """,
            [r for r, _ in ranked],
        ).fetchall()
        by_row = {row["row_index"]: row for row in rows}
        out = []
        for row_index, score in ranked:
            row = by_row.get(row_index)
            if row is not None:
                out.append(_to_summary(row, score=score))
        return out

    # --- full-text search -------------------------------------------------

    def search_captions(self, query: str, *, split: str | None, limit: int) -> list[ImageSummary]:
        """Rank images by BM25 over their captions (best caption wins)."""
        # FTS5 exposes the BM25 score through the special `rank` column. Calling
        # bm25() explicitly is rejected once the query aggregates or joins, so the
        # CTE ranks first (lower is better) and the joins happen outside it.
        sql = """
            WITH hits AS (
                SELECT image_id, MIN(rank) AS rank
                FROM captions_fts
                WHERE captions_fts MATCH ?
                GROUP BY image_id
            )
            SELECT i.id, i.split, i.width, i.height, c.text AS caption, hits.rank
            FROM hits
            JOIN images i ON i.id = hits.image_id
            JOIN captions c ON c.image_id = i.id AND c.caption_index = 0
            {split_filter}
            ORDER BY hits.rank
            LIMIT ?
        """
        params: list[object] = [query]
        split_filter = ""
        if split:
            split_filter = "WHERE i.split = ?"
            params.append(split)
        params.append(limit)
        rows = self.conn.execute(sql.format(split_filter=split_filter), params).fetchall()
        # bm25 returns lower-is-better; expose it as a descending 0..1-ish score.
        return [_to_summary(row, score=round(1.0 / (1.0 + abs(row["rank"])), 4)) for row in rows]

    # --- projection --------------------------------------------------------

    def projection(self, split: str | None = None) -> list[ProjectionPoint]:
        sql = """
            SELECT id, umap_x, umap_y, cluster, split FROM images
            WHERE umap_x IS NOT NULL {split_filter}
            ORDER BY row_index
        """
        params: list[object] = []
        split_filter = ""
        if split:
            split_filter = "AND split = ?"
            params.append(split)
        rows = self.conn.execute(sql.format(split_filter=split_filter), params).fetchall()
        return [
            ProjectionPoint(
                id=r["id"], x=r["umap_x"], y=r["umap_y"], cluster=r["cluster"], split=r["split"]
            )
            for r in rows
        ]

    def clusters(self) -> list[ClusterInfo]:
        return [
            ClusterInfo(id=r["id"], label=r["label"], size=r["size"])
            for r in self.conn.execute(
                "SELECT id, label, size FROM clusters ORDER BY size DESC"
            )
        ]

    def splits(self) -> list[str]:
        return [
            r["split"]
            for r in self.conn.execute("SELECT DISTINCT split FROM images ORDER BY split")
        ]

    # --- stats -------------------------------------------------------------

    def stats(self) -> DatasetStats:
        # The dataset is immutable once ingested, so compute this once per process.
        if self._stats is None:
            self._stats = _compute_stats(self.conn)
        return self._stats


def _to_summary(row: sqlite3.Row, score: float | None = None) -> ImageSummary:
    return ImageSummary(
        id=row["id"],
        split=row["split"],
        width=row["width"],
        height=row["height"],
        thumb_url=f"/media/thumbs/{row['id']}.jpg",
        caption=row["caption"],
        score=score,
    )


def _compute_stats(conn: sqlite3.Connection) -> DatasetStats:
    n_images = int(conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"])
    n_captions = int(conn.execute("SELECT COUNT(*) AS n FROM captions").fetchone()["n"])

    images_per_split = [
        Bucket(label=r["split"], count=r["n"])
        for r in conn.execute(
            "SELECT split, COUNT(*) AS n FROM images GROUP BY split ORDER BY split"
        )
    ]

    lengths = [r["n_words"] for r in conn.execute("SELECT n_words FROM captions")]
    caption_hist = []
    for low, high in CAPTION_LENGTH_BINS:
        count = sum(1 for n in lengths if low <= n < high)
        label = f"{low}-{high - 1}" if high < 10_000 else f"{low}+"
        caption_hist.append(Bucket(label=label, count=count))

    resolution_hist = [
        Bucket(label=f"{r['width']}x{r['height']}", count=r["n"])
        for r in conn.execute(
            "SELECT width, height, COUNT(*) AS n FROM images "
            "GROUP BY width, height ORDER BY n DESC LIMIT 10"
        )
    ]

    counter: Counter[str] = Counter()
    for (text,) in conn.execute("SELECT text FROM captions"):
        counter.update(w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 2)
    top_words = [Bucket(label=w, count=c) for w, c in counter.most_common(25)]

    return DatasetStats(
        n_images=n_images,
        n_captions=n_captions,
        images_per_split=images_per_split,
        caption_length_histogram=caption_hist,
        resolution_histogram=resolution_hist,
        top_words=top_words,
        mean_caption_words=round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        vocabulary_size=len(counter),
    )
