"""Embedding model and vector index.

Two concerns live here, deliberately separated:

* :class:`Embedder` - turns images/text into vectors. CLIP is one implementation;
  swapping in DINOv2, SigLIP or a domain-specific model means writing one class.
* :class:`VectorIndex` - stores the vectors and answers nearest-neighbour queries.

The index is an exhaustive cosine search over a memory-mapped ``float32`` array.
For Flickr8k that is 8k x 512 = 16 MB and a query costs a few milliseconds, so an
ANN library (FAISS, hnswlib) or an external vector database would add operational
weight for no measurable gain. The interface is narrow enough that swapping in an
ANN backend later is a local change, worth doing somewhere north of ~1M vectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

try:  # Pillow is always present, but keep the import local to typing needs.
    from PIL.Image import Image as PILImage
except ImportError:  # pragma: no cover
    PILImage = object  # type: ignore[assignment,misc]


class Embedder(Protocol):
    """Maps images and text into a shared, L2-normalised vector space."""

    name: str
    dim: int

    def embed_images(self, images: Sequence[PILImage]) -> np.ndarray: ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...


class ClipEmbedder:
    """OpenCLIP backed :class:`Embedder`.

    Torch and open_clip are imported lazily so that the API process, which only
    needs the text tower, does not pay the import cost until the first query, and
    so that the test suite can run without the model weights present.
    """

    def __init__(self, model_name: str, pretrained: str, device: str | None = None) -> None:
        import open_clip
        import torch

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = f"open_clip/{model_name}/{pretrained}"

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self.dim = int(model.text_projection.shape[-1])

    def embed_images(self, images: Sequence[PILImage]) -> np.ndarray:
        torch = self._torch
        batch = torch.stack([self._preprocess(img.convert("RGB")) for img in images])
        with torch.no_grad():
            features = self._model.encode_image(batch.to(self.device))
        return _normalize(features.cpu().numpy())

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        tokens = self._tokenizer(list(texts)).to(self.device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
        return _normalize(features.cpu().numpy())


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class VectorIndex:
    """Exhaustive cosine similarity over a matrix of normalised vectors.

    Row *i* of the matrix corresponds to ``images.row_index = i`` in SQLite.
    """

    def __init__(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2D matrix, got shape {vectors.shape}")
        self._vectors = vectors

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        # mmap keeps startup instant and lets the OS page cache do the work.
        return cls(np.load(path, mmap_mode="r"))

    @property
    def shape(self) -> tuple[int, int]:
        return self._vectors.shape

    def __len__(self) -> int:
        return int(self._vectors.shape[0])

    def vector(self, row_index: int) -> np.ndarray:
        return np.asarray(self._vectors[row_index])

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        exclude: int | None = None,
        candidates: np.ndarray | None = None,
    ) -> list[tuple[int, float]]:
        """Return the ``k`` best ``(row_index, score)`` pairs, best first.

        ``candidates`` restricts the search to a subset of row indices, which is
        how metadata filters (e.g. split) compose with semantic search.
        """
        query = _normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]

        if candidates is None:
            scores = self._vectors @ query
            rows = np.arange(scores.shape[0])
        else:
            candidates = np.asarray(candidates, dtype=np.int64)
            if candidates.size == 0:
                return []
            scores = np.asarray(self._vectors[candidates]) @ query
            rows = candidates

        if exclude is not None:
            keep = rows != exclude
            scores, rows = scores[keep], rows[keep]

        if scores.size == 0:
            return []

        k = min(k, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(rows[i]), float(scores[i])) for i in top]
