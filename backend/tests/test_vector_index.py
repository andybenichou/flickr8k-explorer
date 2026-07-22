"""Unit tests for the vector index, the one place with non-trivial numeric logic."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.embeddings import VectorIndex


@pytest.fixture
def index() -> VectorIndex:
    return VectorIndex(np.eye(5, dtype=np.float32))


def test_search_returns_scores_in_descending_order(index):
    query = np.array([0.9, 0.4, 0.1, 0.0, 0.0], dtype=np.float32)
    results = index.search(query, k=3)
    assert [row for row, _ in results] == [0, 1, 2]
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_search_normalises_an_unnormalised_query(index):
    unit = index.search(np.array([1.0, 0, 0, 0, 0], dtype=np.float32), k=1)
    scaled = index.search(np.array([17.0, 0, 0, 0, 0], dtype=np.float32), k=1)
    assert unit == scaled
    assert scaled[0][1] == pytest.approx(1.0)


def test_exclude_drops_the_query_row(index):
    results = index.search(index.vector(2), k=4, exclude=2)
    assert 2 not in [row for row, _ in results]


def test_candidates_restrict_the_search_space(index):
    results = index.search(index.vector(0), k=5, candidates=np.array([3, 4]))
    assert sorted(row for row, _ in results) == [3, 4]


def test_empty_candidate_set_returns_nothing(index):
    assert index.search(index.vector(0), k=5, candidates=np.array([], dtype=np.int64)) == []


def test_k_larger_than_the_index_is_clamped(index):
    assert len(index.search(index.vector(0), k=99)) == 5


def test_rejects_a_non_matrix():
    with pytest.raises(ValueError):
        VectorIndex(np.zeros(5, dtype=np.float32))
