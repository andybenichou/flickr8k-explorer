"""End-to-end tests over the HTTP surface, against the miniature fixture dataset."""

from __future__ import annotations


def test_dataset_info(client):
    body = client.get("/api/dataset").json()
    assert body["n_images"] == 4
    assert body["n_captions"] == 8
    assert body["splits"] == ["test", "train"]
    assert body["has_projection"] is True


def test_list_images_paginates_and_filters(client):
    page = client.get("/api/images", params={"limit": 2}).json()
    assert len(page["items"]) == 2
    assert page["total"] == 4

    second = client.get("/api/images", params={"limit": 2, "offset": 2}).json()
    assert {i["id"] for i in page["items"]} & {i["id"] for i in second["items"]} == set()

    train = client.get("/api/images", params={"split": "train"}).json()
    assert train["total"] == 2
    assert all(item["split"] == "train" for item in train["items"])


def test_image_detail(client):
    detail = client.get("/api/images/train-00000").json()
    assert len(detail["captions"]) == 2
    assert detail["captions"][0]["index"] == 0
    assert detail["aspect_ratio"] == 1.25
    assert detail["umap"] is not None


def test_unknown_image_returns_404(client):
    assert client.get("/api/images/does-not-exist").status_code == 404


def test_semantic_search_ranks_the_matching_image_first(client):
    body = client.get("/api/search", params={"q": "a cat sleeping"}).json()
    assert body["mode"] == "semantic"
    assert body["items"][0]["id"] == "test-00003"
    assert body["items"][0]["score"] == 1.0


def test_semantic_search_respects_the_split_filter(client):
    body = client.get("/api/search", params={"q": "a cat sleeping", "split": "train"}).json()
    assert all(item["split"] == "train" for item in body["items"])


def test_text_search_matches_caption_words(client):
    body = client.get("/api/search", params={"q": "mountain", "mode": "text"}).json()
    assert [item["id"] for item in body["items"]] == ["test-00002"]


def test_text_search_survives_fts_syntax_characters(client):
    # A raw `"` or `*` is a syntax error if passed straight to FTS5.
    quoted = client.get("/api/search", params={"q": '"dog"', "mode": "text"})
    assert quoted.status_code == 200
    assert [item["id"] for item in quoted.json()["items"]] == ["train-00000"]

    for hostile in ('dog" OR *', "NEAR(", "*", '""'):
        assert client.get("/api/search", params={"q": hostile, "mode": "text"}).status_code == 200


def test_similar_excludes_the_query_image(client):
    body = client.get("/api/images/train-00000/similar", params={"limit": 3}).json()
    ids = [item["id"] for item in body["items"]]
    assert "train-00000" not in ids
    assert len(ids) == 3


def test_projection(client):
    body = client.get("/api/projection").json()
    assert len(body["points"]) == 4
    assert body["n_clusters"] == 2
    assert all(0.0 <= point["x"] <= 1.0 for point in body["points"])


def test_stats(client):
    body = client.get("/api/stats").json()
    assert body["n_images"] == 4
    assert sum(bucket["count"] for bucket in body["caption_length_histogram"]) == 8
    labels = [bucket["label"] for bucket in body["top_words"]]
    assert "the" not in labels  # stopwords are filtered
    assert body["resolution_histogram"][0]["label"] == "100x80"


def test_media_is_served(client):
    response = client.get("/media/thumbs/train-00000.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
