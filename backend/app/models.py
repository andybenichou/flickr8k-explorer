"""API response schemas. These are the contract the frontend codes against."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Caption(BaseModel):
    index: int
    text: str
    n_words: int


class ImageSummary(BaseModel):
    """Everything the grid needs, and nothing more."""

    id: str
    split: str
    width: int
    height: int
    thumb_url: str
    caption: str = Field(description="First caption, used as the grid label")
    score: float | None = Field(default=None, description="Similarity score, when ranked")


class ImageDetail(BaseModel):
    id: str
    split: str
    width: int
    height: int
    file_size: int
    aspect_ratio: float
    image_url: str
    thumb_url: str
    captions: list[Caption]
    umap: tuple[float, float] | None = None
    cluster: int | None = None


class ImagePage(BaseModel):
    items: list[ImageSummary]
    total: int
    offset: int
    limit: int


class SearchResults(BaseModel):
    items: list[ImageSummary]
    total: int
    mode: str
    query: str


class ProjectionPoint(BaseModel):
    id: str
    x: float
    y: float
    cluster: int | None = None
    split: str


class Projection(BaseModel):
    points: list[ProjectionPoint]
    n_clusters: int


class Bucket(BaseModel):
    label: str
    count: int


class DatasetStats(BaseModel):
    n_images: int
    n_captions: int
    images_per_split: list[Bucket]
    caption_length_histogram: list[Bucket]
    resolution_histogram: list[Bucket]
    top_words: list[Bucket]
    mean_caption_words: float
    vocabulary_size: int


class DatasetInfo(BaseModel):
    name: str
    n_images: int
    n_captions: int
    splits: list[str]
    embedding_model: str | None
    embedding_dim: int | None
    has_projection: bool
    ingested_at: str | None
