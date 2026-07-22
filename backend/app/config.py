"""Central configuration. Every path used by the app is derived from here."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Settings are overridable through environment variables (prefix ``F8K_``)."""

    model_config = SettingsConfigDict(env_prefix="F8K_", env_file=".env")

    # --- storage -------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"

    # --- dataset -------------------------------------------------------
    hf_dataset_id: str = "jxie/flickr8k"
    thumbnail_size: int = 320

    # --- embedding model -----------------------------------------------
    # ViT-B-32 is the smallest CLIP that still gives usable retrieval quality
    # and it runs on a laptop CPU in a few minutes for 8k images.
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    embed_batch_size: int = 32

    # --- 2D projection ---------------------------------------------------
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1
    n_clusters: int = 12

    # --- api -------------------------------------------------------------
    page_size_max: int = 200

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "flickr8k.db"

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / "embeddings.npy"


@lru_cache
def get_settings() -> Settings:
    return Settings()
