"""FastAPI application: HTTP routing only, all logic lives in services/repository."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .db import get_meta
from .models import (
    DatasetInfo,
    DatasetStats,
    ImageDetail,
    ImagePage,
    Projection,
    SearchResults,
)
from .services import AppState, DatasetNotIngested

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = get_settings()
    app.state.app_state = AppState(settings)
    logger.info("Serving %d images from %s", app.state.app_state.repo.count(), settings.db_path)
    yield
    app.state.app_state.close()


def get_state(request: Request) -> AppState:
    return request.app.state.app_state


class LazyStaticFiles(StaticFiles):
    """Static files whose directory is allowed to appear after startup.

    Starlette validates the directory on the first request and then keeps raising
    for the rest of the process. The ingest and the frontend build both create
    their directory, and either may run after the server started, so validation is
    skipped: a file that is not there yet is simply a 404, and it starts being
    served as soon as it exists.
    """

    async def check_config(self) -> None:
        return None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Flickr8k Explorer",
        version="0.1.0",
        summary="Local exploration tool for the Flickr8k image-captioning dataset",
        lifespan=lifespan,
    )

    # The Vite dev server runs on a different port, and picks a free one when the
    # default is taken, so any localhost port is allowed. In production the frontend
    # is built and served as static files from the same origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.exception_handler(DatasetNotIngested)
    async def _not_ingested(request: Request, exc: DatasetNotIngested):  # pragma: no cover
        raise HTTPException(status_code=503, detail=str(exc))

    register_routes(app)

    # Mounted unconditionally: testing for the directory here would freeze a missing
    # one into a permanent 404 for the lifetime of the process, even long after the
    # ingest created it.
    app.mount(
        "/media/images",
        LazyStaticFiles(directory=settings.images_dir, check_dir=False),
        name="images",
    )
    app.mount(
        "/media/thumbs",
        LazyStaticFiles(directory=settings.thumbs_dir, check_dir=False),
        name="thumbs",
    )

    # Serve the built frontend, so the whole app is one process.
    dist = settings.data_dir.parent / "frontend" / "dist"
    app.mount(
        "/",
        LazyStaticFiles(directory=dist, html=True, check_dir=False),
        name="frontend",
    )

    return app


def register_routes(app: FastAPI) -> None:
    @app.get("/api/dataset", response_model=DatasetInfo)
    def dataset_info(state: AppState = Depends(get_state)) -> DatasetInfo:
        conn = state.conn
        return DatasetInfo(
            name=get_meta(conn, "dataset", "flickr8k") or "flickr8k",
            n_images=state.repo.count(),
            n_captions=int(
                conn.execute("SELECT COUNT(*) AS n FROM captions").fetchone()["n"]
            ),
            splits=state.repo.splits(),
            embedding_model=get_meta(conn, "embedding_model"),
            embedding_dim=int(get_meta(conn, "embedding_dim") or 0) or None,
            has_projection=get_meta(conn, "has_projection") == "1",
            ingested_at=get_meta(conn, "ingested_at"),
        )

    @app.get("/api/images", response_model=ImagePage)
    def list_images(
        split: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(60, ge=1, le=200),
        state: AppState = Depends(get_state),
    ) -> ImagePage:
        return ImagePage(
            items=state.repo.list_images(split=split, offset=offset, limit=limit),
            total=state.repo.count(split),
            offset=offset,
            limit=limit,
        )

    @app.get("/api/images/{image_id}", response_model=ImageDetail)
    def image_detail(image_id: str, state: AppState = Depends(get_state)) -> ImageDetail:
        detail = state.repo.get_detail(image_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Unknown image {image_id!r}")
        return detail

    @app.get("/api/images/{image_id}/similar", response_model=SearchResults)
    def similar_images(
        image_id: str,
        limit: int = Query(12, ge=1, le=100),
        state: AppState = Depends(get_state),
    ) -> SearchResults:
        items = state.similar_images(image_id, limit=limit)
        if items is None:
            raise HTTPException(status_code=404, detail=f"Unknown image {image_id!r}")
        return SearchResults(items=items, total=len(items), mode="similar", query=image_id)

    @app.get("/api/search", response_model=SearchResults)
    def search(
        q: str = Query(..., min_length=1),
        mode: Literal["semantic", "text"] = "semantic",
        split: str | None = None,
        limit: int = Query(48, ge=1, le=200),
        state: AppState = Depends(get_state),
    ) -> SearchResults:
        if mode == "semantic":
            if not state.semantic_available:
                raise HTTPException(
                    status_code=503,
                    detail="Semantic search unavailable: embeddings were not generated.",
                )
            items = state.semantic_search(q, split=split, limit=limit)
        else:
            items = state.text_search(q, split=split, limit=limit)
        return SearchResults(items=items, total=len(items), mode=mode, query=q)

    @app.get("/api/projection", response_model=Projection)
    def projection(split: str | None = None, state: AppState = Depends(get_state)) -> Projection:
        points = state.repo.projection(split)
        clusters = state.repo.clusters()
        return Projection(points=points, n_clusters=len(clusters), clusters=clusters)

    @app.get("/api/stats", response_model=DatasetStats)
    def stats(state: AppState = Depends(get_state)) -> DatasetStats:
        return state.repo.stats()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}


app = create_app()
