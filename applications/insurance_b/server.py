"""Serve the native query router with B's frozen natural-answer and audit hook."""

import sys
from contextlib import asynccontextmanager

from .common import workspace_lock
from .config import Settings
from .ingest import accepted_readback
from .models import initialize_rag, make_rag


def native_routes(rag, top_k):
    # The pinned native API parses argv on import. B's subcommands belong to B.
    original = sys.argv
    try:
        sys.argv = [original[0]]
        from lightrag.api.routers.query_routes import create_query_routes
    finally:
        sys.argv = original
    return create_query_routes(rag, top_k=top_k)


def create_app(settings: Settings):
    settings.require_models()
    settings.activate()
    from fastapi import FastAPI

    rag = make_rag(settings)

    @asynccontextmanager
    async def lifespan(app):
        accepted_readback(settings)
        with workspace_lock(settings.root, settings.workspace):
            await initialize_rag(rag, settings)
            try:
                yield
            finally:
                await rag.finalize_storages()

    app = FastAPI(title="Independent insurance B", lifespan=lifespan)
    app.include_router(native_routes(rag, top_k=settings.integer("TOP_K")))

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "workspace": settings.workspace,
            "variant": settings.run_id,
            "engine": "LightRAG 1.5.7",
            "database_ports": [15433, 17687],
        }

    return app


def serve(settings: Settings):
    import uvicorn

    uvicorn.run(
        create_app(settings), host="127.0.0.1", port=settings.integer("HTTP_PORT")
    )
