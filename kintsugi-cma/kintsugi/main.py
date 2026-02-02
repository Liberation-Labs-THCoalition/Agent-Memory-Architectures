"""Kintsugi FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from kintsugi import __version__
from kintsugi.config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from kintsugi.db import engine
    # Engine is created at import time; just verify connectivity at startup
    async with engine.connect() as conn:
        await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    yield
    await engine.dispose()


app = FastAPI(
    title="Kintsugi Engine",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Route registration (graceful if modules missing) ---
_route_modules = [
    "kintsugi.api.routes.health",
    "kintsugi.api.routes.agent",
    "kintsugi.api.routes.memory",
    "kintsugi.api.routes.config",
]

for _mod_path in _route_modules:
    try:
        import importlib
        _mod = importlib.import_module(_mod_path)
        app.include_router(_mod.router)
    except (ImportError, AttributeError):
        pass


# --- Exception handlers ---

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(PermissionError)
async def permission_error_handler(request: Request, exc: PermissionError) -> JSONResponse:
    return JSONResponse(status_code=403, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
