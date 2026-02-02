"""Health check endpoint."""

from fastapi import APIRouter

from kintsugi import __version__
from kintsugi.config.settings import settings

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "tier": settings.DEPLOYMENT_TIER,
        "version": __version__,
    }
