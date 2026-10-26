"""Liveness/readiness plus which model backends this deployment can reach."""

from fastapi import APIRouter

from app import __version__
from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    mock = settings.extractor_mock
    return {
        "status": "ok",
        "version": __version__,
        "mock": mock,
        "backends": {
            "finetuned": mock or settings.finetuned_configured,
            "base": mock or settings.base_configured,
            "gemini": mock or settings.gemini_configured,
        },
    }
