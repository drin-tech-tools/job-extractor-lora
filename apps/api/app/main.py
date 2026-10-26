"""FastAPI application entry."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.config import get_settings
from app.routes import extract, health

settings = get_settings()

app = FastAPI(
    title="Job Extractor LoRA API",
    version=__version__,
    description="Job posting -> structured JSON: fine-tuned small model vs base vs Gemini",
)

# Harmless for local direct access; production traffic is internal-only via BFF.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def enforce_internal_api_token(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Defense in depth: when INTERNAL_API_TOKEN is set, require matching header."""
    expected = get_settings().internal_api_token
    if expected:
        got = request.headers.get("x-internal-api-token") or ""
        if got != expected:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


app.include_router(health.router)
app.include_router(extract.router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "job-extractor-lora-api",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }
