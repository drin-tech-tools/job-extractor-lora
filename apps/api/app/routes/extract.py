"""Extraction endpoints: one model, or several side by side."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.models.schemas import (
    CompareRequest,
    CompareResponse,
    ExtractRequest,
    ExtractResponse,
    ModelKey,
)
from app.services.extractors import ExtractorUnavailable, get_extractor

router = APIRouter(prefix="/extract", tags=["extract"])


async def _run(model: ModelKey, text: str) -> ExtractResponse:
    """Run one model. Missing config raises; model-side failures come back in the response."""
    extractor = get_extractor(model, get_settings())
    result = await extractor.extract(text)
    return ExtractResponse(
        model=model,
        model_id=result.model_id,
        extraction=result.extraction,
        valid_json=result.valid_json,
        latency_ms=result.latency_ms,
        error=result.error,
        raw=result.raw,
    )


@router.post("", response_model=ExtractResponse)
async def extract(body: ExtractRequest) -> ExtractResponse:
    try:
        return await _run(body.model, body.text)
    except ExtractorUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/compare", response_model=CompareResponse)
async def compare(body: CompareRequest) -> CompareResponse:
    """Run the requested models concurrently; an unconfigured model reports an error row
    instead of failing the whole comparison."""

    async def one(model: ModelKey) -> ExtractResponse:
        try:
            return await _run(model, body.text)
        except ExtractorUnavailable as exc:
            return ExtractResponse(
                model=model, model_id="unavailable", valid_json=False, latency_ms=0, error=str(exc)
            )

    models = list(dict.fromkeys(body.models))  # dedupe, keep order
    return CompareResponse(results=await asyncio.gather(*(one(m) for m in models)))
