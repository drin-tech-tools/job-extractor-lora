"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Ordered fallback when GEMINI_MODEL is unset.
# 2.5 models are unavailable to many new API keys — skip them.
GEMINI_MODEL_FALLBACK: list[str] = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-3.8-flash",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_model: str | None = None

    # Fine-tuned and base models are each reached through an OpenAI-compatible
    # /chat/completions endpoint (URL includes the /v1 prefix). One shared HF token.
    finetuned_api_url: str = ""
    finetuned_model: str = ""
    base_api_url: str = ""
    base_model: str = ""
    hf_token: str = ""

    # CORS is unused in production (API is internal-only behind the Next.js BFF).
    # Kept for optional local direct browser→API access (e.g. hitting /docs on :8000).
    allowed_origins: str = "http://localhost:3000"

    # Optional shared secret. When set, every request must include matching
    # X-Internal-Api-Token (the web BFF sends it from INTERNAL_API_TOKEN).
    internal_api_token: str = ""

    # When truthy (EXTRACTOR_MOCK=1), every model key is served by a deterministic
    # heuristic extractor — no Gemini, no model endpoints. For demos and CI only.
    extractor_mock: bool = False

    # Cold-start endpoints can take a while to wake; keep this above their warm-up time.
    request_timeout_s: float = 60.0

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def gemini_models(self) -> list[str]:
        if self.gemini_model:
            pinned = self.gemini_model
            rest = [m for m in GEMINI_MODEL_FALLBACK if m != pinned]
            return [pinned, *rest]
        return list(GEMINI_MODEL_FALLBACK)

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def finetuned_configured(self) -> bool:
        return bool(self.finetuned_api_url and self.finetuned_model)

    @property
    def base_configured(self) -> bool:
        return bool(self.base_api_url and self.base_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()
