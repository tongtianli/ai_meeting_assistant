from app.core.config import settings
from app.services.llm.base import LLMError, LLMProvider, LLMResponse
from app.services.llm.mock import MockLLMProvider
from app.services.llm.openai_compat import OpenAICompatProvider
from app.services.llm.router import LLMExhaustedError, LLMRouter


def _build_provider(name: str) -> LLMProvider | None:
    if name == "mock":
        return MockLLMProvider()
    if name == "gemini":
        if not settings.gemini_api_key:
            return None
        return OpenAICompatProvider(
            "gemini", settings.gemini_base_url, settings.gemini_api_key, settings.gemini_model
        )
    if name == "glm":
        if not settings.glm_api_key:
            return None
        return OpenAICompatProvider(
            "glm", settings.glm_base_url, settings.glm_api_key, settings.glm_model
        )
    raise ValueError(f"unknown LLM provider: {name!r}")


def build_router() -> LLMRouter:
    """按 LLM_PROVIDERS 优先级构建 Router；缺 key 的 provider 跳过并告警。"""
    import logging

    providers: list[LLMProvider] = []
    for name in [n.strip() for n in settings.llm_providers.split(",") if n.strip()]:
        provider = _build_provider(name)
        if provider is None:
            logging.getLogger(__name__).warning(
                "LLM provider %r skipped: api key not configured", name
            )
        else:
            providers.append(provider)
    return LLMRouter(providers)


__all__ = [
    "LLMError",
    "LLMExhaustedError",
    "LLMProvider",
    "LLMResponse",
    "LLMRouter",
    "MockLLMProvider",
    "OpenAICompatProvider",
    "build_router",
]
