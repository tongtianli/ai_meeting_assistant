from app.core.config import settings
from app.services.embeddings.base import Embedder, EmbeddingError
from app.services.embeddings.mock import MockEmbedder
from app.services.embeddings.openai_compat import OpenAICompatEmbedder


def get_embedder() -> Embedder:
    """按 settings.embedding_provider 构建单一 embedder；缺 key 明确报错，
    绝不静默降级（混用模型会破坏向量空间一致性）。"""
    provider = settings.embedding_provider
    if provider == "mock":
        return MockEmbedder()
    if provider == "glm":
        if not settings.glm_api_key:
            raise EmbeddingError("EMBEDDING_PROVIDER=glm 但 GLM_API_KEY 未配置")
        return OpenAICompatEmbedder(
            "glm", settings.glm_base_url, settings.glm_api_key,
            settings.glm_embedding_model,
        )
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise EmbeddingError("EMBEDDING_PROVIDER=gemini 但 GEMINI_API_KEY 未配置")
        return OpenAICompatEmbedder(
            "gemini", settings.gemini_base_url, settings.gemini_api_key,
            settings.gemini_embedding_model,
        )
    raise EmbeddingError(f"unknown embedding provider: {provider!r}")


__all__ = [
    "Embedder",
    "EmbeddingError",
    "MockEmbedder",
    "OpenAICompatEmbedder",
    "get_embedder",
]
