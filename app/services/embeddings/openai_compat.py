"""OpenAI 兼容 embedding provider：Gemini 与 GLM 都暴露 /embeddings 端点，
一份实现 + 两组配置覆盖两家（base_url / api_key / model）。
"""
import httpx

from app.core.config import settings
from app.services.embeddings.base import Embedder, EmbeddingError

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class OpenAICompatEmbedder(Embedder):
    def __init__(self, name: str, base_url: str, api_key: str, model: str) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        batch = max(settings.embedding_batch_size, 1)
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for i in range(0, len(texts), batch):
                chunk = texts[i : i + batch]
                try:
                    resp = await client.post(
                        f"{self.base_url}/embeddings",
                        json={"model": self.model, "input": chunk},
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                except httpx.HTTPError as exc:
                    raise EmbeddingError(f"{self.name}: transport error: {exc}") from exc
                if resp.status_code != 200:
                    raise EmbeddingError(
                        f"{self.name}: HTTP {resp.status_code}: {resp.text[:300]}"
                    )
                data = resp.json()
                try:
                    # data 按 index 排序返回；显式按 index 复原顺序以防乱序
                    items = sorted(data["data"], key=lambda d: d["index"])
                    out.extend(item["embedding"] for item in items)
                except (KeyError, TypeError) as exc:
                    raise EmbeddingError(
                        f"{self.name}: malformed response: {str(data)[:300]}"
                    ) from exc
        return out
