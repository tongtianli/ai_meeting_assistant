"""OpenAI 兼容 embedding provider：Gemini 与 GLM 都暴露 /embeddings 端点，
一份实现 + 两组配置覆盖两家（base_url / api_key / model）。
"""
import time

import httpx

from app.core.config import settings
from app.services.embeddings.base import Embedder, EmbeddingError
from app.services.llm.usage import record_usage

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class OpenAICompatEmbedder(Embedder):
    def __init__(self, name: str, base_url: str, api_key: str, model: str) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def _record(
        self,
        *,
        success: bool,
        started: float,
        input_tokens: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        # embedding 用量单独记账（Tech Design M4 §6.6），与 LLM 调用区分统计
        await record_usage(
            task_type="embedding",
            provider=self.name,
            model=self.model,
            success=success,
            input_tokens=input_tokens,
            output_tokens=0 if success else None,
            error_type=error_type,
            error_message=error_message,
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        batch = max(settings.embedding_batch_size, 1)
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for i in range(0, len(texts), batch):
                chunk = texts[i : i + batch]
                started = time.monotonic()
                try:
                    resp = await client.post(
                        f"{self.base_url}/embeddings",
                        json={"model": self.model, "input": chunk},
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                except httpx.HTTPError as exc:
                    await self._record(
                        success=False,
                        started=started,
                        error_type="transport",
                        error_message=str(exc),
                    )
                    raise EmbeddingError(f"{self.name}: transport error: {exc}") from exc
                if resp.status_code != 200:
                    await self._record(
                        success=False,
                        started=started,
                        error_type="transport",
                        error_message=f"HTTP {resp.status_code}: {resp.text[:300]}",
                    )
                    raise EmbeddingError(
                        f"{self.name}: HTTP {resp.status_code}: {resp.text[:300]}"
                    )
                data = resp.json()
                try:
                    # data 按 index 排序返回；显式按 index 复原顺序以防乱序
                    items = sorted(data["data"], key=lambda d: d["index"])
                    out.extend(item["embedding"] for item in items)
                except (KeyError, TypeError) as exc:
                    await self._record(
                        success=False,
                        started=started,
                        error_type="schema_validation",
                        error_message=f"malformed response: {str(data)[:300]}",
                    )
                    raise EmbeddingError(
                        f"{self.name}: malformed response: {str(data)[:300]}"
                    ) from exc
                usage = data.get("usage") or {}
                tokens = usage.get("prompt_tokens", usage.get("total_tokens"))
                await self._record(
                    success=True,
                    started=started,
                    input_tokens=tokens if isinstance(tokens, int) else None,
                )
        return out
