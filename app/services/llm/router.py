"""LLM Router：按优先级尝试 provider，失败降级到下一个（PRD §4 统一 service 抽象）。

- 传输错误/限流/非 2xx → 换下一个 provider
- JSON 解析或 schema 校验失败 → 同 provider 带错误反馈重试一次，再失败换下一个
- 每次调用记录 tokens 用量与耗时（成本记录）
"""
import json
import logging
import re
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.services.llm.base import LLMError, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class LLMExhaustedError(RuntimeError):
    """全部 provider 均失败。"""


def extract_json(text: str) -> dict:
    """容错解析：剥掉可能的 markdown 代码栅栏后解析 JSON 对象。"""
    candidate = text.strip()
    m = _JSON_BLOCK.search(candidate)
    if m:
        candidate = m.group(1)
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object, got {type(data).__name__}")
    return data


class LLMRouter:
    def __init__(self, providers: list[LLMProvider]) -> None:
        if not providers:
            raise ValueError(
                "no LLM provider available; set GEMINI_API_KEY/GLM_API_KEY "
                "or LLM_PROVIDERS=mock"
            )
        self.providers = providers

    async def _call(self, provider: LLMProvider, system: str, user: str, json_mode: bool) -> LLMResponse:
        started = time.monotonic()
        resp = await provider.complete(system, user, json_mode=json_mode)
        logger.info(
            "llm call ok provider=%s model=%s prompt_tokens=%s completion_tokens=%s latency=%.1fs",
            resp.provider,
            resp.model,
            resp.prompt_tokens,
            resp.completion_tokens,
            time.monotonic() - started,
        )
        return resp

    async def generate_json(
        self, system: str, user: str, schema: type[T]
    ) -> tuple[T, LLMResponse]:
        """结构化输出：JSON 解析 + pydantic schema 校验，失败自动重试/降级。"""
        errors: list[str] = []
        for provider in self.providers:
            prompt = user
            for attempt in (1, 2):
                try:
                    resp = await self._call(provider, system, prompt, json_mode=True)
                except LLMError as exc:
                    logger.warning("llm provider %s failed: %s", provider.name, exc)
                    errors.append(str(exc))
                    break  # 传输层失败：换下一个 provider
                try:
                    parsed = schema.model_validate(extract_json(resp.text))
                    return parsed, resp
                except (ValueError, ValidationError) as exc:
                    logger.warning(
                        "llm output failed schema validation (provider=%s attempt=%d): %s",
                        provider.name,
                        attempt,
                        exc,
                    )
                    errors.append(f"{provider.name}: schema validation: {exc}")
                    # 带错误反馈重试同一 provider 一次
                    prompt = (
                        f"{user}\n\n"
                        f"你上一次的输出未通过校验，错误信息：{exc}\n"
                        "请严格按要求重新输出合法的 JSON 对象，不要输出任何其他内容。"
                    )
        raise LLMExhaustedError("; ".join(errors[-3:]))

    async def generate_text(self, system: str, user: str) -> LLMResponse:
        """纯文本输出（结构化失败后的降级路径）。"""
        errors: list[str] = []
        for provider in self.providers:
            try:
                return await self._call(provider, system, user, json_mode=False)
            except LLMError as exc:
                logger.warning("llm provider %s failed: %s", provider.name, exc)
                errors.append(str(exc))
        raise LLMExhaustedError("; ".join(errors[-3:]))
