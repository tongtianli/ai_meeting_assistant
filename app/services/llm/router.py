"""LLM Router：按优先级尝试 provider，失败降级到下一个（PRD §4 统一 service 抽象）。

- 传输错误/限流/非 2xx → 换下一个 provider
- JSON 解析或 schema 校验失败 → 同 provider 带错误反馈重试一次，再失败换下一个
- 每次尝试（成功与失败）写入 llm_usage_records 审计（Tech Design M4 §11）
"""
import json
import logging
import re
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.services.llm.base import LLMError, LLMProvider, LLMResponse, LLMTaskType
from app.services.llm.usage import record_usage

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
    def __init__(
        self,
        providers: list[LLMProvider],
        task: LLMTaskType | None = None,
    ) -> None:
        if not providers:
            raise ValueError(
                "no LLM provider available; set GEMINI_API_KEY/GLM_API_KEY "
                "or LLM_PROVIDERS=mock"
            )
        self.providers = providers
        self.task_type = task.value if task else "unspecified"

    async def _call(
        self, provider: LLMProvider, system: str, user: str, json_mode: bool
    ) -> tuple[LLMResponse, int]:
        started = time.monotonic()
        resp = await provider.complete(system, user, json_mode=json_mode)
        latency_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "llm call ok provider=%s model=%s prompt_tokens=%s completion_tokens=%s latency=%.1fs",
            resp.provider,
            resp.model,
            resp.prompt_tokens,
            resp.completion_tokens,
            latency_ms / 1000,
        )
        return resp, latency_ms

    async def _record(
        self,
        provider: LLMProvider,
        index: int,
        *,
        success: bool,
        resp: LLMResponse | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        latency_ms: int | None = None,
    ) -> None:
        await record_usage(
            task_type=self.task_type,
            provider=provider.name,
            model=resp.model if resp else provider.model,
            success=success,
            fallback_index=index,
            input_tokens=resp.prompt_tokens if resp else None,
            output_tokens=resp.completion_tokens if resp else None,
            error_type=error_type,
            error_message=error_message,
            latency_ms=latency_ms,
        )

    async def _routable_providers(self) -> list[LLMProvider]:
        """额度熔断过滤（Tech Design M4 §12.3）：调用时即时判定。

        过滤后无 provider 可用时抛出带熔断原因的 LLMExhaustedError——
        错误文案与模型故障（transport/schema）可区分，并落一条审计记录。
        """
        from app.services.llm.quota import filter_providers

        providers, note = await filter_providers(self.providers, self.task_type)
        if note:
            logger.warning("%s (task=%s)", note, self.task_type)
        if not providers:
            await record_usage(
                task_type=self.task_type,
                provider="glm_air",
                model="-",
                success=False,
                error_type="quota",
                error_message=note,
            )
            raise LLMExhaustedError(note or "no provider available after quota filter")
        return providers

    async def generate_json(
        self, system: str, user: str, schema: type[T]
    ) -> tuple[T, LLMResponse]:
        """结构化输出：JSON 解析 + pydantic schema 校验，失败自动重试/降级。"""
        errors: list[str] = []
        for index, provider in enumerate(await self._routable_providers()):
            prompt = user
            for attempt in (1, 2):
                started = time.monotonic()
                try:
                    resp, latency_ms = await self._call(
                        provider, system, prompt, json_mode=True
                    )
                except LLMError as exc:
                    logger.warning("llm provider %s failed: %s", provider.name, exc)
                    errors.append(str(exc))
                    await self._record(
                        provider,
                        index,
                        success=False,
                        error_type="transport",
                        error_message=str(exc),
                        latency_ms=int((time.monotonic() - started) * 1000),
                    )
                    break  # 传输层失败：换下一个 provider
                try:
                    parsed = schema.model_validate(extract_json(resp.text))
                    await self._record(
                        provider, index, success=True, resp=resp, latency_ms=latency_ms
                    )
                    return parsed, resp
                except (ValueError, ValidationError) as exc:
                    logger.warning(
                        "llm output failed schema validation (provider=%s attempt=%d): %s",
                        provider.name,
                        attempt,
                        exc,
                    )
                    errors.append(f"{provider.name}: schema validation: {exc}")
                    # 调用本身成功、消耗了 token——记录之，但标记校验失败
                    await self._record(
                        provider,
                        index,
                        success=False,
                        resp=resp,
                        error_type="schema_validation",
                        error_message=str(exc),
                        latency_ms=latency_ms,
                    )
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
        for index, provider in enumerate(await self._routable_providers()):
            started = time.monotonic()
            try:
                resp, latency_ms = await self._call(
                    provider, system, user, json_mode=False
                )
                await self._record(
                    provider, index, success=True, resp=resp, latency_ms=latency_ms
                )
                return resp
            except LLMError as exc:
                logger.warning("llm provider %s failed: %s", provider.name, exc)
                errors.append(str(exc))
                await self._record(
                    provider,
                    index,
                    success=False,
                    error_type="transport",
                    error_message=str(exc),
                    latency_ms=int((time.monotonic() - started) * 1000),
                )
        raise LLMExhaustedError("; ".join(errors[-3:]))
