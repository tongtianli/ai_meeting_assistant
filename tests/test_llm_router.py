"""LLM Router：降级、校验重试、成本记录与容错解析。"""
import asyncio

import pytest
from pydantic import BaseModel

from app.services.llm.base import LLMError, LLMProvider, LLMResponse
from app.services.llm.router import LLMExhaustedError, LLMRouter, extract_json


class _Out(BaseModel):
    answer: str


class _FakeProvider(LLMProvider):
    """按脚本依次返回结果或抛错的假 provider。"""

    def __init__(self, name: str, script: list) -> None:
        self.name = name
        self.model = f"{name}-model"
        self.script = list(script)
        self.calls = 0

    async def complete(self, system, user, json_mode=True, temperature=0.2):
        self.calls += 1
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return LLMResponse(text=action, provider=self.name, model=self.model)


def test_extract_json_strips_code_fence() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('{"a": 1}') == {"a": 1}
    with pytest.raises(Exception):
        extract_json("[1, 2]")  # 非对象


def test_router_falls_back_on_transport_error() -> None:
    primary = _FakeProvider("gemini", [LLMError("boom")])
    fallback = _FakeProvider("glm", ['{"answer": "ok"}'])
    router = LLMRouter([primary, fallback])

    parsed, resp = asyncio.run(router.generate_json("s", "u", _Out))
    assert parsed.answer == "ok"
    assert resp.provider == "glm"
    assert primary.calls == 1 and fallback.calls == 1


def test_router_retries_same_provider_on_invalid_json() -> None:
    provider = _FakeProvider("gemini", ["not json at all", '{"answer": "second try"}'])
    router = LLMRouter([provider])

    parsed, _ = asyncio.run(router.generate_json("s", "u", _Out))
    assert parsed.answer == "second try"
    assert provider.calls == 2


def test_router_schema_failure_then_fallback() -> None:
    # 主 provider 两次都输出不符合 schema 的 JSON → 落到备用
    primary = _FakeProvider("gemini", ['{"wrong": 1}', '{"wrong": 2}'])
    fallback = _FakeProvider("glm", ['{"answer": "rescued"}'])
    router = LLMRouter([primary, fallback])

    parsed, resp = asyncio.run(router.generate_json("s", "u", _Out))
    assert parsed.answer == "rescued"
    assert primary.calls == 2


def test_router_exhausted_raises() -> None:
    router = LLMRouter([_FakeProvider("gemini", [LLMError("a")])])
    with pytest.raises(LLMExhaustedError):
        asyncio.run(router.generate_json("s", "u", _Out))


def test_router_requires_providers() -> None:
    with pytest.raises(ValueError, match="no LLM provider"):
        LLMRouter([])
