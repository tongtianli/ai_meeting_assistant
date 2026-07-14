"""任务级 LLM 路由（Tech Design M4 Phase 1）：
glm_air/glm_flash 多模型实例、按任务选择 provider 顺序、fallback 语义。"""
import asyncio

import pytest

from app.core.config import settings
from app.services.llm import build_router, provider_order_for
from app.services.llm.base import LLMError, LLMProvider, LLMResponse, LLMTaskType
from app.services.llm.router import LLMRouter
from pydantic import BaseModel


class _Out(BaseModel):
    answer: str


class _FakeProvider(LLMProvider):
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


@pytest.fixture
def _cloud_keys(monkeypatch):
    """配好云端 key，让 glm_air/glm_flash/gemini 三个节点都可构建。"""
    monkeypatch.setattr(settings, "llm_routing_mode", "task_based")
    monkeypatch.setattr(settings, "glm_api_key", "glm-test-key")
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test-key")


# ---------- 任务 → provider 顺序 ----------

def test_task_provider_orders_default(_cloud_keys) -> None:
    """默认路由与设计文档 §19 一致：高价值任务 Air 优先，轻量任务 Flash 优先。"""
    assert provider_order_for(LLMTaskType.SUMMARY_FINAL) == "glm_air,glm_flash,gemini"
    assert provider_order_for(LLMTaskType.SUMMARY_EDIT) == "glm_air,glm_flash,gemini"
    assert provider_order_for(LLMTaskType.SUMMARY_MAP) == "glm_flash,glm_air,gemini"
    assert provider_order_for(LLMTaskType.QA_ANSWER) == "glm_flash,gemini,glm_air"
    assert provider_order_for(LLMTaskType.QUERY_REWRITE) == "glm_flash,gemini"
    assert provider_order_for(LLMTaskType.INTENT_CLASSIFY) == "glm_flash,gemini"


def test_summary_final_router_prefers_air(_cloud_keys) -> None:
    router = build_router(LLMTaskType.SUMMARY_FINAL)
    assert [p.name for p in router.providers] == ["glm_air", "glm_flash", "gemini"]


def test_qa_router_does_not_lead_with_air(_cloud_keys) -> None:
    """QA/改写/分类默认不优先消耗 Air 赠送额度。"""
    for task in (
        LLMTaskType.QA_ANSWER,
        LLMTaskType.QUERY_REWRITE,
        LLMTaskType.INTENT_CLASSIFY,
    ):
        names = [p.name for p in build_router(task).providers]
        assert names[0] == "glm_flash", task
    # 改写/分类的路由里完全没有 air
    assert "glm_air" not in [
        p.name for p in build_router(LLMTaskType.INTENT_CLASSIFY).providers
    ]


def test_air_and_flash_are_distinct_nodes(_cloud_keys) -> None:
    """同一 GLM key 下 air/flash 是独立节点：不同 name、不同模型。"""
    router = build_router(LLMTaskType.SUMMARY_FINAL)
    air, flash = router.providers[0], router.providers[1]
    assert air.model == settings.glm_air_model
    assert flash.model == settings.glm_flash_model
    assert air.model != flash.model
    assert air.api_key == flash.api_key == "glm-test-key"


# ---------- 缺 key / 回退 ----------

def test_missing_glm_key_skips_glm_nodes(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_routing_mode", "task_based")
    monkeypatch.setattr(settings, "glm_api_key", "")
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test-key")
    router = build_router(LLMTaskType.SUMMARY_FINAL)
    assert [p.name for p in router.providers] == ["gemini"]


def test_task_route_empty_falls_back_to_llm_providers(monkeypatch) -> None:
    """任务路由全不可用（无云端 key）→ 回退 LLM_PROVIDERS（mock 联调场景）。"""
    monkeypatch.setattr(settings, "llm_routing_mode", "task_based")
    monkeypatch.setattr(settings, "glm_api_key", "")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "llm_providers", "mock")
    router = build_router(LLMTaskType.SUMMARY_FINAL)
    assert [p.name for p in router.providers] == ["mock"]


def test_legacy_mode_ignores_task_routes(monkeypatch) -> None:
    """回滚开关：LLM_ROUTING_MODE=legacy 时所有任务共用 LLM_PROVIDERS。"""
    monkeypatch.setattr(settings, "llm_routing_mode", "legacy")
    monkeypatch.setattr(settings, "glm_api_key", "glm-test-key")
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test-key")
    monkeypatch.setattr(settings, "llm_providers", "gemini,glm")
    for task in LLMTaskType:
        assert [p.name for p in build_router(task).providers] == ["gemini", "glm"]
    # legacy 的 glm 节点用 GLM_MODEL
    glm = build_router(LLMTaskType.QA_ANSWER).providers[1]
    assert glm.model == settings.glm_model


def test_no_task_uses_llm_providers(monkeypatch) -> None:
    """未指定任务的调用（尚未迁移的场景）沿用 LLM_PROVIDERS。"""
    monkeypatch.setattr(settings, "llm_routing_mode", "task_based")
    monkeypatch.setattr(settings, "llm_providers", "mock")
    assert [p.name for p in build_router().providers] == ["mock"]


def test_unknown_provider_name_raises(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_routing_mode", "task_based")
    monkeypatch.setattr(settings, "summary_final_providers", "glm_ultra")
    with pytest.raises(ValueError, match="unknown LLM provider"):
        build_router(LLMTaskType.SUMMARY_FINAL)


def test_invalid_routing_mode_fails_at_startup() -> None:
    """LLM_ROUTING_MODE 拼写错误应在配置加载时报错，而非静默落入 legacy。"""
    from pydantic import ValidationError

    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(llm_routing_mode="task-base")


# ---------- fallback 链语义（Air → Flash → Gemini） ----------

def test_air_success_does_not_touch_flash_or_gemini() -> None:
    air = _FakeProvider("glm_air", ['{"answer": "from-air"}'])
    flash = _FakeProvider("glm_flash", [])
    gemini = _FakeProvider("gemini", [])
    router = LLMRouter([air, flash, gemini])

    parsed, resp = asyncio.run(router.generate_json("s", "u", _Out))
    assert parsed.answer == "from-air"
    assert resp.provider == "glm_air"
    assert flash.calls == 0 and gemini.calls == 0


def test_air_429_falls_back_to_flash() -> None:
    air = _FakeProvider("glm_air", [LLMError("glm_air: HTTP 429: rate limited")])
    flash = _FakeProvider("glm_flash", ['{"answer": "from-flash"}'])
    router = LLMRouter([air, flash])

    parsed, resp = asyncio.run(router.generate_json("s", "u", _Out))
    assert parsed.answer == "from-flash"
    assert resp.provider == "glm_flash"
    assert air.calls == 1


def test_glm_total_outage_falls_back_to_gemini() -> None:
    air = _FakeProvider("glm_air", [LLMError("timeout")])
    flash = _FakeProvider("glm_flash", [LLMError("HTTP 503")])
    gemini = _FakeProvider("gemini", ['{"answer": "from-gemini"}'])
    router = LLMRouter([air, flash, gemini])

    parsed, resp = asyncio.run(router.generate_json("s", "u", _Out))
    assert parsed.answer == "from-gemini"
    assert resp.provider == "gemini"
    assert air.calls == 1 and flash.calls == 1


def test_meta_model_records_actual_node() -> None:
    """_meta.model 记录实际生效的 provider/model（摘要侧拼 f"{provider}/{model}"）。"""
    air = _FakeProvider("glm_air", [LLMError("down")])
    flash = _FakeProvider("glm_flash", ['{"answer": "ok"}'])
    router = LLMRouter([air, flash])

    _, resp = asyncio.run(router.generate_json("s", "u", _Out))
    assert f"{resp.provider}/{resp.model}" == "glm_flash/glm_flash-model"
