"""额度软限制与付费熔断（Tech Design M4 §12.3 Phase 3）：
软上限低价值让位、硬上限/到期熔断、付费开关、fail-open 与管理端状态。"""
import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models import DEFAULT_USER_ID, LlmUsageRecord
from app.services.llm import quota
from app.services.llm.base import LLMProvider, LLMResponse, LLMTaskType
from app.services.llm.router import LLMExhaustedError, LLMRouter
from pydantic import BaseModel
from tests.conftest import auth_headers, requires_db

pytestmark = requires_db


class _FakeProvider(LLMProvider):
    def __init__(self, name: str) -> None:
        self.name = name
        self.model = f"{name}-model"
        self.calls = 0

    async def complete(self, system, user, json_mode=True, temperature=0.2):
        self.calls += 1
        return LLMResponse(
            text='{"answer": "ok"}', provider=self.name, model=self.model
        )


class _Out(BaseModel):
    answer: str


async def _wipe() -> None:
    async with SessionLocal() as session:
        await session.execute(delete(LlmUsageRecord))
        await session.commit()
    quota.invalidate_cache()


async def _seed_air(total_tokens: int) -> None:
    async with SessionLocal() as session:
        session.add(
            LlmUsageRecord(
                meeting_id=uuid.uuid4(),
                user_id=DEFAULT_USER_ID,
                task_type="summary_final",
                provider="glm_air",
                model="glm-4.5-air",
                total_tokens=total_tokens,
                success=True,
                fallback_index=0,
            )
        )
        await session.commit()
    quota.invalidate_cache()


@pytest.fixture(autouse=True)
def _clean():
    asyncio.run(_wipe())
    yield


def _ask(providers: list[_FakeProvider], task: LLMTaskType):
    return asyncio.run(LLMRouter(list(providers), task=task).generate_json("s", "u", _Out))


# ---- 软上限：Air 从低价值任务让位，高价值保留 ----


def test_soft_limit_removes_air_from_low_value_tasks() -> None:
    asyncio.run(_seed_air(settings.glm_air_soft_limit_tokens))
    air, flash = _FakeProvider("glm_air"), _FakeProvider("glm_flash")

    _, resp = _ask([air, flash], LLMTaskType.QA_ANSWER)
    assert resp.provider == "glm_flash"
    assert air.calls == 0 and flash.calls == 1


def test_soft_limit_keeps_air_for_high_value_tasks() -> None:
    asyncio.run(_seed_air(settings.glm_air_soft_limit_tokens))
    air, flash = _FakeProvider("glm_air"), _FakeProvider("glm_flash")

    for task in (LLMTaskType.SUMMARY_FINAL, LLMTaskType.SUMMARY_EDIT):
        _, resp = _ask([air, flash], task)
        assert resp.provider == "glm_air"
    assert flash.calls == 0


def test_below_soft_limit_air_available_everywhere() -> None:
    asyncio.run(_seed_air(settings.glm_air_soft_limit_tokens - 1))
    air = _FakeProvider("glm_air")
    _, resp = _ask([air], LLMTaskType.QA_ANSWER)
    assert resp.provider == "glm_air"


# ---- 硬上限 / 到期：按配置熔断 ----


def test_hard_limit_blocks_air_for_all_tasks() -> None:
    asyncio.run(_seed_air(settings.glm_air_grant_total_tokens))
    air, flash = _FakeProvider("glm_air"), _FakeProvider("glm_flash")

    _, resp = _ask([air, flash], LLMTaskType.SUMMARY_FINAL)
    assert resp.provider == "glm_flash"
    assert air.calls == 0


def test_hard_limit_with_paid_allowed_keeps_air_for_high_value(monkeypatch) -> None:
    monkeypatch.setattr(settings, "glm_allow_paid_after_grant", True)
    asyncio.run(_seed_air(settings.glm_air_grant_total_tokens))
    air, flash = _FakeProvider("glm_air"), _FakeProvider("glm_flash")

    # 允许付费：高价值任务继续用 Air；软上限规则仍让低价值任务让位
    _, resp = _ask([air, flash], LLMTaskType.SUMMARY_FINAL)
    assert resp.provider == "glm_air"
    _, resp = _ask([air, flash], LLMTaskType.QA_ANSWER)
    assert resp.provider == "glm_flash"


def test_expired_paid_allowed_below_soft_limit_blocks_low_value(monkeypatch) -> None:
    """到期 + 允许付费 + 用量仍低于软上限：低价值任务不得静默付费走 Air。

    回归 review：此前只在累计达软上限时让低价值任务让位，到期后用量
    尚低会继续付费调用 Air（意外付费）。
    """
    monkeypatch.setattr(
        settings, "glm_grant_expires_at", "2020-01-01T00:00:00+08:00"
    )
    monkeypatch.setattr(settings, "glm_allow_paid_after_grant", True)
    asyncio.run(_seed_air(1000))  # 远低于软上限
    air, flash = _FakeProvider("glm_air"), _FakeProvider("glm_flash")

    # 低价值任务：即便用量很低，也让位到免费模型
    _, resp = _ask([air, flash], LLMTaskType.QA_ANSWER)
    assert resp.provider == "glm_flash"
    assert air.calls == 0
    # 高价值任务：允许付费续用 Air
    _, resp = _ask([air, flash], LLMTaskType.SUMMARY_FINAL)
    assert resp.provider == "glm_air"


def test_exhausted_paid_allowed_blocks_low_value(monkeypatch) -> None:
    """耗尽 + 允许付费：低价值任务让位（paid_high_value_only），高价值付费续用。"""
    monkeypatch.setattr(settings, "glm_allow_paid_after_grant", True)
    asyncio.run(_seed_air(settings.glm_air_grant_total_tokens))
    air, flash = _FakeProvider("glm_air"), _FakeProvider("glm_flash")

    _, resp = _ask([air, flash], LLMTaskType.INTENT_CLASSIFY)
    assert resp.provider == "glm_flash"
    assert air.calls == 0
    _, resp = _ask([air, flash], LLMTaskType.SUMMARY_EDIT)
    assert resp.provider == "glm_air"


def test_expired_grant_blocks_air(monkeypatch) -> None:
    monkeypatch.setattr(
        settings, "glm_grant_expires_at", "2020-01-01T00:00:00+08:00"
    )
    air, flash = _FakeProvider("glm_air"), _FakeProvider("glm_flash")
    _, resp = _ask([air, flash], LLMTaskType.SUMMARY_FINAL)
    assert resp.provider == "glm_flash"
    assert air.calls == 0


def test_blocked_only_air_raises_distinguishable_error(monkeypatch) -> None:
    """全列表被熔断时：错误文案区分到期/耗尽，且落 quota 审计记录。"""
    monkeypatch.setattr(
        settings, "glm_grant_expires_at", "2020-01-01T00:00:00+08:00"
    )
    air = _FakeProvider("glm_air")
    with pytest.raises(LLMExhaustedError, match="到期"):
        _ask([air], LLMTaskType.SUMMARY_FINAL)
    assert air.calls == 0

    async def _quota_records():
        async with SessionLocal() as session:
            return list(
                await session.scalars(
                    select(LlmUsageRecord).where(
                        LlmUsageRecord.error_type == "quota"
                    )
                )
            )

    records = asyncio.run(_quota_records())
    assert len(records) == 1
    assert records[0].task_type == "summary_final"
    assert "到期" in records[0].error_message


def test_exhausted_message_differs_from_expired() -> None:
    asyncio.run(_seed_air(settings.glm_air_grant_total_tokens))
    with pytest.raises(LLMExhaustedError, match="资源包总量"):
        _ask([_FakeProvider("glm_air")], LLMTaskType.SUMMARY_FINAL)


def test_non_air_router_untouched() -> None:
    """无 Air 的路由完全不受配额影响（也不查库）。"""
    asyncio.run(_seed_air(settings.glm_air_grant_total_tokens))
    flash = _FakeProvider("glm_flash")
    _, resp = _ask([flash], LLMTaskType.QA_ANSWER)
    assert resp.provider == "glm_flash"


def test_quota_failopen_when_db_unavailable(monkeypatch) -> None:
    """审计库不可用时 fail-open：不熔断，业务照常。"""

    def _boom(*args, **kwargs):
        raise RuntimeError("db down")

    # quota 在调用时才 import SessionLocal，patch 模块属性即可模拟断库
    monkeypatch.setattr("app.db.session.SessionLocal", _boom)
    air = _FakeProvider("glm_air")
    _, resp = _ask([air], LLMTaskType.QA_ANSWER)
    assert resp.provider == "glm_air"


# ---- 快照缓存 ----


def test_snapshot_cache_ttl(monkeypatch) -> None:
    monkeypatch.setattr(settings, "glm_quota_cache_ttl_seconds", 3600)
    asyncio.run(_seed_air(100))
    snap1 = asyncio.run(quota.get_quota_snapshot())
    assert snap1.air_tracked == 100

    async def _more():
        async with SessionLocal() as session:
            session.add(
                LlmUsageRecord(
                    task_type="summary_final", provider="glm_air",
                    model="glm-4.5-air", total_tokens=900, success=True,
                    fallback_index=0,
                )
            )
            await session.commit()

    asyncio.run(_more())
    assert asyncio.run(quota.get_quota_snapshot()).air_tracked == 100  # 命中缓存
    quota.invalidate_cache()
    assert asyncio.run(quota.get_quota_snapshot()).air_tracked == 1000


# ---- 管理端状态（§12.3「管理端显示明确状态」）----


def _air_grant(client, headers) -> dict:
    body = client.get("/api/llm-usage/stats", headers=headers).json()
    return next(g for g in body["grants"] if g["name"] == "glm_air")


def test_stats_shows_enforcement_states(monkeypatch) -> None:
    with TestClient(app) as client:
        headers = auth_headers(client)

        g = _air_grant(client, headers)
        assert g["enforcement"] == "none" and not g["soft_limit_reached"]
        assert g["soft_limit_tokens"] == settings.glm_air_soft_limit_tokens

        asyncio.run(_seed_air(settings.glm_air_soft_limit_tokens))
        g = _air_grant(client, headers)
        assert g["soft_limit_reached"] and not g["hard_limit_reached"]
        assert g["enforcement"] == "high_value_only"

        asyncio.run(_seed_air(settings.glm_air_grant_total_tokens))
        g = _air_grant(client, headers)
        assert g["hard_limit_reached"]
        assert g["enforcement"] == "blocked"
        assert g["enforcement_reason"] == "exhausted"


def test_stats_shows_expired_reason(monkeypatch) -> None:
    monkeypatch.setattr(
        settings, "glm_grant_expires_at", "2020-01-01T00:00:00+08:00"
    )
    with TestClient(app) as client:
        headers = auth_headers(client)
        g = _air_grant(client, headers)
        assert g["expired"] is True
        assert g["enforcement"] == "blocked"
        assert g["enforcement_reason"] == "expired"


def test_stats_expired_paid_shows_high_value_only(monkeypatch) -> None:
    """到期 + 允许付费 + 用量低于软上限：管理端显示 high_value_only 而非 blocked。"""
    monkeypatch.setattr(
        settings, "glm_grant_expires_at", "2020-01-01T00:00:00+08:00"
    )
    monkeypatch.setattr(settings, "glm_allow_paid_after_grant", True)
    asyncio.run(_seed_air(1000))
    with TestClient(app) as client:
        headers = auth_headers(client)
        g = _air_grant(client, headers)
        assert g["expired"] is True and not g["soft_limit_reached"]
        assert g["enforcement"] == "high_value_only"


def test_general_grant_warns_but_never_enforces() -> None:
    async def _seed_flash(total: int) -> None:
        async with SessionLocal() as session:
            session.add(
                LlmUsageRecord(
                    task_type="qa_answer", provider="glm_flash",
                    model="glm-4-flash", total_tokens=total, success=True,
                    fallback_index=0,
                )
            )
            await session.commit()

    asyncio.run(_seed_flash(settings.glm_general_soft_limit_tokens))
    with TestClient(app) as client:
        headers = auth_headers(client)
        body = client.get("/api/llm-usage/stats", headers=headers).json()
        g = next(x for x in body["grants"] if x["name"] == "glm_general")
    # embedding 不可降级混用：通用包只预警不熔断
    assert g["soft_limit_reached"] is True
    assert g["enforcement"] == "none"
