"""LLM 用量审计（Tech Design M4 §11/§12 Phase 2）：
记录成功/失败/fallback、会议上下文标签、统计端点与资源包预警。"""
import asyncio
import io
import uuid
import wave

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlalchemy import delete, select

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models import DEFAULT_USER_ID, LlmUsageRecord
from app.services.llm.base import LLMError, LLMProvider, LLMResponse, LLMTaskType
from app.services.llm.router import LLMExhaustedError, LLMRouter
from app.services.llm.usage import usage_context
from tests.conftest import auth_headers, requires_db

pytestmark = requires_db


class _Out(BaseModel):
    answer: str


class _FakeProvider(LLMProvider):
    """按脚本依次返回结果或抛错的假 provider（带确定性 token 数）。"""

    def __init__(self, name: str, script: list) -> None:
        self.name = name
        self.model = f"{name}-model"
        self.script = list(script)

    async def complete(self, system, user, json_mode=True, temperature=0.2):
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return LLMResponse(
            text=action,
            provider=self.name,
            model=self.model,
            prompt_tokens=100,
            completion_tokens=50,
        )


def _wav_bytes(seconds: float = 3.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


async def _records(**filters) -> list[LlmUsageRecord]:
    async with SessionLocal() as session:
        stmt = select(LlmUsageRecord).order_by(
            LlmUsageRecord.created_at, LlmUsageRecord.fallback_index
        )
        for key, value in filters.items():
            stmt = stmt.where(getattr(LlmUsageRecord, key) == value)
        return list(await session.scalars(stmt))


async def _wipe_records() -> None:
    async with SessionLocal() as session:
        await session.execute(delete(LlmUsageRecord))
        await session.commit()


@pytest.fixture(autouse=True)
def _clean_usage():
    asyncio.run(_wipe_records())
    yield


# ---- Router 层记录 ----


def test_success_recorded_with_context_and_tokens() -> None:
    provider = _FakeProvider("glm_air", ['{"answer": "ok"}'])
    router = LLMRouter([provider], task=LLMTaskType.SUMMARY_FINAL)
    meeting_id, user_id = uuid.uuid4(), uuid.uuid4()

    async def _run():
        with usage_context(meeting_id=meeting_id, user_id=user_id):
            await router.generate_json("s", "u", _Out)

    asyncio.run(_run())
    rows = asyncio.run(_records(task_type="summary_final"))
    assert len(rows) == 1
    r = rows[0]
    assert r.provider == "glm_air" and r.model == "glm_air-model"
    assert r.success is True and r.fallback_index == 0
    assert (r.input_tokens, r.output_tokens, r.total_tokens) == (100, 50, 150)
    assert r.meeting_id == meeting_id and r.user_id == user_id
    assert r.latency_ms is not None and r.error_type is None


def test_transport_failure_and_fallback_recorded() -> None:
    primary = _FakeProvider("glm_air", [LLMError("boom")])
    fallback = _FakeProvider("glm_flash", ['{"answer": "ok"}'])
    router = LLMRouter([primary, fallback], task=LLMTaskType.QA_ANSWER)

    asyncio.run(router.generate_json("s", "u", _Out))
    rows = asyncio.run(_records(task_type="qa_answer"))
    assert [(r.provider, r.success, r.fallback_index) for r in rows] == [
        ("glm_air", False, 0),
        ("glm_flash", True, 1),
    ]
    failed = rows[0]
    # 传输层失败：无 token 消耗，不伪造
    assert failed.error_type == "transport" and "boom" in failed.error_message
    assert failed.total_tokens is None
    # 上下文未设置时 meeting/user 留空
    assert failed.meeting_id is None and failed.user_id is None


def test_schema_validation_failure_records_tokens() -> None:
    # HTTP 调用成功但 JSON 校验失败：token 已消耗，必须入账
    provider = _FakeProvider("glm_air", ['{"wrong": 1}', '{"wrong": 2}'])
    router = LLMRouter([provider], task=LLMTaskType.SUMMARY_EDIT)

    with pytest.raises(LLMExhaustedError):
        asyncio.run(router.generate_json("s", "u", _Out))
    rows = asyncio.run(_records(task_type="summary_edit"))
    assert len(rows) == 2
    for r in rows:
        assert r.success is False and r.error_type == "schema_validation"
        assert r.total_tokens == 150


def test_generate_text_recorded() -> None:
    router = LLMRouter(
        [_FakeProvider("glm_air", ["纯文本纪要"])], task=LLMTaskType.SUMMARY_FINAL
    )
    asyncio.run(router.generate_text("s", "u"))
    rows = asyncio.run(_records(provider="glm_air"))
    assert len(rows) == 1 and rows[0].success is True


# ---- 端到端：管道与聊天入口打上会议标签 ----


def test_pipeline_tags_usage_with_meeting(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.post(
            "/api/meetings",
            files={"file": ("u.wav", _wav_bytes(), "audio/wav")},
            data={"title": "用量审计"},
            headers=headers,
        )
        assert resp.status_code == 201, resp.text
        mid = uuid.UUID(resp.json()["id"])
        assert (
            client.get(f"/api/meetings/{mid}", headers=headers).json()["status"]
            == "done"
        )

    rows = asyncio.run(_records(meeting_id=mid))
    assert rows, "pipeline 应产生带会议标签的用量记录"
    assert {r.task_type for r in rows} == {"summary_final"}  # 短会议单块 single-pass
    assert all(r.user_id == DEFAULT_USER_ID and r.success for r in rows)


def test_chat_tags_usage_with_meeting(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.post(
            "/api/meetings",
            files={"file": ("c.wav", _wav_bytes(), "audio/wav")},
            data={"title": "聊天用量"},
            headers=headers,
        )
        mid = uuid.UUID(resp.json()["id"])
        r = client.post(
            f"/api/meetings/{mid}/chat",
            json={"question": "上传接口谁负责？"},
            headers=headers,
        )
        assert r.status_code == 201, r.text

    task_types = {r.task_type for r in asyncio.run(_records(meeting_id=mid))}
    # 聊天链路：意图分类 + RAG 回答（mock embedder 不记账）
    assert {"intent_classify", "qa_answer"} <= task_types


# ---- 统计端点与资源包预警 ----


async def _seed(records: list[dict]) -> None:
    async with SessionLocal() as session:
        for kw in records:
            session.add(LlmUsageRecord(**kw))
        await session.commit()


def _seed_air_scenario() -> tuple[uuid.UUID, uuid.UUID]:
    m1, m2 = uuid.uuid4(), uuid.uuid4()
    asyncio.run(
        _seed(
            [
                # 两场会议各消耗 1M Air token
                dict(
                    meeting_id=m1, user_id=DEFAULT_USER_ID,
                    task_type="summary_final", provider="glm_air",
                    model="glm-4.5-air", input_tokens=800_000, output_tokens=200_000,
                    total_tokens=1_000_000, success=True, fallback_index=0,
                    latency_ms=1200,
                ),
                dict(
                    meeting_id=m2, user_id=DEFAULT_USER_ID,
                    task_type="summary_final", provider="glm_air",
                    model="glm-4.5-air", input_tokens=900_000, output_tokens=100_000,
                    total_tokens=1_000_000, success=True, fallback_index=0,
                    latency_ms=1500,
                ),
                # Air 失败一次 → fallback 到 flash 成功
                dict(
                    meeting_id=m2, user_id=DEFAULT_USER_ID,
                    task_type="qa_answer", provider="glm_air",
                    model="glm-4.5-air", success=False, fallback_index=0,
                    error_type="transport", error_message="HTTP 429: rate limited",
                    latency_ms=300,
                ),
                dict(
                    meeting_id=m2, user_id=DEFAULT_USER_ID,
                    task_type="qa_answer", provider="glm_flash",
                    model="glm-4-flash", input_tokens=1000, output_tokens=200,
                    total_tokens=1200, success=True, fallback_index=1, latency_ms=800,
                ),
            ]
        )
    )
    return m1, m2


def test_stats_endpoint_math() -> None:
    _seed_air_scenario()
    with TestClient(app) as client:
        headers = auth_headers(client)
        assert client.get("/api/llm-usage/stats").status_code == 401  # 需认证
        resp = client.get("/api/llm-usage/stats", headers=headers)
        assert resp.status_code == 200, resp.text
        body = resp.json()

    assert body["total_calls"] == 4
    assert body["total_tokens"] == 2_001_200

    providers = {p["provider"]: p for p in body["by_provider"]}
    air = providers["glm_air"]
    assert air["calls"] == 3 and air["success_calls"] == 2
    assert air["success_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert air["total_tokens"] == 2_000_000
    flash = providers["glm_flash"]
    assert flash["fallback_calls"] == 1 and flash["fallback_rate"] == 1.0

    tasks = {t["task_type"]: t for t in body["by_task"]}
    assert tasks["summary_final"]["total_tokens"] == 2_000_000
    assert tasks["qa_answer"]["calls"] == 2
    assert tasks["qa_answer"]["success_rate"] == 0.5

    grants = {g["name"]: g for g in body["grants"]}
    air_grant = grants["glm_air"]
    assert air_grant["grant_total_tokens"] == settings.glm_air_grant_total_tokens
    assert air_grant["tracked_total_tokens"] == 2_000_000
    # estimated_remaining = configured_grant_total - tracked_total_tokens（§11）
    assert (
        air_grant["estimated_remaining_tokens"]
        == settings.glm_air_grant_total_tokens - 2_000_000
    )
    # 每场平均 1M → 预计还能生成 (12M-2M)/1M = 10 场
    assert air_grant["avg_tokens_per_meeting"] == 1_000_000
    assert air_grant["estimated_remaining_meetings"] == 10
    assert grants["glm_general"]["tracked_total_tokens"] == 1200

    failures = body["recent_failures"]
    assert len(failures) == 1
    assert failures[0]["provider"] == "glm_air"
    assert failures[0]["error_type"] == "transport"
    assert "429" in failures[0]["error_message"]


def test_stats_isolated_per_user_but_grant_shared() -> None:
    """用户隔离：A 看不到 B 的明细/失败/会议；资源包额度账号级共享。"""
    other_user, other_meeting = uuid.uuid4(), uuid.uuid4()
    asyncio.run(
        _seed(
            [
                # 当前用户（DEFAULT_USER）自己的一次成功调用
                dict(
                    meeting_id=uuid.uuid4(), user_id=DEFAULT_USER_ID,
                    task_type="summary_final", provider="glm_air",
                    model="glm-4.5-air", input_tokens=400, output_tokens=100,
                    total_tokens=500, success=True, fallback_index=0,
                ),
                # 其他用户的成功与失败调用
                dict(
                    meeting_id=other_meeting, user_id=other_user,
                    task_type="qa_answer", provider="glm_flash",
                    model="glm-4-flash", input_tokens=8000, output_tokens=2000,
                    total_tokens=10_000, success=True, fallback_index=0,
                ),
                dict(
                    meeting_id=other_meeting, user_id=other_user,
                    task_type="qa_answer", provider="glm_flash",
                    model="glm-4-flash", success=False, fallback_index=0,
                    error_type="transport", error_message="secret failure of B",
                ),
            ]
        )
    )
    with TestClient(app) as client:
        headers = auth_headers(client)
        body = client.get("/api/llm-usage/stats", headers=headers).json()

    # 明细统计只含当前用户：他人的调用/任务/失败一概不可见
    assert body["total_calls"] == 1 and body["total_tokens"] == 500
    assert {p["provider"] for p in body["by_provider"]} == {"glm_air"}
    assert {t["task_type"] for t in body["by_task"]} == {"summary_final"}
    assert body["recent_failures"] == []

    grants = {g["name"]: g for g in body["grants"]}
    # 资源包额度绑定共享的 GLM key：累计消耗跨用户统计
    assert grants["glm_general"]["tracked_total_tokens"] == 10_000
    assert grants["glm_air"]["tracked_total_tokens"] == 500
    # 每场平均是用户级指标：不被他人的 1 万 token 会议污染
    assert grants["glm_air"]["avg_tokens_per_meeting"] == 500
    assert grants["glm_general"]["avg_tokens_per_meeting"] is None


def test_grant_expiry_warning_levels(monkeypatch) -> None:
    from datetime import datetime, timedelta, timezone

    with TestClient(app) as client:
        headers = auth_headers(client)

        def _get_air():
            body = client.get("/api/llm-usage/stats", headers=headers).json()
            return next(g for g in body["grants"] if g["name"] == "glm_air")

        # 距离 2026-10-10 还有约 88 天：不触发提醒
        air = _get_air()
        assert air["days_until_expiry"] > 30 and air["expiry_warning"] is None

        # 5 天后到期 → 触发「到期前 7 天」提醒
        soon = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()
        monkeypatch.setattr(settings, "glm_grant_expires_at", soon)
        air = _get_air()
        assert air["expiry_warning"] is not None and "7 天" in air["expiry_warning"]

        # 已过期
        past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        monkeypatch.setattr(settings, "glm_grant_expires_at", past)
        air = _get_air()
        assert air["days_until_expiry"] < 0
        assert "已于" in air["expiry_warning"]


def test_usage_records_survive_meeting_deletion(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.post(
            "/api/meetings",
            files={"file": ("d.wav", _wav_bytes(), "audio/wav")},
            data={"title": "删除后审计存活"},
            headers=headers,
        )
        mid = uuid.UUID(resp.json()["id"])
        assert (
            client.get(f"/api/meetings/{mid}", headers=headers).json()["status"]
            == "done"
        )
        assert asyncio.run(_records(meeting_id=mid))
        assert client.delete(f"/api/meetings/{mid}", headers=headers).status_code == 204

    # 无外键：会议删除后成本审计记录仍在
    assert asyncio.run(_records(meeting_id=mid))
