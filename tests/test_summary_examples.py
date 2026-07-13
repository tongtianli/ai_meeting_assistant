"""纪要范例库：CRUD、采纳为范例、few-shot 选取与 prompt 注入。"""
import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.llm.prompts import reduce_prompt, single_pass_prompt
from app.services.summary_text import render_summary_text
from tests.conftest import auth_headers, requires_db

_CONTENT = {
    "title": "每日例会：工作安排会",
    "participants": ["张三", "李四"],
    "summary": "布置了本周重点工作。",
    "topics": [
        {
            "title": "资产盘点推进",
            "owner": "张三",
            "items": ["资产盘点加快推进，外勤优先打车往返。", "盘点结果周五前上报。"],
        }
    ],
    "decisions": ["下周启动二期盘点"],
    "todos": [],
}


# ---------- 纯函数：文本渲染与 prompt 注入（无需 DB） ----------


def test_render_summary_text_company_format() -> None:
    text = render_summary_text(_CONTENT)
    assert "会议主题：每日例会：工作安排会" in text
    assert "参会人员：张三、李四" in text
    assert "一、资产盘点推进（责任人：张三）" in text
    assert "1．资产盘点加快推进" in text
    assert "2．盘点结果周五前上报。" in text
    assert "二、会议决定事项" in text


def test_render_summary_text_degraded_passthrough() -> None:
    assert render_summary_text({"text": "纯文本纪要"}) == "纯文本纪要"


def test_prompts_inject_examples_only_when_present() -> None:
    ex = ["会议主题：范例会\n一、议题\n1．条目"]
    with_ex = single_pass_prompt("[1] [00:00:01] 张三: 开会", ex)
    without = single_pass_prompt("[1] [00:00:01] 张三: 开会")
    assert "--- 范例 1 ---" in with_ex
    assert "范例会" in with_ex
    assert "严禁出现在本次纪要" in with_ex
    assert "范例" not in without.split("正文写作规范")[0]  # 无范例时不出现范例块
    assert "--- 范例 1 ---" not in without

    reduced = reduce_prompt(['{"title":"部分1"}'], ex)
    assert "--- 范例 1 ---" in reduced
    assert "--- 范例 1 ---" not in reduce_prompt(['{"title":"部分1"}'])


# ---------- 选取逻辑（DB） ----------


@requires_db
def test_load_style_examples_budget_and_toggle(monkeypatch) -> None:
    from app.db.session import SessionLocal
    from app.models import DEFAULT_USER_ID, SummaryExample
    from app.services.summarize import load_style_examples

    monkeypatch.setattr(settings, "summary_examples_max_count", 2)
    monkeypatch.setattr(settings, "summary_examples_max_chars", 100)

    async def scenario() -> None:
        async with SessionLocal() as session:
            rows = [
                SummaryExample(
                    user_id=DEFAULT_USER_ID, title="过长被跳过", content="x" * 200
                ),
                SummaryExample(
                    user_id=DEFAULT_USER_ID, title="停用不选", content="短", enabled=False
                ),
                SummaryExample(user_id=DEFAULT_USER_ID, title="范例A", content="a" * 40),
                SummaryExample(user_id=DEFAULT_USER_ID, title="范例B", content="b" * 40),
                SummaryExample(user_id=DEFAULT_USER_ID, title="范例C", content="c" * 40),
            ]
            session.add_all(rows)
            await session.commit()
            ids = [r.id for r in rows]

        try:
            async with SessionLocal() as session:
                selected = await load_style_examples(session, DEFAULT_USER_ID)
                titles = [e.title for e in selected]
                # 条数上限 2；停用与超预算的不入选
                assert len(titles) == 2
                assert "停用不选" not in titles
                assert "过长被跳过" not in titles
        finally:
            async with SessionLocal() as session:
                for eid in ids:
                    obj = await session.get(SummaryExample, eid)
                    if obj is not None:
                        await session.delete(obj)
                await session.commit()

    asyncio.run(scenario())


# ---------- API（DB） ----------


@requires_db
def test_examples_crud_flow() -> None:
    with TestClient(app) as client:
        headers = auth_headers(client)
        created: list[str] = []
        try:
            resp = client.post(
                "/api/summary-examples",
                json={"title": "手工范例", "content": "会议主题：x\n一、议题\n1．条目"},
                headers=headers,
            )
            assert resp.status_code == 201, resp.text
            ex = resp.json()
            created.append(ex["id"])
            assert ex["enabled"] is True
            assert ex["source_meeting_id"] is None

            # 列表可见
            resp = client.get("/api/summary-examples", headers=headers)
            assert any(e["id"] == ex["id"] for e in resp.json())

            # 停用 + 润色
            resp = client.patch(
                f"/api/summary-examples/{ex['id']}",
                json={"enabled": False, "content": "润色后的正文"},
                headers=headers,
            )
            assert resp.status_code == 200
            assert resp.json()["enabled"] is False
            assert resp.json()["content"] == "润色后的正文"

            # 删除
            resp = client.delete(
                f"/api/summary-examples/{ex['id']}", headers=headers
            )
            assert resp.status_code == 204
            created.remove(ex["id"])
            resp = client.get("/api/summary-examples", headers=headers)
            assert all(e["id"] != ex["id"] for e in resp.json())
        finally:
            for eid in created:
                client.delete(f"/api/summary-examples/{eid}", headers=headers)


@requires_db
def test_example_from_meeting_and_degraded_rejected() -> None:
    from app.db.session import SessionLocal
    from app.models import DEFAULT_USER_ID, Meeting, Summary

    async def make_meeting(content: dict) -> uuid.UUID:
        async with SessionLocal() as session:
            meeting = Meeting(user_id=DEFAULT_USER_ID, title="范例来源会议")
            session.add(meeting)
            await session.flush()
            session.add(
                Summary(meeting_id=meeting.id, version=1, content_json=content)
            )
            await session.commit()
            return meeting.id

    async def drop_meeting(mid: uuid.UUID) -> None:
        async with SessionLocal() as session:
            obj = await session.get(Meeting, mid)
            if obj is not None:
                await session.delete(obj)
                await session.commit()

    good = dict(_CONTENT, _meta={"degraded": False})
    degraded = {"text": "纯文本", "_meta": {"degraded": True}}
    mid = asyncio.run(make_meeting(good))
    bad_mid = asyncio.run(make_meeting(degraded))

    with TestClient(app) as client:
        headers = auth_headers(client)
        try:
            resp = client.post(
                f"/api/summary-examples/from-meeting/{mid}", headers=headers
            )
            assert resp.status_code == 201, resp.text
            ex = resp.json()
            assert ex["title"] == "每日例会：工作安排会"
            assert ex["source_meeting_id"] == str(mid)
            assert "一、资产盘点推进（责任人：张三）" in ex["content"]

            # 降级纪要不能采纳
            resp = client.post(
                f"/api/summary-examples/from-meeting/{bad_mid}", headers=headers
            )
            assert resp.status_code == 409

            # 无纪要的会议 404
            resp = client.post(
                f"/api/summary-examples/from-meeting/{uuid.uuid4()}",
                headers=headers,
            )
            assert resp.status_code == 404

            client.delete(f"/api/summary-examples/{ex['id']}", headers=headers)
        finally:
            asyncio.run(drop_meeting(mid))
            asyncio.run(drop_meeting(bad_mid))


def test_examples_require_auth() -> None:
    with TestClient(app) as client:
        assert client.get("/api/summary-examples").status_code in (401, 403)
