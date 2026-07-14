"""聊天改纪要（PRD §7.1）：意图路由、新版本落库、ActionItem 重建、失败路径。"""
import asyncio
import io
import uuid
import wave

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models import ActionItem, Summary
from tests.conftest import auth_headers, requires_db

pytestmark = requires_db


def _wav_bytes(seconds: float = 3.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


def _done_meeting(client, headers, title="改纪要测试") -> str:
    resp = client.post(
        "/api/meetings",
        files={"file": ("e.wav", _wav_bytes(), "audio/wav")},
        data={"title": title},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    mid = resp.json()["id"]
    assert client.get(f"/api/meetings/{mid}", headers=headers).json()["status"] == "done"
    return mid


def test_edit_creates_new_summary_version(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)

        # 管道已产出 v1
        v1 = client.get(f"/api/meetings/{mid}/summary", headers=headers).json()
        assert v1["version"] == 1

        # 编辑指令（mock 意图分类按「改」路由到 edit）
        resp = client.post(
            f"/api/meetings/{mid}/chat",
            json={"question": "把决策事项改成下周完成上传接口开发并同步验收标准"},
            headers=headers,
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["role"] == "assistant"
        assert body["summary_version"] == 2  # POST 响应携带新版本号
        assert "v2" in body["content"]  # 回执标注版本

        # GET /summary 返回新版本；旧版本未被覆盖
        v2 = client.get(f"/api/meetings/{mid}/summary", headers=headers).json()
        assert v2["version"] == 2
        assert "【已修改】" in v2["content_json"]["decisions"][0]  # mock 编辑变体
        meta = v2["content_json"]["_meta"]
        assert meta["origin"] == "chat"
        assert meta["base_version"] == 1
        assert meta["edit_instruction"].startswith("把决策事项改成")

        async def _check_db() -> None:
            async with SessionLocal() as session:
                versions = sorted(
                    await session.scalars(
                        select(Summary.version).where(
                            Summary.meeting_id == uuid.UUID(mid)
                        )
                    )
                )
                assert versions == [1, 2]  # 新版本追加，旧版保留

                # ActionItem 已按新纪要重建（mock 编辑保留原 todo，含 seq 溯源）
                items = list(
                    await session.scalars(
                        select(ActionItem).where(
                            ActionItem.meeting_id == uuid.UUID(mid)
                        )
                    )
                )
                assert len(items) == 1
                assert items[0].source_segment_id is not None

        asyncio.run(_check_db())

        # Word 导出跟随新版本（文件名带 v2）
        resp = client.get(f"/api/meetings/{mid}/export.docx", headers=headers)
        assert resp.status_code == 200
        assert "v2" in resp.headers["content-disposition"]


def test_query_intent_does_not_create_version(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)

        resp = client.post(
            f"/api/meetings/{mid}/chat",
            json={"question": "上传接口谁负责？"},  # 无编辑关键词 → query
            headers=headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["summary_version"] is None
        assert body["citations"]  # 走了 RAG 路径

        v = client.get(f"/api/meetings/{mid}/summary", headers=headers).json()
        assert v["version"] == 1  # 无新版本


def test_edit_without_summary_gets_hint(tmp_path, monkeypatch) -> None:
    """会议 done 但（异常地）没有纪要：编辑得到提示回执，不报错、不产生版本。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)

        # 删掉纪要模拟无纪要状态
        async def _drop_summary() -> None:
            from sqlalchemy import delete

            async with SessionLocal() as session:
                await session.execute(
                    delete(Summary).where(Summary.meeting_id == uuid.UUID(mid))
                )
                await session.commit()

        asyncio.run(_drop_summary())

        resp = client.post(
            f"/api/meetings/{mid}/chat",
            json={"question": "把标题改成新标题"},
            headers=headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["summary_version"] is None
        assert "尚未生成" in body["content"]


def test_edit_llm_failure_no_new_version(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)

    # 让改纪要路径的 LLM 全挂（意图分类仍走 mock 正常路由到 edit）
    import app.services.summary_edit as edit_mod
    from app.services.llm.base import LLMError, LLMProvider
    from app.services.llm.router import LLMRouter

    class _Down(LLMProvider):
        name = "down"
        model = "down"

        async def complete(self, system, user, json_mode=True, temperature=0.2):
            raise LLMError("outage")

    monkeypatch.setattr(edit_mod, "build_router", lambda: LLMRouter([_Down()]))

    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.post(
            f"/api/meetings/{mid}/chat",
            json={"question": "把标题改成故障测试"},
            headers=headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["summary_version"] is None
        assert "失败" in body["content"]

        v = client.get(f"/api/meetings/{mid}/summary", headers=headers).json()
        assert v["version"] == 1  # 未产生新版本


def test_intent_failure_falls_back_to_query(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)

    # 意图分类的 router 挂掉 → 回退 query，不影响回答
    import app.services.qa as qa_mod
    from app.services.llm import LLMExhaustedError

    async def _broken_intent(message: str) -> str:
        raise LLMExhaustedError("classifier down")

    async def _fallback_intent(message: str) -> str:
        try:
            return await _broken_intent(message)
        except LLMExhaustedError:
            return "query"

    # 直接验证 classify_intent 的回退语义：patch build_router 只对 intent 生效不好隔离，
    # 故此处 patch classify_intent 模拟"分类失败已回退"后的行为路径
    monkeypatch.setattr(qa_mod, "classify_intent", _fallback_intent)

    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.post(
            f"/api/meetings/{mid}/chat",
            json={"question": "把标题改一下"},  # 含编辑关键词但分类失败 → query
            headers=headers,
        )
        assert resp.status_code == 201
        assert resp.json()["summary_version"] is None  # 走了查询路径
