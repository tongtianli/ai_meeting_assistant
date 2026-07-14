"""AI 会议问答 RAG：嵌入+检索、回答引用反查、历史、归属/就绪校验、维度自愈。"""
import asyncio
import io
import uuid
import wave

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models import DEFAULT_USER_ID, ChatMessage, Meeting, TranscriptSegment
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


def _done_meeting(client, headers, title="问答测试") -> str:
    resp = client.post(
        "/api/meetings",
        files={"file": ("q.wav", _wav_bytes(), "audio/wav")},
        data={"title": title},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    mid = resp.json()["id"]
    assert client.get(f"/api/meetings/{mid}", headers=headers).json()["status"] == "done"
    return mid


async def _count_embedded(meeting_id: str) -> int:
    async with SessionLocal() as session:
        rows = await session.scalars(
            select(TranscriptSegment).where(
                TranscriptSegment.meeting_id == uuid.UUID(meeting_id),
                TranscriptSegment.embedding.is_not(None),
            )
        )
        return len(list(rows))


def test_ask_returns_answer_with_resolved_citation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)

        # 提问前 embedding 全空（暗桩）
        assert asyncio.run(_count_embedded(mid)) == 0

        resp = client.post(
            f"/api/meetings/{mid}/chat",
            json={"question": "上传接口谁负责，什么时候完成？"},
            headers=headers,
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["role"] == "assistant"
        assert body["content"]
        # mock QA 引用检索片段中最小 seq，反查到真实 segment
        assert len(body["citations"]) >= 1
        c = body["citations"][0]
        assert "seq" in c and "start_time" in c and c["text"]

        # 懒加载：提问后 segment 已回填 embedding（缓存）
        assert asyncio.run(_count_embedded(mid)) > 0

        # cited_segment_ids 落库为真实 segment UUID
        async def _check_persisted() -> None:
            async with SessionLocal() as session:
                assistant = await session.scalar(
                    select(ChatMessage).where(
                        ChatMessage.meeting_id == uuid.UUID(mid),
                        ChatMessage.role == "assistant",
                    )
                )
                assert assistant.cited_segment_ids
                seg = await session.get(
                    TranscriptSegment,
                    uuid.UUID(str(assistant.cited_segment_ids[0])),
                )
                assert seg is not None and seg.meeting_id == uuid.UUID(mid)

        asyncio.run(_check_persisted())


def test_match_speaker_person_ids() -> None:
    from app.services.qa import match_speaker_person_ids

    pid1, pid2 = uuid.uuid4(), uuid.uuid4()
    names = {
        "speaker_001": (pid1, "张三"),
        "speaker_002": (pid2, "李"),  # 单字名不参与匹配
        "speaker_003": (pid1, "张三"),  # 同人绑多个 label → 去重
    }
    assert match_speaker_person_ids(names, "张三说了什么") == [pid1]
    assert match_speaker_person_ids(names, "李说了什么") == []
    assert match_speaker_person_ids(names, "今天讨论了哪些议题") == []
    assert match_speaker_person_ids({}, "张三说了什么") == []


def test_speaker_name_boosts_recall(tmp_path, monkeypatch) -> None:
    """问题命中绑定真名时，该说话人的片段并入候选集（混合检索）。

    mock 词袋向量对中文整句几乎无重叠，向量排序不可控——把 qa_top_k
    压成 0 关掉向量召回，候选集是否非空完全由名字命中决定，测试确定。
    """
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "qa_top_k", 0)
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)

        # 未绑定真名：名字无从命中，向量召回又为 0 → 未找到
        resp = client.post(
            f"/api/meetings/{mid}/chat",
            json={"question": "张三说了什么"},
            headers=headers,
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["citations"] == []

        resp = client.post(
            f"/api/meetings/{mid}/speaker-bindings",
            json={"speaker_label": "speaker_001", "name": "张三"},
            headers=headers,
        )
        assert resp.status_code < 300, resp.text

        # 绑定后重问：候选集为张三的片段（seq 0/2/4/6），mock 引用最小 seq
        resp = client.post(
            f"/api/meetings/{mid}/chat",
            json={"question": "张三说了什么"},
            headers=headers,
        )
        body = resp.json()
        assert body["citations"], body
        assert body["citations"][0]["seq"] == 0
        assert body["citations"][0]["speaker_name"] == "张三"


def test_speaker_boost_merges_and_dedupes(tmp_path, monkeypatch) -> None:
    """默认 top_k 下向量召回与说话人补召回合并去重，引用仍落在真实 segment。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)
        client.post(
            f"/api/meetings/{mid}/speaker-bindings",
            json={"speaker_label": "speaker_001", "name": "张三"},
            headers=headers,
        )
        resp = client.post(
            f"/api/meetings/{mid}/chat",
            json={"question": "张三说了什么"},
            headers=headers,
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        # 补召回保证张三的 seq=0 一定在候选集里 → mock 的 min-seq 引用恒为 0
        assert body["citations"][0]["seq"] == 0
        assert body["citations"][0]["speaker_name"] == "张三"


def test_chat_history_ordered(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)

        client.post(
            f"/api/meetings/{mid}/chat",
            json={"question": "第一个问题"},
            headers=headers,
        )
        client.post(
            f"/api/meetings/{mid}/chat",
            json={"question": "第二个问题"},
            headers=headers,
        )
        history = client.get(f"/api/meetings/{mid}/chat", headers=headers).json()
        # 两轮 → 4 条（user/assistant × 2），user 在前
        assert [m["role"] for m in history] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert history[0]["content"] == "第一个问题"
        assert history[2]["content"] == "第二个问题"


def test_ask_requires_done_meeting(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    async def _seed_uploaded() -> uuid.UUID:
        async with SessionLocal() as session:
            m = Meeting(
                user_id=DEFAULT_USER_ID, title="未完成", audio_url="local://x.wav"
            )
            session.add(m)
            await session.commit()
            return m.id

    mid = asyncio.run(_seed_uploaded())
    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.post(
            f"/api/meetings/{mid}/chat",
            json={"question": "在吗"},
            headers=headers,
        )
        assert resp.status_code == 409


def test_ask_other_users_meeting_404(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    other_user = uuid.uuid4()
    other_meeting = uuid.uuid4()

    async def _seed() -> None:
        from app.models import MeetingStatus, User

        async with SessionLocal() as session:
            session.add(User(id=other_user, name="别人"))
            await session.flush()
            session.add(
                Meeting(
                    id=other_meeting,
                    user_id=other_user,
                    title="别人的会议",
                    status=MeetingStatus.done,
                    audio_url="local://y.wav",
                )
            )
            await session.commit()

    asyncio.run(_seed())
    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.post(
            f"/api/meetings/{other_meeting}/chat",
            json={"question": "偷看"},
            headers=headers,
        )
        assert resp.status_code == 404


def test_embedding_dim_self_heal(tmp_path, monkeypatch) -> None:
    """换 embedding 维度（模拟换 provider）后提问触发整场重嵌。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)
        # 首次提问：mock embedder（64 维）回填
        client.post(
            f"/api/meetings/{mid}/chat", json={"question": "问题一"}, headers=headers
        )

    # 篡改某 segment 的 embedding 维度，模拟历史遗留的异维向量
    async def _corrupt() -> None:
        async with SessionLocal() as session:
            seg = await session.scalar(
                select(TranscriptSegment).where(
                    TranscriptSegment.meeting_id == uuid.UUID(mid)
                )
            )
            seg.embedding = [0.1, 0.2, 0.3]  # 3 维，与 mock 的 64 维不符
            await session.commit()

    asyncio.run(_corrupt())

    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.post(
            f"/api/meetings/{mid}/chat", json={"question": "问题二"}, headers=headers
        )
        assert resp.status_code == 201  # 自愈重嵌后正常作答

    # 所有 segment 维度恢复一致（64），检索不再因维度不符报错
    async def _check_dims() -> None:
        async with SessionLocal() as session:
            rows = list(
                await session.scalars(
                    select(TranscriptSegment).where(
                        TranscriptSegment.meeting_id == uuid.UUID(mid)
                    )
                )
            )
            assert all(len(s.embedding) == 64 for s in rows)

    asyncio.run(_check_dims())
