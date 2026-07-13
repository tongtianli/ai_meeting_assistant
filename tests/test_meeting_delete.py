"""会议删除（PRD §9.4）：级联清库 + 磁盘文件清理 + 权限校验。"""
import asyncio
import io
import uuid
import wave

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models import (
    DEFAULT_USER_ID,
    ActionItem,
    Meeting,
    Summary,
    TranscriptSegment,
    User,
    VoiceSample,
)
from tests.conftest import auth_headers, requires_db

pytestmark = requires_db


def _wav_bytes(seconds: float = 1.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


def _upload(client, headers, title="待删除会议", filename="del.wav"):
    return client.post(
        "/api/meetings",
        files={"file": (filename, _wav_bytes(), "audio/wav")},
        data={"title": title},
        headers=headers,
    )


async def _count(model, **filters) -> int:
    async with SessionLocal() as session:
        stmt = select(func.count()).select_from(model)
        for col, val in filters.items():
            stmt = stmt.where(getattr(model, col) == val)
        return await session.scalar(stmt)


def test_delete_meeting_cascades_db_and_disk(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        meeting_id = _upload(client, headers).json()["id"]

        # 管道已 done：segments / voice_samples / summary / action_items 都已生成
        assert asyncio.run(_count(TranscriptSegment, meeting_id=meeting_id)) > 0
        assert asyncio.run(_count(VoiceSample, source_meeting_id=meeting_id)) > 0
        assert asyncio.run(_count(Summary, meeting_id=meeting_id)) > 0

        # 记录磁盘上的音频与转码产物路径
        audio_dir = tmp_path / "audio"
        transcoded_dir = tmp_path / "transcoded"
        audio_files = list(audio_dir.glob("*"))
        transcoded_files = list(transcoded_dir.glob("*"))
        assert audio_files, "上传后应有音频文件落盘"
        assert transcoded_files, "转码后应有中间产物落盘"

        # 删除
        resp = client.delete(f"/api/meetings/{meeting_id}", headers=headers)
        assert resp.status_code == 204
        assert resp.content == b""

        # 会议本身与全部派生数据消失
        assert client.get(f"/api/meetings/{meeting_id}", headers=headers).status_code == 404
        assert asyncio.run(_count(Meeting, id=meeting_id)) == 0
        assert asyncio.run(_count(TranscriptSegment, meeting_id=meeting_id)) == 0
        assert asyncio.run(_count(VoiceSample, source_meeting_id=meeting_id)) == 0
        assert asyncio.run(_count(Summary, meeting_id=meeting_id)) == 0
        assert asyncio.run(_count(ActionItem, meeting_id=meeting_id)) == 0

        # 磁盘文件清理干净
        assert list(audio_dir.glob("*")) == []
        assert list(transcoded_dir.glob("*")) == []


def test_delete_meeting_keeps_example_but_nulls_source(tmp_path, monkeypatch) -> None:
    """已采纳为范例的会议删除后，范例保留、source_meeting_id 置空（SET NULL）。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        meeting_id = _upload(client, headers).json()["id"]

        # 把纪要采纳为范例
        resp = client.post(
            f"/api/summary-examples/from-meeting/{meeting_id}", headers=headers
        )
        assert resp.status_code == 201, resp.text
        example_id = resp.json()["id"]
        assert resp.json()["source_meeting_id"] == meeting_id

        # 删除会议
        assert client.delete(f"/api/meetings/{meeting_id}", headers=headers).status_code == 204

        # 范例仍在，来源指针被置空
        examples = client.get("/api/summary-examples", headers=headers).json()
        match = [e for e in examples if e["id"] == example_id]
        assert match, "范例应在会议删除后保留"
        assert match[0]["source_meeting_id"] is None


def test_delete_missing_meeting_404() -> None:
    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.delete(
            "/api/meetings/00000000-0000-0000-0000-0000000000ff", headers=headers
        )
        assert resp.status_code == 404


def test_delete_other_users_meeting_404(tmp_path, monkeypatch) -> None:
    """删除他人会议应 404（本次权限修复：_get_meeting_or_404 校验 user_id）。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    other_user_id = uuid.uuid4()
    other_meeting_id = uuid.uuid4()

    async def _seed() -> None:
        async with SessionLocal() as session:
            session.add(User(id=other_user_id, name="别的用户"))
            await session.flush()  # 无 ORM 关系，需先落 user 再插 meeting
            session.add(
                Meeting(
                    id=other_meeting_id,
                    user_id=other_user_id,
                    title="别人的会议",
                    audio_url="local://someone-else.wav",
                )
            )
            await session.commit()

    asyncio.run(_seed())

    with TestClient(app) as client:
        headers = auth_headers(client)  # 默认用户
        resp = client.delete(f"/api/meetings/{other_meeting_id}", headers=headers)
        assert resp.status_code == 404
        # 未被删除：仍属于原用户
        assert asyncio.run(_count(Meeting, id=other_meeting_id)) == 1


def test_delete_meeting_without_audio_does_not_crash() -> None:
    """audio_url 为空的会议删除不应因磁盘清理而报错。"""
    meeting_id = uuid.uuid4()

    async def _seed() -> None:
        async with SessionLocal() as session:
            session.add(
                Meeting(
                    id=meeting_id,
                    user_id=DEFAULT_USER_ID,
                    title="无音频会议",
                    audio_url=None,
                )
            )
            await session.commit()

    asyncio.run(_seed())

    with TestClient(app) as client:
        headers = auth_headers(client)
        assert client.delete(f"/api/meetings/{meeting_id}", headers=headers).status_code == 204
        assert asyncio.run(_count(Meeting, id=meeting_id)) == 0
