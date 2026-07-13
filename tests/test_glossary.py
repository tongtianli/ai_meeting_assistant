"""全局术语表：CRUD、去重、归属校验，以及热词注入管道的端到端验证。"""
import asyncio
import io
import uuid
import wave

from fastapi.testclient import TestClient

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models import DEFAULT_USER_ID, GlossaryTerm, Meeting
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


def _wipe_glossary() -> None:
    """测试库会话级共享、不逐条重置——注入类断言前清空术语，避免跨测试污染。"""
    from sqlalchemy import delete

    async def _run() -> None:
        async with SessionLocal() as session:
            await session.execute(delete(GlossaryTerm))
            await session.commit()

    asyncio.run(_run())


def test_glossary_crud_flow() -> None:
    _wipe_glossary()
    with TestClient(app) as client:
        headers = auth_headers(client)

        # 批量新增，含空白/重复/超长——应规范化去重
        resp = client.post(
            "/api/glossary",
            json={"terms": ["  声纹  ", "pgvector", "声纹", "", "diarization"]},
            headers=headers,
        )
        assert resp.status_code == 201, resp.text
        created = resp.json()
        assert sorted(t["term"] for t in created) == [
            "diarization",
            "pgvector",
            "声纹",
        ]
        assert all(t["enabled"] for t in created)

        # 再次提交已存在的 + 一个新词 → 只新增新词
        resp = client.post(
            "/api/glossary",
            json={"terms": ["声纹", "Paraformer"]},
            headers=headers,
        )
        assert [t["term"] for t in resp.json()] == ["Paraformer"]

        # 列表共 4 条
        terms = client.get("/api/glossary", headers=headers).json()
        assert len(terms) == 4

        # 停用一条
        tid = terms[0]["id"]
        resp = client.patch(
            f"/api/glossary/{tid}", json={"enabled": False}, headers=headers
        )
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

        # 删除一条
        assert client.delete(f"/api/glossary/{tid}", headers=headers).status_code == 204
        assert len(client.get("/api/glossary", headers=headers).json()) == 3


def test_glossary_rejects_all_blank() -> None:
    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.post(
            "/api/glossary", json={"terms": ["   ", ""]}, headers=headers
        )
        assert resp.status_code == 422


def test_glossary_requires_auth() -> None:
    with TestClient(app) as client:
        assert client.get("/api/glossary").status_code in (401, 403)


def test_glossary_other_users_term_404() -> None:
    other_user = uuid.uuid4()
    other_term = uuid.uuid4()

    async def _seed() -> None:
        from app.models import User

        async with SessionLocal() as session:
            session.add(User(id=other_user, name="别的用户"))
            await session.flush()
            session.add(
                GlossaryTerm(id=other_term, user_id=other_user, term="机密术语")
            )
            await session.commit()

    asyncio.run(_seed())
    with TestClient(app) as client:
        headers = auth_headers(client)
        assert client.patch(
            f"/api/glossary/{other_term}", json={"enabled": False}, headers=headers
        ).status_code == 404
        assert client.delete(
            f"/api/glossary/{other_term}", headers=headers
        ).status_code == 404


def test_enabled_terms_injected_as_hotwords(tmp_path, monkeypatch) -> None:
    """核心：转写时管道把启用的术语作为 hotwords 传给 provider，停用的不传。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    _wipe_glossary()

    captured: dict[str, object] = {}
    from app.services.asr.mock import MockASRProvider

    real_transcribe = MockASRProvider.transcribe

    async def spy_transcribe(self, audio_path, hotwords=None, voiceprint_ids=None):
        captured["hotwords"] = hotwords
        return await real_transcribe(
            self, audio_path, hotwords=hotwords, voiceprint_ids=voiceprint_ids
        )

    monkeypatch.setattr(MockASRProvider, "transcribe", spy_transcribe)

    async def _seed_terms() -> None:
        async with SessionLocal() as session:
            session.add_all(
                [
                    GlossaryTerm(user_id=DEFAULT_USER_ID, term="声纹", enabled=True),
                    GlossaryTerm(user_id=DEFAULT_USER_ID, term="pgvector", enabled=True),
                    GlossaryTerm(
                        user_id=DEFAULT_USER_ID, term="停用词", enabled=False
                    ),
                ]
            )
            await session.commit()

    asyncio.run(_seed_terms())

    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.post(
            "/api/meetings",
            files={"file": ("g.wav", _wav_bytes(), "audio/wav")},
            data={"title": "热词注入测试"},
            headers=headers,
        )
        assert resp.status_code == 201
        mid = resp.json()["id"]
        assert client.get(f"/api/meetings/{mid}", headers=headers).json()["status"] == "done"

    assert captured["hotwords"] is not None
    assert set(captured["hotwords"]) == {"声纹", "pgvector"}  # 停用词不注入


def test_hotwords_capped_by_config(tmp_path, monkeypatch) -> None:
    """启用术语超过 glossary_max_terms 时按上限截断。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "glossary_max_terms", 3)
    _wipe_glossary()

    captured: dict[str, object] = {}
    from app.services.asr.mock import MockASRProvider

    real_transcribe = MockASRProvider.transcribe

    async def spy_transcribe(self, audio_path, hotwords=None, voiceprint_ids=None):
        captured["hotwords"] = hotwords
        return await real_transcribe(
            self, audio_path, hotwords=hotwords, voiceprint_ids=voiceprint_ids
        )

    monkeypatch.setattr(MockASRProvider, "transcribe", spy_transcribe)

    async def _seed() -> None:
        async with SessionLocal() as session:
            session.add_all(
                GlossaryTerm(user_id=DEFAULT_USER_ID, term=f"术语{i}")
                for i in range(10)
            )
            await session.commit()

    asyncio.run(_seed())

    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.post(
            "/api/meetings",
            files={"file": ("c.wav", _wav_bytes(), "audio/wav")},
            data={"title": "截断测试"},
            headers=headers,
        )
        mid = resp.json()["id"]
        assert client.get(f"/api/meetings/{mid}", headers=headers).json()["status"] == "done"

    assert len(captured["hotwords"]) == 3


def test_no_terms_passes_none(tmp_path, monkeypatch) -> None:
    """无启用术语时 hotwords 为 None（不影响现有行为）。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    _wipe_glossary()

    captured: dict[str, object] = {}
    from app.services.asr.mock import MockASRProvider

    real_transcribe = MockASRProvider.transcribe

    async def spy_transcribe(self, audio_path, hotwords=None, voiceprint_ids=None):
        captured["hotwords"] = hotwords
        return await real_transcribe(
            self, audio_path, hotwords=hotwords, voiceprint_ids=voiceprint_ids
        )

    monkeypatch.setattr(MockASRProvider, "transcribe", spy_transcribe)

    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.post(
            "/api/meetings",
            files={"file": ("n.wav", _wav_bytes(), "audio/wav")},
            data={"title": "无术语"},
            headers=headers,
        )
        mid = resp.json()["id"]
        assert client.get(f"/api/meetings/{mid}", headers=headers).json()["status"] == "done"

    assert captured["hotwords"] is None
