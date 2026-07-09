"""上传接口 → 管道 → 状态查询的 API 层集成测试。"""
import io
import wave

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from tests.conftest import requires_db

pytestmark = requires_db


def _wav_bytes(seconds: float = 1.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


def test_upload_runs_pipeline_to_done(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        resp = client.post(
            "/api/meetings",
            files={"file": ("weekly.wav", _wav_bytes(), "audio/wav")},
            data={"title": "项目周会"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["title"] == "项目周会"
        meeting_id = body["id"]

        # TestClient 在响应后同步执行 BackgroundTasks，此时管道已跑完
        resp = client.get(f"/api/meetings/{meeting_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "done"
        assert body["duration"] and body["duration"] > 0
        assert body["error_message"] is None


def test_upload_rejects_unsupported_format(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        resp = client.post(
            "/api/meetings",
            files={"file": ("notes.txt", b"hello", "text/plain")},
            data={"title": "bad"},
        )
        assert resp.status_code == 422


def test_get_missing_meeting_404() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/meetings/00000000-0000-0000-0000-0000000000ff")
        assert resp.status_code == 404


def test_retry_only_allowed_when_failed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        resp = client.post(
            "/api/meetings",
            files={"file": ("ok.wav", _wav_bytes(), "audio/wav")},
            data={"title": "重试测试"},
        )
        meeting_id = resp.json()["id"]
        # 管道已成功（done），此时重试应被拒绝
        resp = client.post(f"/api/meetings/{meeting_id}/retry")
        assert resp.status_code == 409
