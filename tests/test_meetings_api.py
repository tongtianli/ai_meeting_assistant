"""上传接口 → 管道 → 状态查询的 API 层集成测试。"""
import io
import wave

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
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


def _upload(client, headers, title="项目周会", filename="weekly.wav"):
    return client.post(
        "/api/meetings",
        files={"file": (filename, _wav_bytes(), "audio/wav")},
        data={"title": title},
        headers=headers,
    )


def test_upload_runs_pipeline_to_done(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = _upload(client, headers)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["title"] == "项目周会"
        meeting_id = body["id"]

        # TestClient 在响应后同步执行 BackgroundTasks，此时管道已跑完
        resp = client.get(f"/api/meetings/{meeting_id}", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "done"
        assert body["duration"] and body["duration"] > 0
        assert body["error_message"] is None

        # 列表接口能看到这条会议
        resp = client.get("/api/meetings", headers=headers)
        assert meeting_id in [m["id"] for m in resp.json()]

        # 摘要已生成（mock LLM）
        resp = client.get(f"/api/meetings/{meeting_id}/summary", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["version"] == 1
        assert body["content_json"]["title"]


def test_upload_rejects_unsupported_format(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.post(
            "/api/meetings",
            files={"file": ("notes.txt", b"hello", "text/plain")},
            data={"title": "bad"},
            headers=headers,
        )
        assert resp.status_code == 422


def test_get_missing_meeting_404() -> None:
    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.get(
            "/api/meetings/00000000-0000-0000-0000-0000000000ff", headers=headers
        )
        assert resp.status_code == 404


def test_retry_only_allowed_when_failed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        meeting_id = _upload(client, headers, title="重试测试", filename="ok.wav").json()["id"]
        # 管道已成功（done），此时重试应被拒绝
        resp = client.post(f"/api/meetings/{meeting_id}/retry", headers=headers)
        assert resp.status_code == 409


def test_resummarize_appends_new_version(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        meeting_id = _upload(client, headers, title="重跑纪要", filename="rs.wav").json()["id"]

        resp = client.post(f"/api/meetings/{meeting_id}/resummarize", headers=headers)
        assert resp.status_code == 200, resp.text

        # TestClient 同步执行 BackgroundTasks，此时重跑已完成
        body = client.get(f"/api/meetings/{meeting_id}", headers=headers).json()
        assert body["status"] == "done"
        assert body["error_message"] is None
        summary = client.get(
            f"/api/meetings/{meeting_id}/summary", headers=headers
        ).json()
        assert summary["version"] == 2  # 追加新版本而非覆盖
        # 版本来源可追溯：重跑产生的版本标记 origin=resummarize（首版为 pipeline）
        assert summary["content_json"]["_meta"]["origin"] == "resummarize"


def test_resummarize_concurrent_single_winner(tmp_path, monkeypatch) -> None:
    """并发锁：done→summarizing 是原子条件更新，同会议重复请求只有一个入队。

    把后台任务换成不复位状态的桩，模拟第一个请求的重跑仍在进行中——
    第二个请求必须拿到 409 且不入队（原子迁移由 UPDATE ... WHERE status='done'
    保证，读-判-写实现会让两个请求都通过检查）。
    """
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    import app.api.routes.meetings as meetings_mod

    queued: list = []

    async def fake_resummarize(meeting_id) -> None:
        queued.append(meeting_id)  # 不复位状态：模拟任务尚未跑完

    with TestClient(app) as client:
        headers = auth_headers(client)
        meeting_id = _upload(client, headers, title="并发重跑", filename="cc.wav").json()["id"]

        monkeypatch.setattr(meetings_mod, "run_resummarize", fake_resummarize)
        first = client.post(f"/api/meetings/{meeting_id}/resummarize", headers=headers)
        second = client.post(f"/api/meetings/{meeting_id}/resummarize", headers=headers)

        assert sorted([first.status_code, second.status_code]) == [200, 409]
        assert len(queued) == 1  # 只有拿到锁的请求真正入队
        body = client.get(f"/api/meetings/{meeting_id}", headers=headers).json()
        assert body["status"] == "summarizing"


def test_resummarize_only_allowed_when_done(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    import app.services.pipeline as pipeline_mod

    async def boom(session, meeting, **kwargs):
        raise RuntimeError("summarize down")

    monkeypatch.setattr(pipeline_mod, "summarize_meeting", boom)
    with TestClient(app) as client:
        headers = auth_headers(client)
        meeting_id = _upload(client, headers, title="失败会", filename="f.wav").json()["id"]
        body = client.get(f"/api/meetings/{meeting_id}", headers=headers).json()
        assert body["status"] == "failed"

        resp = client.post(f"/api/meetings/{meeting_id}/resummarize", headers=headers)
        assert resp.status_code == 409


def test_resummarize_failure_keeps_done_and_old_summary(tmp_path, monkeypatch) -> None:
    """重跑失败不能吞掉旧纪要：状态回 done，错误存 error_message。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    import app.services.pipeline as pipeline_mod

    with TestClient(app) as client:
        headers = auth_headers(client)
        meeting_id = _upload(client, headers, title="重跑失败", filename="rf.wav").json()["id"]

        async def boom(session, meeting, **kwargs):
            raise RuntimeError("llm outage")

        monkeypatch.setattr(pipeline_mod, "summarize_meeting", boom)
        resp = client.post(f"/api/meetings/{meeting_id}/resummarize", headers=headers)
        assert resp.status_code == 200, resp.text

        body = client.get(f"/api/meetings/{meeting_id}", headers=headers).json()
        assert body["status"] == "done"
        assert "llm outage" in body["error_message"]
        summary = client.get(
            f"/api/meetings/{meeting_id}/summary", headers=headers
        ).json()
        assert summary["version"] == 1  # 旧版纪要原样保留
