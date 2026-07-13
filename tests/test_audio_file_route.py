"""签名文件路由：云端 ASR 经公网拉取派生音频（URL 模式）。"""
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.security import create_file_token
from app.main import app


def test_signed_file_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    target = tmp_path / "transcoded" / "x.upload.mp3"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"fake-mp3-bytes")

    token = create_file_token("transcoded/x.upload.mp3", ttl_seconds=60)
    with TestClient(app) as client:
        resp = client.get(f"/api/audio/file/{token}")
        assert resp.status_code == 200
        assert resp.content == b"fake-mp3-bytes"
        assert resp.headers["content-type"].startswith("audio/mpeg")


def test_signed_file_rejects_garbage_and_expired(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/audio/file/not-a-jwt").status_code == 403

        expired = create_file_token("transcoded/x.mp3", ttl_seconds=-10)
        assert client.get(f"/api/audio/file/{expired}").status_code == 403


def test_signed_file_blocks_path_traversal(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    (tmp_path / "data").mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("do-not-serve")

    # 即使 token 合法签发，data_dir 之外的路径也必须拒绝（纵深防御）
    token = create_file_token("../secret.txt", ttl_seconds=60)
    with TestClient(app) as client:
        assert client.get(f"/api/audio/file/{token}").status_code == 404
