"""Bearer Token 鉴权（PRD §9.2）。"""
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from tests.conftest import auth_headers


def test_health_is_public() -> None:
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200


def test_token_rejects_wrong_password() -> None:
    with TestClient(app) as client:
        resp = client.post("/api/auth/token", json={"password": "wrong"})
        assert resp.status_code == 401


def test_endpoints_require_token() -> None:
    with TestClient(app) as client:
        assert client.get("/api/meetings").status_code == 401
        resp = client.get(
            "/api/meetings", headers={"Authorization": "Bearer not-a-jwt"}
        )
        assert resp.status_code == 401


def test_valid_token_grants_access(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.get("/api/meetings", headers=headers)
        # 数据库不可用时报 500 以外的错误不在本测试范围；只验证鉴权已通过
        assert resp.status_code != 401
