"""Person 管理 API：声纹 ID 登记、同意记录（默认可选，可开关强制）、删除的清理提示。"""
import uuid

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from tests.conftest import auth_headers, requires_db

pytestmark = requires_db


def _vp() -> str:
    return f"vp-test-{uuid.uuid4().hex[:12]}"


def test_person_crud_and_voiceprint_flow() -> None:
    with TestClient(app) as client:
        headers = auth_headers(client)
        created: list[str] = []
        try:
            # 无声纹的普通人员：不要求同意记录
            resp = client.post(
                "/api/persons", json={"name": "普通同事"}, headers=headers
            )
            assert resp.status_code == 201, resp.text
            plain = resp.json()
            created.append(plain["id"])
            assert plain["voiceprint_id"] is None

            # 登记声纹不再强制同意记录（声纹设计 §1，决议 1）→ 201
            resp = client.post(
                "/api/persons",
                json={"name": "无同意记录", "voiceprint_id": _vp()},
                headers=headers,
            )
            assert resp.status_code == 201, resp.text
            no_consent = resp.json()
            created.append(no_consent["id"])
            assert no_consent["consent_record"] is None

            # 声纹 + 同意记录 → 201，consent_record 带时间戳
            vp_id = _vp()
            resp = client.post(
                "/api/persons",
                json={
                    "name": "小王",
                    "voiceprint_id": vp_id,
                    "consent_note": "2026-07-12 口头同意，样本由本人提供",
                },
                headers=headers,
            )
            assert resp.status_code == 201, resp.text
            wang = resp.json()
            created.append(wang["id"])
            assert wang["voiceprint_id"] == vp_id
            assert wang["consent_record"]["recorded_at"]

            # 声纹 ID 唯一：重复登记 → 409
            resp = client.post(
                "/api/persons",
                json={"name": "冒名者", "voiceprint_id": vp_id, "consent_note": "x"},
                headers=headers,
            )
            assert resp.status_code == 409

            # 列表包含两人
            resp = client.get("/api/persons", headers=headers)
            names = [p["name"] for p in resp.json()]
            assert "普通同事" in names and "小王" in names

            # 改名
            resp = client.patch(
                f"/api/persons/{wang['id']}",
                json={"name": "王工"},
                headers=headers,
            )
            assert resp.status_code == 200
            assert resp.json()["name"] == "王工"

            # 删除带声纹的人员 → 提示云端样本需人工清理（PIPL）
            resp = client.delete(f"/api/persons/{wang['id']}", headers=headers)
            assert resp.status_code == 200
            body = resp.json()
            assert body["cloud_cleanup_required"] is True
            assert vp_id in body["message"]
            created.remove(wang["id"])

            # 删除无声纹人员 → 无需云端清理
            resp = client.delete(f"/api/persons/{plain['id']}", headers=headers)
            assert resp.json()["cloud_cleanup_required"] is False
            created.remove(plain["id"])
        finally:
            for pid in created:
                client.delete(f"/api/persons/{pid}", headers=headers)


def test_person_requires_auth() -> None:
    with TestClient(app) as client:
        assert client.get("/api/persons").status_code in (401, 403)


def test_voiceprint_consent_enforced_by_flag(monkeypatch) -> None:
    """VOICEPRINT_REQUIRE_CONSENT=true 时恢复强制（多用户/商业部署预留）。"""
    monkeypatch.setattr(settings, "voiceprint_require_consent", True)
    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.post(
            "/api/persons",
            json={"name": "需同意", "voiceprint_id": _vp()},
            headers=headers,
        )
        assert resp.status_code == 422
