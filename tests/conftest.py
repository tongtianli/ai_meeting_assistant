import asyncio
import wave
from pathlib import Path

import pytest
from sqlalchemy import text

from app.db.session import engine


def make_wav(path: Path, seconds: float = 2.0, rate: int = 16000) -> Path:
    """生成一段静音 wav 测试音频。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


def db_available() -> bool:
    async def _probe() -> None:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    try:
        asyncio.run(_probe())
        return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(not db_available(), reason="database not reachable")


def auth_headers(client) -> dict[str, str]:
    """用访问口令换 JWT，返回带 Bearer 的请求头。"""
    from app.core.config import settings

    resp = client.post("/api/auth/token", json={"password": settings.auth_password})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}
