import asyncio
import wave
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings

# ---- 测试库隔离（必须在导入 app.db.session 建 engine 之前执行）----
# 测试永远跑在独立的 <主库名>_test 库：不污染开发库；
# 每轮测试开始时整库清空重置（见 _reset_test_db）
_BASE_URL, _DB_NAME = settings.database_url.rsplit("/", 1)
if not _DB_NAME.endswith("_test"):
    _DB_NAME = f"{_DB_NAME}_test"
    settings.database_url = f"{_BASE_URL}/{_DB_NAME}"

from app.db.session import engine  # noqa: E402  engine 已绑定测试库


def make_wav(path: Path, seconds: float = 2.0, rate: int = 16000) -> Path:
    """生成一段静音 wav 测试音频。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


async def _probe() -> None:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


def _ensure_test_db() -> bool:
    """测试库可用性：不存在则创建并迁移到最新 schema；实例不可达则跳过 DB 测试。"""
    try:
        asyncio.run(_probe())
        exists = True
    except Exception as exc:
        if "does not exist" not in str(exc):
            return False  # PostgreSQL 实例不可达
        exists = False

    if not exists:
        admin = create_async_engine(
            f"{_BASE_URL}/postgres", isolation_level="AUTOCOMMIT", poolclass=NullPool
        )

        async def _create() -> None:
            async with admin.connect() as conn:
                await conn.execute(text(f'CREATE DATABASE "{_DB_NAME}"'))
            await admin.dispose()

        try:
            asyncio.run(_create())
        except Exception:
            return False

    # 迁移到最新（幂等）；alembic env.py 读 settings.database_url，已指向测试库
    from alembic import command
    from alembic.config import Config

    ini = Path(__file__).resolve().parent.parent / "alembic.ini"
    command.upgrade(Config(str(ini)), "head")
    return True


_DB_AVAILABLE: bool | None = None


def db_available() -> bool:
    global _DB_AVAILABLE
    if _DB_AVAILABLE is None:
        _DB_AVAILABLE = _ensure_test_db()
    return _DB_AVAILABLE


requires_db = pytest.mark.skipif(not db_available(), reason="database not reachable")


@pytest.fixture(scope="session", autouse=True)
def _reset_test_db():
    """每轮测试开始前整库清空重置（users 级联清掉全部业务数据后重新播种）。

    在会话开始而非结束时清理：失败现场保留在测试库里可供排查。
    """
    if db_available():

        async def _wipe() -> None:
            from app.models import DEFAULT_USER_ID

            async with engine.begin() as conn:
                await conn.execute(text("DELETE FROM users"))
                await conn.execute(
                    text(
                        "INSERT INTO users (id, name, created_at, updated_at) "
                        "VALUES (:id, 'Default User', now(), now())"
                    ),
                    {"id": str(DEFAULT_USER_ID)},
                )

        asyncio.run(_wipe())
    yield


def auth_headers(client) -> dict[str, str]:
    """用访问口令换 JWT，返回带 Bearer 的请求头。"""
    resp = client.post("/api/auth/token", json={"password": settings.auth_password})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture(autouse=True)
def _mock_providers(monkeypatch):
    """测试环境统一走 mock provider，不受本机 .env 配置影响，
    避免真实 ASR 推理与 LLM 网络调用。"""
    monkeypatch.setattr(settings, "llm_providers", "mock")
    monkeypatch.setattr(settings, "asr_provider", "mock")
