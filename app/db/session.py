from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import settings

# NullPool：asyncpg 连接绑定事件循环，不跨请求池化，
# 避免 BackgroundTasks/测试场景下跨 loop 复用失效连接；MVP 流量下开销可忽略
engine = create_async_engine(settings.database_url, poolclass=NullPool)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session
