"""服务重启后，中断的会议应标记为 failed 以便重试（BackgroundTasks 随进程消亡）。"""
import asyncio

from fastapi.testclient import TestClient

from app.db.session import SessionLocal
from app.main import app
from app.models import DEFAULT_USER_ID, Meeting, MeetingStatus
from tests.conftest import requires_db

pytestmark = requires_db


def test_in_flight_meetings_marked_failed_on_startup() -> None:
    async def create_orphan() -> object:
        async with SessionLocal() as session:
            meeting = Meeting(
                user_id=DEFAULT_USER_ID,
                title="重启孤儿",
                status=MeetingStatus.summarizing,
                audio_url="local://orphan.wav",
            )
            session.add(meeting)
            await session.commit()
            return meeting.id

    mid = asyncio.run(create_orphan())
    try:
        with TestClient(app):  # 进入 with 触发 lifespan（模拟服务启动）
            pass

        async def check() -> None:
            async with SessionLocal() as session:
                meeting = await session.get(Meeting, mid)
                assert meeting.status == MeetingStatus.failed
                assert "重启中断" in meeting.error_message

        asyncio.run(check())
    finally:
        async def cleanup() -> None:
            async with SessionLocal() as session:
                meeting = await session.get(Meeting, mid)
                if meeting is not None:
                    await session.delete(meeting)
                    await session.commit()

        asyncio.run(cleanup())
