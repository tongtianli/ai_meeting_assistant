import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import update

from app.api.routes.audio import router as audio_router
from app.api.routes.auth import router as auth_router
from app.api.routes.examples import router as examples_router
from app.api.routes.glossary import router as glossary_router
from app.api.routes.health import router as health_router
from app.api.routes.meetings import router as meetings_router
from app.api.routes.persons import router as persons_router
from app.db.session import SessionLocal
from app.models import Meeting, MeetingStatus

logger = logging.getLogger(__name__)

_IN_FLIGHT = (
    MeetingStatus.uploaded,
    MeetingStatus.transcoding,
    MeetingStatus.transcribing,
    MeetingStatus.summarizing,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # MVP 管道跑在 BackgroundTasks 里，随进程消亡：启动时把上次中断的
    # 会议标记为 failed，前端出现重试按钮（二期换 Celery 后此逻辑移除）
    try:
        async with SessionLocal() as session:
            result = await session.execute(
                update(Meeting)
                .where(Meeting.status.in_(_IN_FLIGHT))
                .values(
                    status=MeetingStatus.failed,
                    error_message="处理被服务重启中断，请点击重试",
                )
            )
            await session.commit()
            if result.rowcount:
                logger.warning(
                    "recovered %d meeting(s) interrupted by restart", result.rowcount
                )
    except Exception:
        # 数据库暂不可用不阻塞启动（如无库跑测试）
        logger.warning("startup recovery skipped: database unavailable", exc_info=True)
    yield


app = FastAPI(title="AI Meeting Assistant", lifespan=lifespan)
app.include_router(health_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(meetings_router, prefix="/api")
app.include_router(persons_router, prefix="/api")
app.include_router(examples_router, prefix="/api")
app.include_router(glossary_router, prefix="/api")
app.include_router(audio_router, prefix="/api")

# 生产部署：构建产物存在时由 FastAPI 同源托管（无 CORS 问题）；
# 开发时用 vite dev server（/api 代理到本服务）
_WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
if _WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
