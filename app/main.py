from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.audio import router as audio_router
from app.api.routes.auth import router as auth_router
from app.api.routes.health import router as health_router
from app.api.routes.meetings import router as meetings_router

app = FastAPI(title="AI Meeting Assistant")
app.include_router(health_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(meetings_router, prefix="/api")
app.include_router(audio_router, prefix="/api")

# 生产部署：构建产物存在时由 FastAPI 同源托管（无 CORS 问题）；
# 开发时用 vite dev server（/api 代理到本服务）
_WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
if _WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
