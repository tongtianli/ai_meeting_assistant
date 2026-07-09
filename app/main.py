from fastapi import FastAPI

from app.api.routes.audio import router as audio_router
from app.api.routes.auth import router as auth_router
from app.api.routes.health import router as health_router
from app.api.routes.meetings import router as meetings_router

app = FastAPI(title="AI Meeting Assistant")
app.include_router(health_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(meetings_router, prefix="/api")
app.include_router(audio_router, prefix="/api")
