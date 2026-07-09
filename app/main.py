from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.meetings import router as meetings_router

app = FastAPI(title="AI Meeting Assistant")
app.include_router(health_router, prefix="/api")
app.include_router(meetings_router, prefix="/api")
