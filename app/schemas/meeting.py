import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.enums import MeetingStatus


class MeetingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    status: MeetingStatus
    duration: float | None
    error_message: str | None
    created_at: datetime
