import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import ConfirmedBy, MeetingStatus


class MeetingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    status: MeetingStatus
    duration: float | None
    error_message: str | None
    location: str | None = None
    host: str | None = None
    recorder: str | None = None
    importance: str | None = None
    created_at: datetime


class SegmentOut(BaseModel):
    seq: int
    start_time: float
    end_time: float
    speaker_label: str
    speaker_name: str  # 生效绑定的真名；未绑定时回退为 speaker_label
    person_id: uuid.UUID | None
    text: str


class TranscriptOut(BaseModel):
    meeting_id: uuid.UUID
    segments: list[SegmentOut]


class AudioUrlOut(BaseModel):
    url: str
    expires_in: int


class SpeakerRenameIn(BaseModel):
    speaker_label: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=255)


class SpeakerBindingOut(BaseModel):
    id: uuid.UUID
    speaker_label: str
    person_id: uuid.UUID
    person_name: str
    confirmed_by: ConfirmedBy
    confidence: float | None
