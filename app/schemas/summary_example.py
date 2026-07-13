import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# 单条范例长度上限：远超 few-shot 预算的范例存了也用不上，入口直接拦截
MAX_EXAMPLE_CHARS = 20_000


class SummaryExampleIn(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=MAX_EXAMPLE_CHARS)


class SummaryExampleUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    content: str | None = Field(
        default=None, min_length=1, max_length=MAX_EXAMPLE_CHARS
    )
    enabled: bool | None = None


class SummaryExampleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    content: str
    source_meeting_id: uuid.UUID | None
    enabled: bool
    created_at: datetime
    updated_at: datetime
