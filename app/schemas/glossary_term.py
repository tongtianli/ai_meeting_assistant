import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class GlossaryTermsBulkIn(BaseModel):
    """批量录入：前端把 textarea 按换行/逗号拆成多条一次提交。"""

    terms: list[str] = Field(min_length=1)


class GlossaryTermUpdate(BaseModel):
    enabled: bool | None = None


class GlossaryTermOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    term: str
    enabled: bool
    created_at: datetime
    updated_at: datetime
