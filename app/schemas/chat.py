import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class QaAnswer(BaseModel):
    """LLM 回答的结构化输出（经 schema 校验，PRD Feature 5）。"""

    answer: str
    cited_segment_seqs: list[int] = []


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class CitationOut(BaseModel):
    """回答引用的 segment，反查真实文本/时间戳供前端跳播。"""

    seq: int
    start_time: float
    speaker_name: str
    text: str


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str  # user | assistant
    content: str
    citations: list[CitationOut] = []
    created_at: datetime
