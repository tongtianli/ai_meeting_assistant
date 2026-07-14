import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QaAnswer(BaseModel):
    """LLM 回答的结构化输出（经 schema 校验，PRD Feature 5）。"""

    answer: str
    cited_segment_seqs: list[int] = []


class IntentOut(BaseModel):
    """聊天意图路由（PRD §7.1）：查询 or 修改纪要。"""

    intent: Literal["query", "edit"]


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
    # 本轮改纪要产出的新版本号；仅 POST 响应携带（历史消息文本中已含版本号）
    summary_version: int | None = None
    created_at: datetime
