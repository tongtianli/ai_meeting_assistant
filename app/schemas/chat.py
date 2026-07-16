import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QaAnswer(BaseModel):
    """LLM 回答的结构化输出（经 schema 校验，PRD Feature 5）。

    insufficient_evidence（RAG 设计 §4.1.1-F）：证据不足时显式拒答；
    程序侧规则——事实性回答（false）必须带 ≥1 条本轮 context 内的合法引用。
    confidence（§6.1）：模型自评证据充分度；拒答/回退路径程序强制 low。
    """

    answer: str
    cited_segment_seqs: list[int] = []
    insufficient_evidence: bool = False
    confidence: Literal["high", "medium", "low"] = "medium"


class ChatIntent(BaseModel):
    """聊天意图 + 多轮独立检索问题（一次 LLM 调用同时产出，§4.1.1-A）。

    standalone_query 缺省为 None：无历史/模型未改写时回退原问题。
    """

    intent: Literal["query", "edit"]
    standalone_query: str | None = None


# 兼容旧名（intent-only 场景仍可用）
IntentOut = ChatIntent


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
    # 本轮回答的证据置信度（§6.1）；仅 POST 响应携带，历史消息不回填
    confidence: str | None = None
    created_at: datetime
