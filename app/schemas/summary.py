"""Summary 结构化 JSON 的 schema（LLM 输出经此校验，PRD Feature 3）。"""
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TodoItem(BaseModel):
    task: str = Field(min_length=1)
    owner: str | None = None
    deadline: str | None = None
    source_segment_seq: int | None = None


class TopicGroup(BaseModel):
    """议题分组（公司纪要正文格式：一、议题（责任人：X）+ 编号条目）。"""

    title: str = Field(min_length=1)
    owner: str | None = None  # 责任人，可多人顿号分隔
    items: list[str] = []


class SummaryContent(BaseModel):
    title: str
    participants: list[str] = []
    summary: str
    # 新版正文主体；discussions 保留兼容旧版已存 JSON（导出时二选一）
    topics: list[TopicGroup] = []
    discussions: list[str] = []
    decisions: list[str] = []
    todos: list[TodoItem] = []


class MapFacts(BaseModel):
    """长会议 map 阶段的精简事实抽取产物（Tech Design M4 §6.2/Phase 4）。

    map 只做事实抽取、不写 title/participants/summary 叙述——省输出 token；
    正式文风与叙述由 reduce 阶段（glm_air + 范文）统一定型。reduce 消费的是
    各块 MapFacts（而非完整纪要），最终仍产出 SummaryContent。
    """

    topics: list[TopicGroup] = []
    decisions: list[str] = []
    todos: list[TodoItem] = []
    risks: list[str] = []
    open_questions: list[str] = []
    source_segment_seqs: list[int] = []


class SummaryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    meeting_id: uuid.UUID
    version: int
    content_json: dict[str, Any]
    created_at: datetime
