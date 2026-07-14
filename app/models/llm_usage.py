import uuid

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPkMixin


class LlmUsageRecord(UUIDPkMixin, TimestampMixin, Base):
    """LLM/embedding 调用审计（Tech Design M4 §11，Phase 2）。

    成功与失败都记；meeting_id/user_id 为普通列而非外键——成本审计
    必须在会议删除后存活。provider 不返回 usage 时 token 字段留空，
    不伪造数字。
    """

    __tablename__ = "llm_usage_records"

    meeting_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True))
    # LLMTaskType 值或 "embedding"；字符串列，加新任务无需迁移
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    # 该 provider 在本次任务尝试链中的序号（0=首选，>0=发生了 fallback）
    fallback_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_type: Mapped[str | None] = mapped_column(String(32))
    error_message: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
