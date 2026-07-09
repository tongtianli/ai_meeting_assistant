import uuid

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPkMixin


class ActionItem(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "action_items"

    meeting_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    task: Mapped[str] = mapped_column(Text, nullable=False)
    owner_person_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("persons.id", ondelete="SET NULL")
    )
    owner_text: Mapped[str | None] = mapped_column(String(255))  # 未绑定 Person 时的原文归属
    # LLM 抽取结果可能是「下周五」等模糊表述，存原文
    deadline: Mapped[str | None] = mapped_column(String(255))
    # 溯源要求（PRD 设计原则 1）：TODO 绑定来源 segment
    source_segment_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("transcript_segments.id", ondelete="SET NULL"), index=True
    )
