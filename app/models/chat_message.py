import uuid
from typing import Any

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPkMixin


class ChatMessage(UUIDPkMixin, TimestampMixin, Base):
    """二期 AI 问答暗桩表；回答强制携带 cited_segment_ids（PRD Feature 5）。"""

    __tablename__ = "chat_messages"

    meeting_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    cited_segment_ids: Mapped[list[Any] | None] = mapped_column(JSONB)
