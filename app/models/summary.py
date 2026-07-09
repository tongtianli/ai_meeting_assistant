import uuid
from typing import Any

from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPkMixin


class Summary(UUIDPkMixin, TimestampMixin, Base):
    """版本化：二期聊天修改纪要时生成新版本而非覆盖（PRD Feature 6）。"""

    __tablename__ = "summaries"
    __table_args__ = (
        UniqueConstraint("meeting_id", "version", name="uq_summaries_meeting_version"),
    )

    meeting_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    style_profile_id: Mapped[str | None] = mapped_column(String(128))
    content_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
