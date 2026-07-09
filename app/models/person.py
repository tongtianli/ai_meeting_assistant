import uuid
from typing import Any

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPkMixin


class Person(UUIDPkMixin, TimestampMixin, Base):
    """全局身份，与会议内 Speaker 标签分离（PRD 设计原则 3）。"""

    __tablename__ = "persons"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # 声纹属敏感个人信息，需单独同意记录（PRD §9.4）
    consent_record: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
