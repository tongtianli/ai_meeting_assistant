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
    # 云端声纹库 ID（火山控制台「声纹管理」注册后回填）；
    # 删除 Person 时云端样本需同步删除（PIPL 级联，注册 API 打通前为人工步骤）
    voiceprint_id: Mapped[str | None] = mapped_column(String(128), unique=True)
