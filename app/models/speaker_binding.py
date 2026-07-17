import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPkMixin
from app.models.enums import ConfirmedBy, enum_column


class SpeakerBinding(UUIDPkMixin, TimestampMixin, Base):
    """Speaker→Person 映射，追加式、可审计、可回滚（PRD §5）。

    生效判定 = superseded_by IS NULL 且 revoked_at IS NULL：
    改绑走 superseded_by（有后继），撤销走 revoked_at（无后继、审计行保留，
    声纹设计 §4.1）。"""

    __tablename__ = "speaker_bindings"

    meeting_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    speaker_label: Mapped[str] = mapped_column(String(64), nullable=False)
    person_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    confirmed_by: Mapped[ConfirmedBy] = mapped_column(
        enum_column(ConfirmedBy, "binding_confirmed_by"), nullable=False
    )
    confidence: Mapped[float | None] = mapped_column(Float)
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("speaker_bindings.id", ondelete="SET NULL")
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
