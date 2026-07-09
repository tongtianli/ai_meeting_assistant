import uuid

from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPkMixin
from app.models.enums import ConfirmedBy, enum_column


class SpeakerBinding(UUIDPkMixin, TimestampMixin, Base):
    """Speaker→Person 映射，追加式、可审计、可回滚（PRD §5）。"""

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
