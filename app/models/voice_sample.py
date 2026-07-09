import uuid

from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.db.base import Base, TimestampMixin, UUIDPkMixin


class VoiceSample(UUIDPkMixin, TimestampMixin, Base):
    """一期暗桩：diarization 产出的 speaker embedding 入库留存（person_id 暂空、带模型版本），
    为二期声纹避免全量重跑历史音频（PRD §3 Feature 2）。
    删除 Person 时物理级联删除全部 VoiceSample 与 embedding（PRD §5 / §9.4）。"""

    __tablename__ = "voice_samples"

    person_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("persons.id", ondelete="CASCADE"), index=True
    )
    source_meeting_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    sample_start: Mapped[float] = mapped_column(Float, nullable=False)  # 秒
    sample_end: Mapped[float] = mapped_column(Float, nullable=False)
