import uuid

from sqlalchemy import Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.db.base import Base, TimestampMixin, UUIDPkMixin


class TranscriptSegment(UUIDPkMixin, TimestampMixin, Base):
    """逐行存储，不存大 JSON（PRD §5）。"""

    __tablename__ = "transcript_segments"
    __table_args__ = (
        UniqueConstraint("meeting_id", "seq", name="uq_transcript_segments_meeting_seq"),
    )

    meeting_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)  # 秒
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    speaker_label: Mapped[str] = mapped_column(String(64), nullable=False)  # "speaker_001"
    # 暗桩：全局身份，绑定后物化
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("persons.id", ondelete="SET NULL"), index=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # 暗桩：二期问答用，一期可空；维度随 embedding 模型定，暂不约束
    embedding: Mapped[list[float] | None] = mapped_column(Vector())
