import uuid

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPkMixin


class SpeakerIdentityDismissal(UUIDPkMixin, TimestampMixin, Base):
    """确认卡「跳过」记录（声纹设计 §5，决议 4）：独立轻量表，
    不复用 SpeakerBinding（其语义保持"真实人物绑定"）。

    pending 状态不持久化，动态计算（见 services.voiceprints.speaker_identity_status）：
    有有效绑定 → 已识别；无绑定有 dismissal → 已跳过；两者皆无 → 未知（出确认卡）。
    手动绑定时删除对应 dismissal，之后撤销绑定确认卡自然重新出现。
    dismissed_at 即 created_at（TimestampMixin）。"""

    __tablename__ = "speaker_identity_dismissals"
    __table_args__ = (
        UniqueConstraint(
            "meeting_id", "speaker_label", name="uq_speaker_dismissal_meeting_label"
        ),
    )

    meeting_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    speaker_label: Mapped[str] = mapped_column(String(64), nullable=False)
