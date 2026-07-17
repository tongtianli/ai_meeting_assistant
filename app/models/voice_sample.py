import uuid

from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.db.base import Base, TimestampMixin, UUIDPkMixin


class VoiceSample(UUIDPkMixin, TimestampMixin, Base):
    """diarization 产出的 speaker embedding 留存（带模型版本），声纹记忆的数据底座。

    归属规则（声纹设计 §4.1）：person_id/assigned_by_binding_id 由 human 确认的
    SpeakerBinding 物化；撤销绑定时回无主池（两列置空）不物理删除；
    物理删除仅随删除 Person（CASCADE）或删除会议发生。
    「长期参考样本」= person_id 非空且 assigned_by_binding_id 指向
    confirmed_by=human 的有效绑定——跨会议匹配只比对长期参考样本。"""

    __tablename__ = "voice_samples"

    person_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("persons.id", ondelete="CASCADE"), index=True
    )
    source_meeting_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # 会议内说话人标签（如 speaker_001）：绑定→样本归属的关联键。
    # 可空：存量数据按时间重叠 best-effort 回填，歧义留空（迁移 0009）
    speaker_label: Mapped[str | None] = mapped_column(String(64))
    # 归属来源（可审计、可回滚）：指向产生本次归属的 SpeakerBinding；
    # 绑定删除时置空（样本归属随撤销逻辑处理，FK 仅兜底）
    assigned_by_binding_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("speaker_bindings.id", ondelete="SET NULL")
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    sample_start: Mapped[float] = mapped_column(Float, nullable=False)  # 秒
    sample_end: Mapped[float] = mapped_column(Float, nullable=False)

    @property
    def duration(self) -> float:
        return self.sample_end - self.sample_start
