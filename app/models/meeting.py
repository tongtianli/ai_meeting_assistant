import uuid

from sqlalchemy import Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPkMixin
from app.models.enums import MeetingStatus, enum_column


class Meeting(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "meetings"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[MeetingStatus] = mapped_column(
        enum_column(MeetingStatus, "meeting_status"),
        nullable=False,
        default=MeetingStatus.uploaded,
        server_default=MeetingStatus.uploaded.value,
    )
    duration: Mapped[float | None] = mapped_column(Float)  # 秒
    audio_url: Mapped[str | None] = mapped_column(String(1024))
    error_message: Mapped[str | None] = mapped_column(Text)
    # 公司纪要模板抬头字段（上传时选填；空则导出留白，打印后手写）
    location: Mapped[str | None] = mapped_column(String(255))
    host: Mapped[str | None] = mapped_column(String(255))
    recorder: Mapped[str | None] = mapped_column(String(255))
    importance: Mapped[str | None] = mapped_column(String(32))  # 一般/重要/加急
    # 本会议 segment 向量的生成模型身份（"provider/model:dim"，RAG 设计 §4.1.1-G）。
    # 同维不同模型的向量空间不通，只比维度会静默错检索——身份不符即整场重嵌。
    # 空 = 旧数据或尚未嵌入，首次问答时重嵌并回填
    embedding_model_key: Mapped[str | None] = mapped_column(String(128))
