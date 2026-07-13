import uuid

from sqlalchemy import Boolean, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPkMixin


class SummaryExample(UUIDPkMixin, TimestampMixin, Base):
    """纪要范例库（公司文风第二层）：优质纪要存为范例，
    摘要时作为 few-shot 注入 prompt，让 LLM 模仿真实文风。

    范例来源两种：1) 已生成纪要「采纳为范例」；2) 手工粘贴既有公司纪要。
    """

    __tablename__ = "summary_examples"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    # 公司格式的纪要正文纯文本（few-shot 直接注入的内容）
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 采纳自某次会议时记录来源；会议删除后范例保留（SET NULL）
    source_meeting_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("meetings.id", ondelete="SET NULL")
    )
    # 关闭后不参与 few-shot，便于试验对比不同范例组合
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
