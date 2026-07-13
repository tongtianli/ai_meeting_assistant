import uuid

from sqlalchemy import Boolean, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPkMixin


class GlossaryTerm(UUIDPkMixin, TimestampMixin, Base):
    """全局术语表（跨会议热词）：per-user 的术语/人名/专名列表，
    转写阶段作为 hotwords 注入 ASR（provider 无关，PRD Feature 1）。

    一次维护、所有会议受益；切换 ASR provider 同一份术语表继续生效。
    """

    __tablename__ = "glossary_terms"
    __table_args__ = (
        UniqueConstraint("user_id", "term", name="uq_glossary_terms_user_term"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    term: Mapped[str] = mapped_column(String(255), nullable=False)
    # 关闭后不注入，便于超过 provider 上限时挑选生效子集
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
