import uuid

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPkMixin
from app.models.enums import AuthIdentityType, enum_column


class AuthIdentity(UUIDPkMixin, TimestampMixin, Base):
    """身份与登录方式分离；微信侧按 UnionID 设计（PRD §5）。"""

    __tablename__ = "auth_identities"
    __table_args__ = (
        UniqueConstraint("type", "identifier", name="uq_auth_identities_type_identifier"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    type: Mapped[AuthIdentityType] = mapped_column(
        enum_column(AuthIdentityType, "auth_identity_type"), nullable=False
    )
    identifier: Mapped[str] = mapped_column(String(255), nullable=False)
    credential: Mapped[str | None] = mapped_column(String(255))
