import uuid

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPkMixin

# MVP 单默认用户（PRD §6：多用户注册登录不在一期范围，user_id 先写死）
DEFAULT_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


class User(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "users"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
