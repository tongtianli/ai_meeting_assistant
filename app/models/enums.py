import enum

from sqlalchemy import Enum as SAEnum


class AuthIdentityType(str, enum.Enum):
    email = "email"
    wechat_unionid = "wechat_unionid"
    phone = "phone"


class MeetingStatus(str, enum.Enum):
    uploaded = "uploaded"
    transcoding = "transcoding"
    transcribing = "transcribing"
    summarizing = "summarizing"
    done = "done"
    failed = "failed"


class ConfirmedBy(str, enum.Enum):
    auto = "auto"
    human = "human"


def enum_column(enum_cls: type[enum.Enum], name: str) -> SAEnum:
    """VARCHAR + CHECK 约束（非原生 PG enum），后续加枚举值只需改约束，迁移简单。"""
    return SAEnum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        values_callable=lambda e: [m.value for m in e],
    )
