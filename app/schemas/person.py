import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PersonIn(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    # 火山控制台「声纹管理」注册样本后得到的 ID；登记后转写自动做声纹匹配
    voiceprint_id: str | None = Field(default=None, max_length=128)
    # 声纹属 PIPL 敏感个人信息：登记 voiceprint_id 时应记录本人同意
    consent_note: str | None = Field(default=None, max_length=2000)


class PersonUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    voiceprint_id: str | None = Field(default=None, max_length=128)
    consent_note: str | None = Field(default=None, max_length=2000)


class PersonOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    voiceprint_id: str | None
    consent_record: dict | None
    created_at: datetime


class PersonDeleteOut(BaseModel):
    deleted: bool
    # 云端声纹注册 API 打通前，物理级联删除需人工在控制台完成（PIPL）
    cloud_cleanup_required: bool
    message: str
