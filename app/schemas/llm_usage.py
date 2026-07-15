"""LLM 用量统计响应模型（Tech Design M4 §11/§12，Phase 2）。"""
import uuid
from datetime import date, datetime

from pydantic import BaseModel


class ProviderStat(BaseModel):
    provider: str
    calls: int
    success_calls: int
    success_rate: float
    fallback_calls: int  # fallback_index > 0：作为降级目标被调用的次数
    fallback_rate: float
    input_tokens: int
    output_tokens: int
    total_tokens: int


class TaskStat(BaseModel):
    task_type: str
    calls: int
    success_calls: int
    success_rate: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    avg_latency_ms: int | None = None


class GrantStatus(BaseModel):
    """单个资源包的应用侧估算状态（预警用，控制台余额是最终真值）。"""

    name: str  # "glm_air" | "glm_general"
    providers: list[str]  # 计入该资源包的 provider 名单
    grant_total_tokens: int
    tracked_total_tokens: int
    estimated_remaining_tokens: int
    usage_ratio: float
    usage_level: str | None = None  # §12.2：observe/evaluate_routing/high_value_only/exhausted
    avg_tokens_per_meeting: int | None = None
    estimated_remaining_meetings: int | None = None
    avg_daily_tokens_7d: int | None = None
    estimated_exhaustion_date: date | None = None
    expires_at: datetime
    days_until_expiry: int
    expiry_warning: str | None = None  # §12.1：到期前 30/14/7/1 天提醒文案


class FailureOut(BaseModel):
    created_at: datetime
    meeting_id: uuid.UUID | None = None
    task_type: str
    provider: str
    model: str
    fallback_index: int
    error_type: str | None = None
    error_message: str | None = None


class UsageStatsOut(BaseModel):
    total_calls: int
    total_tokens: int
    by_provider: list[ProviderStat]
    by_task: list[TaskStat]
    grants: list[GrantStatus]
    recent_failures: list[FailureOut]
