"""额度软限制与付费熔断（Tech Design M4 §12.3，Phase 3）。

- 配额快照 = llm_usage_records 的账号级累计（与资源包口径一致，跨用户），
  带 TTL 缓存避免每次 LLM 调用查库；查库失败 fail-open（不阻断业务）
- 判定是纯函数、每次调用即时求值（配置/到期时间不进缓存）：
  * 软上限（默认 95%）：glm_air 从低价值任务路由移除，仅供最终纪要/纪要编辑
  * 硬上限（100%）或资源包到期：GLM_ALLOW_PAID_AFTER_GRANT=false 时
    全任务停用 glm_air，禁止静默付费
- 应用侧估算不能完全阻止服务商计费：必须同时在智谱控制台配置余额预警
"""
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select

from app.core.config import settings
from app.services.llm.base import LLMProvider, LLMTaskType

logger = logging.getLogger(__name__)

# Air 优先的高价值任务（§8/§12.2：软上限后仅这些任务允许用 Air）
HIGH_VALUE_TASKS = frozenset(
    {LLMTaskType.SUMMARY_FINAL.value, LLMTaskType.SUMMARY_EDIT.value}
)

# 计入 Air 资源包的 provider（与 llm_usage 统计口径一致）
_AIR_PROVIDERS = ("glm_air",)


@dataclass(frozen=True)
class QuotaSnapshot:
    """账号级累计消耗快照（token 数来自审计表，仅应用侧估算）。"""

    air_tracked: int


_cache: tuple[float, QuotaSnapshot] | None = None


def invalidate_cache() -> None:
    global _cache
    _cache = None


async def get_quota_snapshot() -> QuotaSnapshot | None:
    """带 TTL 缓存的配额快照；数据库不可用时返回 None（fail-open）。"""
    global _cache
    ttl = settings.glm_quota_cache_ttl_seconds
    now = time.monotonic()
    if _cache is not None and ttl > 0 and now - _cache[0] < ttl:
        return _cache[1]
    try:
        # 延迟导入：避免 llm 包与 db 层的环形依赖
        from app.db.session import SessionLocal
        from app.models import LlmUsageRecord

        async with SessionLocal() as session:
            air = await session.scalar(
                select(func.coalesce(func.sum(LlmUsageRecord.total_tokens), 0)).where(
                    LlmUsageRecord.provider.in_(_AIR_PROVIDERS)
                )
            )
        snapshot = QuotaSnapshot(air_tracked=int(air or 0))
    except Exception:
        # 审计库不可用（如无库联调）：不做熔断，保持业务可用
        logger.warning("quota snapshot unavailable, fail-open", exc_info=True)
        return None
    _cache = (now, snapshot)
    return snapshot


def grant_expired(now: datetime | None = None) -> bool:
    expires_at = datetime.fromisoformat(settings.glm_grant_expires_at)
    return (now or datetime.now(timezone.utc)) >= expires_at


def air_gate(snapshot: QuotaSnapshot, task_type: str) -> tuple[bool, str | None]:
    """glm_air 是否允许用于该任务；不允许时返回 (False, 原因)。

    原因取值（在日志/审计/管理端均可区分）：
    - expired（资源包到期）/ exhausted（额度耗尽）：不允许付费时全任务熔断；
    - paid_high_value_only：到期/耗尽但允许付费——仅高价值任务可付费续用 Air，
      低价值任务一律让位（即便累计仍低于软上限，也不得静默付费）；
    - soft_limit：仍在有效期/未耗尽，但已达软上限，低价值任务让位。
    """
    high_value = task_type in HIGH_VALUE_TASKS
    expired = grant_expired()
    exhausted = snapshot.air_tracked >= settings.glm_air_grant_total_tokens
    if expired or exhausted:
        if not settings.glm_allow_paid_after_grant:
            return False, "expired" if expired else "exhausted"
        # 允许付费：强制进入 high_value_only——低价值任务不因“用量尚低”
        # 而继续走到期/耗尽后的付费 Air（否则会造成意外付费）
        if not high_value:
            return False, "paid_high_value_only"
        return True, None
    if snapshot.air_tracked >= settings.glm_air_soft_limit_tokens and not high_value:
        return False, "soft_limit"
    return True, None


_REASON_TEXT = {
    "expired": "资源包已到期",
    "exhausted": "应用侧累计已达资源包总量",
    "soft_limit": "接近耗尽（软上限），低价值任务让位",
    "paid_high_value_only": "资源包已到期/耗尽，仅高价值任务允许付费续用",
}

# 让位类原因（Air 被移除但有免费兜底，非硬熔断）
_YIELD_REASONS = frozenset({"soft_limit", "paid_high_value_only"})


def block_message(reason: str) -> str:
    if reason in _YIELD_REASONS:
        return f"glm_air 已让位：{_REASON_TEXT.get(reason, reason)}"
    return (
        f"glm_air 额度熔断：{_REASON_TEXT.get(reason, reason)}"
        "（GLM_ALLOW_PAID_AFTER_GRANT=false 禁止静默付费；"
        "控制台余额是最终真值）"
    )


async def filter_providers(
    providers: list[LLMProvider], task_type: str
) -> tuple[list[LLMProvider], str | None]:
    """按配额状态过滤 provider 列表；移除 glm_air 时返回说明文案。"""
    if not any(p.name in _AIR_PROVIDERS for p in providers):
        return providers, None
    snapshot = await get_quota_snapshot()
    if snapshot is None:
        return providers, None
    allowed, reason = air_gate(snapshot, task_type)
    if allowed:
        return providers, None
    kept = [p for p in providers if p.name not in _AIR_PROVIDERS]
    return kept, block_message(reason or "")
