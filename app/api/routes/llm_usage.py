"""LLM 用量统计与资源包预警（Tech Design M4 §11/§12，Phase 2）。

应用侧只做估算与预警：estimated_remaining = 配置的资源包总量 - 应用侧累计，
智谱控制台余额才是最终真值（服务商可能按自己的顺序抵扣多个资源包）。
"""
import math
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import require_user
from app.db.session import get_db
from app.models import LlmUsageRecord
from app.schemas.llm_usage import (
    FailureOut,
    GrantStatus,
    ProviderStat,
    TaskStat,
    UsageStatsOut,
)

router = APIRouter(prefix="/llm-usage", tags=["llm-usage"])

# §12.1 到期前提醒节点（天）
_EXPIRY_REMINDER_DAYS = (1, 7, 14, 30)


def _usage_level(ratio: float) -> str | None:
    """§12.2 消耗阈值 → 展示级别（Phase 2 仅展示，熔断在 Phase 3）。"""
    if ratio >= 1.0:
        return "exhausted"
    if ratio >= 0.95:
        return "high_value_only"
    if ratio >= 0.85:
        return "evaluate_routing"
    if ratio >= 0.70:
        return "observe"
    return None


def _expiry_warning(expires_at: datetime, days: int) -> str | None:
    if days < 0:
        return f"资源包已于 {expires_at.date()} 到期"
    for threshold in _EXPIRY_REMINDER_DAYS:
        if days <= threshold:
            return (
                f"资源包将于 {expires_at.date()} 到期（剩 {days} 天，"
                f"到期前 {threshold} 天提醒）"
            )
    return None


async def _grant_status(
    db: AsyncSession, user_id: UUID, name: str, providers: list[str], grant_total: int
) -> GrantStatus:
    now = datetime.now(timezone.utc)
    token_sum = func.coalesce(func.sum(LlmUsageRecord.total_tokens), 0)
    in_providers = LlmUsageRecord.provider.in_(providers)

    # 资源包额度绑定 GLM key，账号级共享：累计消耗/剩余/耗尽预测跨用户
    # 统计（含未打用户标签的记录）——否则各用户各看一份"剩余"会同时超卖
    tracked = int(await db.scalar(select(token_sum).where(in_providers)) or 0)

    # 每场会议平均消耗：当前用户打上 meeting 标签的记录（用户级指标）
    per_meeting = (
        await db.execute(
            select(
                token_sum,
                func.count(func.distinct(LlmUsageRecord.meeting_id)),
            ).where(
                in_providers,
                LlmUsageRecord.meeting_id.is_not(None),
                LlmUsageRecord.user_id == user_id,
            )
        )
    ).one()
    meeting_tokens, meeting_count = int(per_meeting[0]), int(per_meeting[1])
    avg_per_meeting = (
        round(meeting_tokens / meeting_count) if meeting_count else None
    )

    remaining = max(grant_total - tracked, 0)
    est_meetings = (
        math.floor(remaining / avg_per_meeting)
        if avg_per_meeting and avg_per_meeting > 0
        else None
    )

    # 最近 7 天平均每日消耗 → 预计耗尽日期（账号级）；
    # 按窗口内实际活跃天数平均，避免系统刚上线时被空白天稀释
    week_row = (
        await db.execute(
            select(
                token_sum,
                func.count(func.distinct(func.date_trunc("day", LlmUsageRecord.created_at))),
            ).where(
                in_providers, LlmUsageRecord.created_at >= now - timedelta(days=7)
            )
        )
    ).one()
    week_tokens, active_days = int(week_row[0]), int(week_row[1])
    avg_daily = round(week_tokens / active_days) if week_tokens and active_days else None
    exhaustion_date = (
        (now + timedelta(days=remaining / avg_daily)).date()
        if avg_daily and avg_daily > 0
        else None
    )

    expires_at = datetime.fromisoformat(settings.glm_grant_expires_at)
    days_until = (expires_at.date() - now.astimezone(expires_at.tzinfo).date()).days
    ratio = tracked / grant_total if grant_total > 0 else 0.0

    return GrantStatus(
        name=name,
        providers=providers,
        grant_total_tokens=grant_total,
        tracked_total_tokens=tracked,
        estimated_remaining_tokens=remaining,
        usage_ratio=round(ratio, 4),
        usage_level=_usage_level(ratio),
        avg_tokens_per_meeting=avg_per_meeting,
        estimated_remaining_meetings=est_meetings,
        avg_daily_tokens_7d=avg_daily,
        estimated_exhaustion_date=exhaustion_date,
        expires_at=expires_at,
        days_until_expiry=days_until,
        expiry_warning=_expiry_warning(expires_at, days_until),
    )


@router.get("/stats", response_model=UsageStatsOut)
async def usage_stats(
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> UsageStatsOut:
    success_count = func.sum(case((LlmUsageRecord.success.is_(True), 1), else_=0))
    fallback_count = func.sum(case((LlmUsageRecord.fallback_index > 0, 1), else_=0))
    in_sum = func.coalesce(func.sum(LlmUsageRecord.input_tokens), 0)
    out_sum = func.coalesce(func.sum(LlmUsageRecord.output_tokens), 0)
    total_sum = func.coalesce(func.sum(LlmUsageRecord.total_tokens), 0)
    # 用户隔离：调用明细/失败列表只暴露当前用户自己的记录；
    # 资源包额度是账号级共享指标，跨用户统计（见 _grant_status）
    mine = LlmUsageRecord.user_id == user_id

    provider_rows = await db.execute(
        select(
            LlmUsageRecord.provider,
            func.count(),
            success_count,
            fallback_count,
            in_sum,
            out_sum,
            total_sum,
        )
        .where(mine)
        .group_by(LlmUsageRecord.provider)
        .order_by(total_sum.desc())
    )
    by_provider = [
        ProviderStat(
            provider=provider,
            calls=calls,
            success_calls=int(ok),
            success_rate=round(int(ok) / calls, 4) if calls else 0.0,
            fallback_calls=int(fb),
            fallback_rate=round(int(fb) / calls, 4) if calls else 0.0,
            input_tokens=int(tin),
            output_tokens=int(tout),
            total_tokens=int(ttotal),
        )
        for provider, calls, ok, fb, tin, tout, ttotal in provider_rows
    ]

    task_rows = await db.execute(
        select(
            LlmUsageRecord.task_type,
            func.count(),
            success_count,
            in_sum,
            out_sum,
            total_sum,
            func.avg(LlmUsageRecord.latency_ms),
        )
        .where(mine)
        .group_by(LlmUsageRecord.task_type)
        .order_by(total_sum.desc())
    )
    by_task = [
        TaskStat(
            task_type=task_type,
            calls=calls,
            success_calls=int(ok),
            success_rate=round(int(ok) / calls, 4) if calls else 0.0,
            input_tokens=int(tin),
            output_tokens=int(tout),
            total_tokens=int(ttotal),
            avg_latency_ms=round(latency) if latency is not None else None,
        )
        for task_type, calls, ok, tin, tout, ttotal, latency in task_rows
    ]

    # 同一 GLM key 下 Air 与通用包分开记账；"glm" 同时是 legacy LLM
    # 节点名与 GLM embedding 的 provider 名，都走通用资源包
    grants = [
        await _grant_status(
            db, user_id, "glm_air", ["glm_air"], settings.glm_air_grant_total_tokens
        ),
        await _grant_status(
            db,
            user_id,
            "glm_general",
            ["glm_flash", "glm"],
            settings.glm_general_grant_total_tokens,
        ),
    ]

    failures = await db.scalars(
        select(LlmUsageRecord)
        .where(LlmUsageRecord.success.is_(False), mine)
        .order_by(LlmUsageRecord.created_at.desc())
        .limit(20)
    )
    recent_failures = [
        FailureOut(
            created_at=r.created_at,
            meeting_id=r.meeting_id,
            task_type=r.task_type,
            provider=r.provider,
            model=r.model,
            fallback_index=r.fallback_index,
            error_type=r.error_type,
            error_message=(r.error_message or "")[:300] or None,
        )
        for r in failures
    ]

    totals = (
        await db.execute(select(func.count(), total_sum).where(mine))
    ).one()

    return UsageStatsOut(
        total_calls=int(totals[0]),
        total_tokens=int(totals[1]),
        by_provider=by_provider,
        by_task=by_task,
        grants=grants,
        recent_failures=recent_failures,
    )
