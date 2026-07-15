"""LLM 用量审计（Tech Design M4 §11，Phase 2）。

- record_usage 用独立 Session 写入：审计必须在调用方事务回滚时存活，
  且写入失败绝不打断业务链路（吞异常，只记日志）
- meeting/user 上下文用 contextvar 传递：Router 深处不必层层透传参数，
  由管道/聊天入口 set 一次即可
"""
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UsageContext:
    meeting_id: uuid.UUID | None = None
    user_id: uuid.UUID | None = None


_usage_ctx: ContextVar[UsageContext] = ContextVar(
    "llm_usage_context", default=UsageContext()
)


@contextmanager
def usage_context(
    meeting_id: uuid.UUID | None = None, user_id: uuid.UUID | None = None
):
    """在作用域内为所有 LLM/embedding 调用打上会议/用户标签。"""
    token = _usage_ctx.set(UsageContext(meeting_id=meeting_id, user_id=user_id))
    try:
        yield
    finally:
        _usage_ctx.reset(token)


def now_ms() -> float:
    return time.monotonic() * 1000


async def record_usage(
    *,
    task_type: str,
    provider: str,
    model: str,
    success: bool,
    fallback_index: int = 0,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    latency_ms: int | None = None,
) -> None:
    """成功与失败调用都记录；provider 未返回 usage 时 token 留空不伪造。"""
    try:
        # 延迟导入：避免 llm 包与 db 层的环形依赖
        from app.db.session import SessionLocal
        from app.models import LlmUsageRecord

        ctx = _usage_ctx.get()
        total = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
        async with SessionLocal() as session:
            session.add(
                LlmUsageRecord(
                    meeting_id=ctx.meeting_id,
                    user_id=ctx.user_id,
                    task_type=task_type,
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total,
                    success=success,
                    fallback_index=fallback_index,
                    error_type=error_type,
                    error_message=(error_message or None) and error_message[:2000],
                    latency_ms=latency_ms,
                )
            )
            await session.commit()
    except Exception:
        # 审计写入失败（如无数据库的联调环境）不得影响业务调用
        logger.debug("llm usage record skipped", exc_info=True)
