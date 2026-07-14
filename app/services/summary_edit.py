"""聊天改纪要（PRD §7.1）：自然语言指令 → LLM 整体重生成 → Summary 新版本。

生成新版本而非覆盖（PRD Feature 6 版本化）；同步重建 ActionItem，
保证 Word 导出的 TODO 表与新纪要一致。
"""
import json
import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Meeting, Summary, TranscriptSegment
from app.schemas.summary import SummaryContent
from app.services.llm import build_router
from app.services.llm.prompts import SYSTEM_SUMMARY_EDIT, summary_edit_prompt
from app.services.summarize import next_summary_version, rebuild_action_items

logger = logging.getLogger(__name__)


class NoSummaryYet(RuntimeError):
    """该会议尚无纪要可改。"""


async def edit_summary(
    session: AsyncSession, meeting: Meeting, instruction: str
) -> tuple[Summary, str]:
    """按指令修改最新纪要，落库为新版本；返回 (新 Summary, 变更简述)。

    LLMExhaustedError 向上传播，由聊天层转为失败回执（不产生新版本）。
    """
    latest = await session.scalar(
        select(Summary)
        .where(Summary.meeting_id == meeting.id)
        .order_by(Summary.version.desc())
        .limit(1)
    )
    if latest is None:
        raise NoSummaryYet("纪要尚未生成")

    old_meta = latest.content_json.get("_meta", {})
    base = {k: v for k, v in latest.content_json.items() if k != "_meta"}
    if old_meta.get("degraded"):
        # 降级纯文本纪要没有结构可改；引导先重跑摘要
        raise NoSummaryYet("当前纪要为降级纯文本，暂不支持聊天修改")

    parsed, resp = await build_router().generate_json(
        SYSTEM_SUMMARY_EDIT,
        summary_edit_prompt(json.dumps(base, ensure_ascii=False), instruction),
        SummaryContent,
    )

    content = parsed.model_dump()
    content["_meta"] = {
        "origin": "chat",
        "edit_instruction": instruction[:500],
        "base_version": latest.version,
        "model": f"{resp.provider}/{resp.model}",
        "degraded": False,
        "meeting_time": old_meta.get("meeting_time")
        or meeting.created_at.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    summary = Summary(
        meeting_id=meeting.id,
        version=await next_summary_version(session, meeting.id),
        content_json=content,
    )
    session.add(summary)

    # Word 导出 TODO 表读 ActionItem——与新纪要同步重建
    segments = list(
        await session.scalars(
            select(TranscriptSegment).where(
                TranscriptSegment.meeting_id == meeting.id
            )
        )
    )
    await rebuild_action_items(
        session, meeting.id, parsed.todos, {s.seq: s for s in segments}
    )
    await session.commit()
    await session.refresh(summary)
    logger.info(
        "summary edited via chat: meeting=%s v%d -> v%d",
        meeting.id,
        latest.version,
        summary.version,
    )
    note = f"已根据指令更新纪要（基于 v{latest.version}）"
    return summary, note
