"""LLM 摘要阶段：map-reduce → 结构化 JSON（校验+重试）→ Summary/ActionItem 入库。

- 长会议分段 map-reduce，跨段 TODO/决策在 reduce 中合并（PRD Feature 3）
- 溯源：LLM 引用 segment seq，程序反查 segment id/文本/时间戳
- 降级：结构化输出全部失败时至少产出纯文本纪要（degraded=true）
- Summary 版本化追加（重试产生新版本而非覆盖）
"""
import logging
from datetime import datetime, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ActionItem, Meeting, Summary, TranscriptSegment
from app.schemas.summary import SummaryContent
from app.services.llm import LLMExhaustedError, build_router
from app.services.llm.prompts import (
    SYSTEM_PLAIN,
    SYSTEM_SUMMARIZER,
    map_prompt,
    plain_text_prompt,
    reduce_prompt,
    single_pass_prompt,
)
from app.services.speakers import active_speaker_names

logger = logging.getLogger(__name__)

# 单块转录的字符预算（中文约 2-4k tokens/块），超出走 map-reduce
CHUNK_CHAR_BUDGET = 8000


def _fmt_ts(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def render_transcript_lines(
    segments: list[TranscriptSegment], names: dict
) -> list[str]:
    return [
        f"[{s.seq}] [{_fmt_ts(s.start_time)}] "
        f"{names.get(s.speaker_label, (None, s.speaker_label))[1]}: {s.text}"
        for s in segments
    ]


def chunk_lines(lines: list[str], budget: int = CHUNK_CHAR_BUDGET) -> list[str]:
    """按字符预算把转录行切块；单行超预算时独占一块，不丢内容。"""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + len(line) > budget:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


async def _generate_content(transcript_lines: list[str]) -> tuple[dict, str, bool]:
    """返回 (content_json, 模型标识, degraded)。"""
    router = build_router()
    chunks = chunk_lines(transcript_lines)
    try:
        if len(chunks) == 1:
            content, resp = await router.generate_json(
                SYSTEM_SUMMARIZER, single_pass_prompt(chunks[0]), SummaryContent
            )
        else:
            logger.info("long meeting: map-reduce over %d chunks", len(chunks))
            partials: list[str] = []
            for i, chunk in enumerate(chunks):
                partial, _ = await router.generate_json(
                    SYSTEM_SUMMARIZER, map_prompt(chunk, i + 1, len(chunks)), SummaryContent
                )
                partials.append(partial.model_dump_json())
            content, resp = await router.generate_json(
                SYSTEM_SUMMARIZER, reduce_prompt(partials), SummaryContent
            )
        return content.model_dump(), f"{resp.provider}/{resp.model}", False
    except LLMExhaustedError as exc:
        # 降级：至少产出纯文本纪要（PRD §9.1）
        logger.warning("structured summary failed, degrading to plain text: %s", exc)
        resp = await router.generate_text(
            SYSTEM_PLAIN, plain_text_prompt("\n".join(transcript_lines))
        )
        return {"text": resp.text}, f"{resp.provider}/{resp.model}", True


async def summarize_meeting(session: AsyncSession, meeting: Meeting) -> Summary:
    segments = list(
        await session.scalars(
            select(TranscriptSegment)
            .where(TranscriptSegment.meeting_id == meeting.id)
            .order_by(TranscriptSegment.seq)
        )
    )
    if not segments:
        raise RuntimeError("no transcript segments to summarize")
    names = await active_speaker_names(session, meeting.id)
    lines = render_transcript_lines(segments, names)

    content, model_id, degraded = await _generate_content(lines)
    content["_meta"] = {
        "model": model_id,
        "degraded": degraded,
        "meeting_time": meeting.created_at.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    next_version = (
        await session.scalar(
            select(func.coalesce(func.max(Summary.version), 0)).where(
                Summary.meeting_id == meeting.id
            )
        )
    ) + 1
    summary = Summary(
        meeting_id=meeting.id, version=next_version, content_json=content
    )
    session.add(summary)

    # ActionItem 幂等重建；溯源：seq → 真实 segment（id / person_id）
    await session.execute(
        delete(ActionItem).where(ActionItem.meeting_id == meeting.id)
    )
    if not degraded:
        seg_by_seq = {s.seq: s for s in segments}
        for todo in SummaryContent.model_validate(
            {k: v for k, v in content.items() if k != "_meta"}
        ).todos:
            source = (
                seg_by_seq.get(todo.source_segment_seq)
                if todo.source_segment_seq is not None
                else None
            )
            session.add(
                ActionItem(
                    meeting_id=meeting.id,
                    task=todo.task,
                    owner_person_id=source.person_id if source else None,
                    owner_text=todo.owner,
                    deadline=todo.deadline,
                    source_segment_id=source.id if source else None,
                )
            )
    await session.commit()
    await session.refresh(summary)
    return summary
