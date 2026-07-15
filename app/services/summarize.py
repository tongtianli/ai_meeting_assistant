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

from app.core.config import settings
from app.models import ActionItem, Meeting, Summary, SummaryExample, TranscriptSegment
from app.schemas.summary import MapFacts, SummaryContent
from app.services.llm import LLMExhaustedError, LLMTaskType, build_router
from app.services.llm.prompts import (
    SYSTEM_MAP_EXTRACT,
    SYSTEM_PLAIN,
    SYSTEM_SUMMARIZER,
    map_prompt,
    plain_text_prompt,
    reduce_prompt,
    single_pass_prompt,
)
from app.services.speakers import active_speaker_names
from app.services.summary_quality import run_quality_check

logger = logging.getLogger(__name__)


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


def _est_tokens(text: str, chars_per_token: float) -> float:
    return len(text) / chars_per_token


def chunk_lines(
    lines: list[str],
    *,
    target_tokens: int | None = None,
    max_tokens: int | None = None,
    overlap_segments: int | None = None,
    chars_per_token: float | None = None,
) -> list[str]:
    """按估算 token 把转录行切块（Tech Design M4 §13）。

    - 达 target_tokens 收口；单行超 max_tokens 时独占一块（不切断整行，不丢内容）；
    - 相邻块重叠 overlap_segments 行以防跨块 TODO/决策丢失——重叠带来的重复由
      reduce 阶段去重、ActionItem 按 seq 幂等重建兜底。
    默认从 settings 取，测试可覆盖。
    """
    tt = target_tokens or settings.summary_chunk_target_tokens
    mt = max_tokens or settings.summary_chunk_max_tokens
    ov = (
        settings.summary_chunk_overlap_segments
        if overlap_segments is None
        else overlap_segments
    )
    cpt = chars_per_token or settings.summary_chars_per_token

    chunks: list[str] = []
    current: list[str] = []
    size = 0.0
    for line in lines:
        lt = _est_tokens(line, cpt)
        if current and (size + lt > tt or lt > mt):
            chunks.append("\n".join(current))
            # 尾部 ov 行带入下一块开头（跨块延续）
            current = current[-ov:] if ov > 0 else []
            size = sum(_est_tokens(x, cpt) for x in current)
        current.append(line)
        size += lt
    if current:
        chunks.append("\n".join(current))
    return chunks


async def load_style_examples(
    session: AsyncSession, user_id
) -> list[SummaryExample]:
    """选取 few-shot 范例：启用的按更新时间取最新，受条数与字符预算约束。

    超预算的单条范例跳过而非截断——被截断的半份纪要会教坏文风。
    """
    rows = await session.scalars(
        select(SummaryExample)
        .where(SummaryExample.user_id == user_id, SummaryExample.enabled.is_(True))
        .order_by(SummaryExample.updated_at.desc())
    )
    selected: list[SummaryExample] = []
    budget = settings.summary_examples_max_chars
    for ex in rows:
        if len(selected) >= settings.summary_examples_max_count:
            break
        if len(ex.content) > budget:
            continue
        selected.append(ex)
        budget -= len(ex.content)
    return selected


async def _generate_content(
    transcript_lines: list[str], examples: list[str] | None = None
) -> tuple[dict, str, bool]:
    """返回 (content_json, 模型标识, degraded)。"""
    # 任务级路由：map 偏事实抽取走免费 Flash 优先；最终纪要（single-pass /
    # reduce / 降级纯文本）是高价值任务，优先消耗 GLM-4.5-Air 赠送额度
    final_router = build_router(LLMTaskType.SUMMARY_FINAL)
    chunks = chunk_lines(transcript_lines)
    try:
        if len(chunks) == 1:
            content, resp = await final_router.generate_json(
                SYSTEM_SUMMARIZER,
                single_pass_prompt(chunks[0], examples),
                SummaryContent,
            )
        else:
            logger.info("long meeting: map-reduce over %d chunks", len(chunks))
            map_router = build_router(LLMTaskType.SUMMARY_MAP)
            partials: list[str] = []
            for i, chunk in enumerate(chunks):
                # map 走精简事实抽取 schema（省 token）；文风由 reduce 定型
                partial, _ = await map_router.generate_json(
                    SYSTEM_MAP_EXTRACT, map_prompt(chunk, i + 1, len(chunks)), MapFacts
                )
                partials.append(partial.model_dump_json())
            content, resp = await final_router.generate_json(
                SYSTEM_SUMMARIZER, reduce_prompt(partials, examples), SummaryContent
            )
        return content.model_dump(), f"{resp.provider}/{resp.model}", False
    except LLMExhaustedError as exc:
        # 降级：至少产出纯文本纪要（PRD §9.1）
        logger.warning("structured summary failed, degrading to plain text: %s", exc)
        resp = await final_router.generate_text(
            SYSTEM_PLAIN, plain_text_prompt("\n".join(transcript_lines))
        )
        return {"text": resp.text}, f"{resp.provider}/{resp.model}", True


async def summarize_meeting(
    session: AsyncSession, meeting: Meeting, origin: str = "pipeline"
) -> Summary:
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
    examples = await load_style_examples(session, meeting.user_id)

    example_texts = [e.content for e in examples] or None
    content, model_id, degraded = await _generate_content(lines, example_texts)

    # 质量诊断（Phase 4）：仅对结构化纪要；高风险自动重跑一次（上限 1）
    quality_meta: dict = {}
    if not degraded:
        transcript_text = "\n".join(lines)
        seqs = {s.seq for s in segments}
        report = await run_quality_check(
            SummaryContent.model_validate(content),
            transcript_text, seqs, example_texts, meeting=meeting,
        )
        if report.has_high_risk:
            logger.warning(
                "summary quality high-risk, regenerating once: %s", report.to_meta()
            )
            content2, model_id2, degraded2 = await _generate_content(lines, example_texts)
            if not degraded2:
                content, model_id, degraded = content2, model_id2, degraded2
                report = await run_quality_check(
                    SummaryContent.model_validate(content2),
                    transcript_text, seqs, example_texts, meeting=meeting,
                )
            report.regenerated = True
        if not report.is_empty:
            logger.warning("summary quality flags: %s", report.to_meta())
        quality_meta = report.to_meta()

    content["_meta"] = {
        # 版本来源：pipeline（首次自动）| resummarize（用户重跑）| chat（对话修改，
        # 见 summary_edit.py）——版本历史/审计按此区分各版本从何而来
        "origin": origin,
        "model": model_id,
        "degraded": degraded,
        "meeting_time": meeting.created_at.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # 可追溯：本次纪要模仿了哪些范例（文风调优时对照用）
        "style_examples": [{"id": str(e.id), "title": e.title} for e in examples],
    }
    if quality_meta:
        content["_meta"]["quality"] = quality_meta

    summary = Summary(
        meeting_id=meeting.id,
        version=await next_summary_version(session, meeting.id),
        content_json=content,
    )
    session.add(summary)

    if not degraded:
        seg_by_seq = {s.seq: s for s in segments}
        todos = SummaryContent.model_validate(
            {k: v for k, v in content.items() if k != "_meta"}
        ).todos
        await rebuild_action_items(session, meeting.id, todos, seg_by_seq)
    else:
        await session.execute(
            delete(ActionItem).where(ActionItem.meeting_id == meeting.id)
        )
    await session.commit()
    await session.refresh(summary)
    return summary


async def next_summary_version(session: AsyncSession, meeting_id) -> int:
    """版本递增（Summary 版本化，PRD Feature 6）；摘要与聊天改纪要共用。"""
    current = await session.scalar(
        select(func.coalesce(func.max(Summary.version), 0)).where(
            Summary.meeting_id == meeting_id
        )
    )
    return current + 1


async def rebuild_action_items(
    session: AsyncSession, meeting_id, todos, seg_by_seq: dict
) -> None:
    """ActionItem 幂等重建；溯源：seq → 真实 segment（id / person_id）。

    Word 导出的 TODO 表读 ActionItem 表而非 content_json——任何产生新
    Summary 版本的路径（摘要 / 聊天改纪要）都必须同步调用本函数。
    """
    await session.execute(
        delete(ActionItem).where(ActionItem.meeting_id == meeting_id)
    )
    for todo in todos:
        source = (
            seg_by_seq.get(todo.source_segment_seq)
            if todo.source_segment_seq is not None
            else None
        )
        session.add(
            ActionItem(
                meeting_id=meeting_id,
                task=todo.task,
                owner_person_id=source.person_id if source else None,
                owner_text=todo.owner,
                deadline=todo.deadline,
                source_segment_id=source.id if source else None,
            )
        )
