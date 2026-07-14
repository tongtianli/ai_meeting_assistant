"""AI 会议问答（RAG，PRD Feature 5）。

嵌入 segment（懒加载缓存到 transcript_segments.embedding）→ pgvector 相似度
检索 → LLM 依据检索片段作答，回答强制引用 segment（seq），程序反查真实
segment 落 chat_messages.cited_segment_ids，杜绝幻觉引用。
"""
import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import ChatMessage, Meeting, TranscriptSegment
from app.schemas.chat import CitationOut, IntentOut, QaAnswer
from app.services.embeddings import get_embedder
from app.services.llm import LLMExhaustedError, build_router
from app.services.llm.prompts import SYSTEM_INTENT, SYSTEM_QA, intent_prompt, qa_prompt
from app.services.speakers import active_speaker_names
from app.services.summarize import _fmt_ts
from app.services.summary_edit import NoSummaryYet, edit_summary

logger = logging.getLogger(__name__)

_NO_ANSWER = "未在本次会议记录中找到相关内容。"
_EDIT_FAILED = "纪要修改失败，请调整指令后重试。"


@dataclass
class QaResult:
    assistant: ChatMessage
    citations: list[CitationOut]
    summary_version: int | None = None


async def classify_intent(message: str) -> str:
    """聊天意图路由（PRD §7.1）；分类失败一律回退 query（查询无副作用）。"""
    try:
        parsed, _ = await build_router().generate_json(
            SYSTEM_INTENT, intent_prompt(message), IntentOut
        )
        return parsed.intent
    except LLMExhaustedError as exc:
        logger.warning("intent classification failed, fallback to query: %s", exc)
        return "query"


async def _all_segments(
    session: AsyncSession, meeting_id: uuid.UUID
) -> list[TranscriptSegment]:
    rows = await session.scalars(
        select(TranscriptSegment)
        .where(TranscriptSegment.meeting_id == meeting_id)
        .order_by(TranscriptSegment.seq)
    )
    return list(rows)


async def ensure_embeddings(
    session: AsyncSession,
    segments: list[TranscriptSegment],
    expected_dim: int,
) -> None:
    """懒加载：segment embedding 缺失或维度不符（换过 provider）时整场重嵌。

    维度自愈——同会议 segment 恒一起嵌入，只要 expected_dim 与已存维度不符
    即判定 provider 变更，重嵌全部以保证与问题向量空间一致。
    """
    stale = any(
        s.embedding is None or len(s.embedding) != expected_dim for s in segments
    )
    if not stale:
        return
    embedder = get_embedder()
    vectors = await embedder.embed([s.text for s in segments])
    for seg, vec in zip(segments, vectors):
        seg.embedding = vec
    await session.commit()
    logger.info(
        "embedded %d segments for meeting %s (provider=%s dim=%d)",
        len(segments),
        segments[0].meeting_id if segments else "-",
        embedder.name,
        expected_dim,
    )


async def _retrieve(
    session: AsyncSession, meeting_id: uuid.UUID, qvec: list[float], k: int
) -> list[TranscriptSegment]:
    rows = await session.scalars(
        select(TranscriptSegment)
        .where(
            TranscriptSegment.meeting_id == meeting_id,
            TranscriptSegment.embedding.is_not(None),
        )
        .order_by(TranscriptSegment.embedding.cosine_distance(qvec))
        .limit(k)
    )
    return list(rows)


async def resolve_citations(
    session: AsyncSession,
    meeting_id: uuid.UUID,
    segment_ids: list | None,
) -> list[CitationOut]:
    """cited_segment_ids（UUID）→ CitationOut（含真名/时间戳/原文）。"""
    if not segment_ids:
        return []
    ids = [uuid.UUID(str(sid)) for sid in segment_ids]
    segs = list(
        await session.scalars(
            select(TranscriptSegment).where(TranscriptSegment.id.in_(ids))
        )
    )
    names = await active_speaker_names(session, meeting_id)
    by_id = {s.id: s for s in segs}
    out: list[CitationOut] = []
    for sid in ids:  # 保持引用顺序
        s = by_id.get(sid)
        if s is None:
            continue
        out.append(
            CitationOut(
                seq=s.seq,
                start_time=s.start_time,
                speaker_name=names.get(s.speaker_label, (None, s.speaker_label))[1],
                text=s.text,
            )
        )
    return out


async def handle_chat(
    session: AsyncSession, meeting: Meeting, question: str
) -> QaResult:
    """聊天入口：先落库用户消息，按意图路由到查询或改纪要（PRD §7.1）。

    用户消息先于 LLM 调用单独 commit——created_at 严格早于 assistant 回复，
    且慢调用失败时问题本身不丢。
    """
    session.add(ChatMessage(meeting_id=meeting.id, role="user", content=question))
    await session.commit()

    if await classify_intent(question) == "edit":
        return await _handle_edit(session, meeting, question)
    return await _answer_question(session, meeting, question)


async def _handle_edit(
    session: AsyncSession, meeting: Meeting, instruction: str
) -> QaResult:
    meeting_id = meeting.id  # rollback 会使 ORM 属性过期，先取成普通值
    version: int | None = None
    try:
        summary, note = await edit_summary(session, meeting, instruction)
        content = f"已生成纪要新版本 v{summary.version}：{note}"
        version = summary.version
    except NoSummaryYet as exc:
        content = str(exc)
    except LLMExhaustedError as exc:
        logger.warning("summary edit LLM exhausted for %s: %s", meeting_id, exc)
        await session.rollback()  # 防御：丢弃半途的未提交改动
        content = _EDIT_FAILED
    assistant = ChatMessage(
        meeting_id=meeting_id, role="assistant", content=content
    )
    session.add(assistant)
    await session.commit()
    await session.refresh(assistant)
    return QaResult(assistant=assistant, citations=[], summary_version=version)


async def _answer_question(
    session: AsyncSession, meeting: Meeting, question: str
) -> QaResult:
    segments = await _all_segments(session, meeting.id)
    embedder = get_embedder()
    qvec = (await embedder.embed([question]))[0]
    await ensure_embeddings(session, segments, expected_dim=len(qvec))

    retrieved = await _retrieve(session, meeting.id, qvec, settings.qa_top_k)
    names = await active_speaker_names(session, meeting.id)
    seg_by_seq = {s.seq: s for s in retrieved}

    citations: list[CitationOut] = []
    if not retrieved:
        answer_text = _NO_ANSWER
    else:
        lines = "\n".join(
            f"[{s.seq}] [{_fmt_ts(s.start_time)}] "
            f"{names.get(s.speaker_label, (None, s.speaker_label))[1]}: {s.text}"
            for s in retrieved
        )
        try:
            parsed, _ = await build_router().generate_json(
                SYSTEM_QA, qa_prompt(question, lines), QaAnswer
            )
            answer_text = parsed.answer.strip() or _NO_ANSWER
            # 溯源硬约束：只采信落在检索片段上的引用，反查真实 segment
            for seq in dict.fromkeys(parsed.cited_segment_seqs):  # 去重保序
                seg = seg_by_seq.get(seq)
                if seg is not None:
                    citations.append(
                        CitationOut(
                            seq=seg.seq,
                            start_time=seg.start_time,
                            speaker_name=names.get(
                                seg.speaker_label, (None, seg.speaker_label)
                            )[1],
                            text=seg.text,
                        )
                    )
        except LLMExhaustedError as exc:
            logger.warning("QA LLM exhausted for meeting %s: %s", meeting.id, exc)
            answer_text = _NO_ANSWER

    assistant = ChatMessage(
        meeting_id=meeting.id,
        role="assistant",
        content=answer_text,
        cited_segment_ids=[str(seg_by_seq[c.seq].id) for c in citations] or None,
    )
    session.add(assistant)
    await session.commit()
    await session.refresh(assistant)
    return QaResult(assistant=assistant, citations=citations)
