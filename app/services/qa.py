"""AI 会议问答（RAG，PRD Feature 5 / RAG 设计 §4.1.1）。

流程：一次 LLM 调用产出意图+standalone query（query_intent.py）→ 用改写后问题
嵌入 → vector / speaker 两路 anchor 召回（每人保底 speaker anchor）→ anchor 优先
的上下文预算（先 anchor 自身、再逐圈补邻居，最后才按 seq 排序展示）→ LLM 依据
本轮 context 作答（证据不足显式拒答）→ 程序反查引用落库，杜绝幻觉引用。

嵌入懒加载缓存到 transcript_segments.embedding；stale 判定同时比对向量维度与
meetings.embedding_model_key（同维不同模型的向量空间不通，§4.1.1-G）。
"""
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import ChatMessage, Meeting, TranscriptSegment
from app.schemas.chat import CitationOut, QaAnswer
from app.services.embeddings import get_embedder
from app.services.llm import LLMExhaustedError, LLMTaskType, build_router
from app.services.llm.prompts import SYSTEM_QA, qa_prompt
from app.services.query_intent import IntentResult, classify_and_rewrite, has_anaphora
from app.services.retrieval import (
    RetrievalCandidate,
    extract_keywords,
    fuse_anchors,
    retrieve_keyword_anchors,
)
from app.services.speakers import active_speaker_names
from app.services.summarize import _fmt_ts
from app.services.summary_edit import NoSummaryYet, edit_summary

logger = logging.getLogger(__name__)

_NO_ANSWER = "未在本次会议记录中找到相关内容。"
_INSUFFICIENT = "会议原文中没有找到明确结论。"
_EDIT_FAILED = "纪要修改失败，请调整指令后重试。"


@dataclass
class QaResult:
    assistant: ChatMessage
    citations: list[CitationOut]
    summary_version: int | None = None


# ---------- 说话人匹配（§4.1.1-C）----------


def match_speaker_person_ids(
    names: dict[str, tuple[uuid.UUID, str]], question: str
) -> list[uuid.UUID]:
    """问题文本中出现的绑定真名 → person_id，按首次出现位置排序、去重。

    仅匹配 ≥2 字符的名字，单字名撞普通字的概率太高；误匹配只会
    多带几段候选（只增召回不减召回），代价可接受。
    """
    matched: dict[uuid.UUID, int] = {}
    for pid, name in dict(names.values()).items():
        if len(name) < 2:
            continue
        idx = question.find(name)
        if idx >= 0 and pid not in matched:
            matched[pid] = idx
    return [pid for pid, _ in sorted(matched.items(), key=lambda kv: kv[1])]


def match_speakers_union(
    names: dict[str, tuple[uuid.UUID, str]],
    original: str,
    standalone: str,
    allowed_context: str,
) -> tuple[list[uuid.UUID], list[str], bool]:
    """原问题 ∪ 改写问题的说话人并集（改写可能遗漏原问题里的第二个人名）。

    改写臆造的人名——只出现在 standalone、且在原问题/历史用户消息/上轮
    引用原文中均不存在——不得成为强 speaker 召回条件，记可疑并忽略。
    返回 (person_ids 按首次出现序并截断, 可疑人名, 是否截断)。
    """
    pid_to_name = {pid: nm for pid, nm in dict(names.values()).items()}
    result = match_speaker_person_ids(names, original)
    suspicious: list[str] = []
    for pid in match_speaker_person_ids(names, standalone):
        if pid in result:
            continue
        name = pid_to_name[pid]
        if name in allowed_context:
            result.append(pid)
        else:
            suspicious.append(name)
    truncated = len(result) > settings.qa_max_speaker_persons
    return result[: settings.qa_max_speaker_persons], suspicious, truncated


# ---------- 嵌入生命周期（§4.1.1-G）----------


def embedding_model_key(name: str, model: str, dim: int) -> str:
    return f"{name}/{model}:{dim}"


async def ensure_embeddings(
    session: AsyncSession,
    meeting: Meeting,
    segments: list[TranscriptSegment],
    expected_dim: int,
) -> None:
    """懒加载：向量缺失、维度不符或**模型身份不符**时整场重嵌。

    只比维度会在换成同维模型时静默错检索——身份键 "provider/model:dim"
    记在 Meeting 上（整场 all-or-nothing 重嵌，一列即可）；旧数据键为空
    视为 stale，首次问答重嵌并回填。
    """
    embedder = get_embedder()
    key = embedding_model_key(embedder.name, embedder.model, expected_dim)
    stale = meeting.embedding_model_key != key or any(
        s.embedding is None or len(s.embedding) != expected_dim for s in segments
    )
    if not stale:
        return
    vectors = await embedder.embed([s.text for s in segments])
    for seg, vec in zip(segments, vectors):
        seg.embedding = vec
    meeting.embedding_model_key = key
    await session.commit()
    logger.info(
        "embedded %d segments for meeting %s (model_key=%s)",
        len(segments),
        meeting.id,
        key,
    )


# ---------- 召回（§4.1.1-E）----------


async def _all_segments(
    session: AsyncSession, meeting_id: uuid.UUID
) -> list[TranscriptSegment]:
    rows = await session.scalars(
        select(TranscriptSegment)
        .where(TranscriptSegment.meeting_id == meeting_id)
        .order_by(TranscriptSegment.seq)
    )
    return list(rows)


async def retrieve_vector_anchors(
    session: AsyncSession, meeting_id: uuid.UUID, qvec: list[float], k: int
) -> list[TranscriptSegment]:
    """pgvector 余弦相似 top-k；返回序即 rank。"""
    if k <= 0:
        return []
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


async def retrieve_speaker_anchors(
    session: AsyncSession,
    meeting_id: uuid.UUID,
    qvec: list[float],
    person_ids: list[uuid.UUID],
    per_person_k: int,
) -> dict[uuid.UUID, list[TranscriptSegment]]:
    """每位命中说话人独立取 top-k（配额不互相挤占），返回按人分组。"""
    out: dict[uuid.UUID, list[TranscriptSegment]] = {}
    if per_person_k <= 0:
        return out
    for pid in person_ids:
        rows = await session.scalars(
            select(TranscriptSegment)
            .where(
                TranscriptSegment.meeting_id == meeting_id,
                TranscriptSegment.embedding.is_not(None),
                TranscriptSegment.person_id == pid,
            )
            .order_by(TranscriptSegment.embedding.cosine_distance(qvec))
            .limit(per_person_k)
        )
        out[pid] = list(rows)
    return out


# Phase 2 起，anchor 合并由 retrieval.fuse_anchors（三路 RRF + 受保护钉前）完成


# ---------- anchor 优先的上下文预算（§4.1.1-D）----------


@dataclass
class ContextStats:
    anchor_count: int = 0
    kept_anchor_count: int = 0
    evicted_anchor_count: int = 0
    context_segment_count: int = 0
    truncated: bool = False


def budget_context(
    anchors_by_rank: list[TranscriptSegment],
    seg_by_seq: dict[int, TranscriptSegment],
    window: int,
    max_segments: int,
) -> tuple[list[TranscriptSegment], ContextStats]:
    """区分"优先级选择"与"最终展示顺序"：

    1. 按 rank 先纳入每个 anchor 自身（超预算则淘汰低 rank anchor 并计数）；
    2. 按距离 ±1、±2… 逐圈为各 anchor 补邻居（不是一个 anchor 一次吃满窗口）；
    3. 达预算即停；
    4. 仅在选择完成后按 seq 排序——排序只影响展示，不影响取舍。
    """
    stats = ContextStats(anchor_count=len(anchors_by_rank))
    selected: dict[uuid.UUID, TranscriptSegment] = {}
    kept: list[TranscriptSegment] = []
    for anchor in anchors_by_rank:
        if len(selected) >= max_segments:
            stats.evicted_anchor_count += 1
            continue
        if anchor.id not in selected:
            selected[anchor.id] = anchor
            kept.append(anchor)
    stats.kept_anchor_count = len(kept)

    full = len(selected) >= max_segments
    for distance in range(1, window + 1):
        if full:
            break
        for anchor in kept:
            for seq in (anchor.seq - distance, anchor.seq + distance):
                seg = seg_by_seq.get(seq)
                if seg is None or seg.id in selected:
                    continue
                selected[seg.id] = seg
                if len(selected) >= max_segments:
                    full = True
                    break
            if full:
                break

    ordered = sorted(selected.values(), key=lambda s: s.seq)
    stats.context_segment_count = len(ordered)
    stats.truncated = full or stats.evicted_anchor_count > 0
    return ordered, stats


# ---------- 检索元数据（§4.1.1-H：默认脱敏）----------


def _log_retrieval(
    question: str,
    intent_res: IntentResult,
    matched_person_count: int,
    suspicious_names: list[str],
    source_counts: dict[str, int],
    stats: ContextStats,
    latency_ms: dict[str, int],
    candidates: list[RetrievalCandidate] | None = None,
) -> None:
    payload: dict = {
        "question_hash": "sha256:"
        + hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
        "standalone_query_changed": intent_res.rewritten,
        "used_cited_context": intent_res.used_cited_context,
        "matched_person_count": matched_person_count,
        "suspicious_rewrite_names": len(suspicious_names),
        "retrieval_strategy_counts": source_counts,
        "anchor_count": stats.anchor_count,
        "kept_anchor_count": stats.kept_anchor_count,
        "evicted_anchor_count": stats.evicted_anchor_count,
        "context_segment_count": stats.context_segment_count,
        "truncated": stats.truncated,
        "latency_ms": latency_ms,
    }
    if candidates:
        # §5.4 可观测性：每个候选的召回来源（只含 seq 与来源名，不含原文）
        payload["candidate_sources"] = [
            {"seq": c.segment.seq, "src": sorted(c.sources)}
            for c in candidates[:40]
        ]
    if settings.qa_retrieval_debug_text:  # 明文仅显式开启才记录
        payload["original_question"] = question
        payload["standalone_query"] = intent_res.standalone_query
    logger.info("qa retrieval %s", json.dumps(payload, ensure_ascii=False))


# ---------- 引用反查 ----------


async def resolve_citations(
    session: AsyncSession,
    meeting_id: uuid.UUID,
    segment_ids: list | None,
) -> list[CitationOut]:
    """cited_segment_ids（UUID）→ CitationOut（含真名/时间戳/原文）。"""
    if not segment_ids:
        return []
    ids = [uuid.UUID(str(sid)) for sid in segment_ids]
    # 强制会议隔离：引用只在本会议内反查（历史脏数据也不外泄他会原文）
    segs = list(
        await session.scalars(
            select(TranscriptSegment).where(
                TranscriptSegment.id.in_(ids),
                TranscriptSegment.meeting_id == meeting_id,
            )
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


# ---------- 聊天编排 ----------


async def _recent_history(
    session: AsyncSession, meeting_id: uuid.UUID
) -> list[ChatMessage]:
    """最近若干条历史（旧→新）。必须在插入当前用户消息**之前**调用（§4.1.1-A 时序）。"""
    limit = settings.qa_rewrite_recent_messages * 2  # user/assistant 交错
    rows = list(
        await session.scalars(
            select(ChatMessage)
            .where(ChatMessage.meeting_id == meeting_id)
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(limit)
        )
    )
    return list(reversed(rows))


async def _last_cited_texts(
    session: AsyncSession, meeting_id: uuid.UUID, history: list[ChatMessage]
) -> list[str]:
    """上一轮 assistant 合法引用的原文（指代改写的事实语境，§4.1.1-B）。

    - 只看**紧邻本轮**的最后一条 assistant 消息：它无引用（拒答/编辑回执）
      就返回空，不回退更旧引用——"他/这个方案"不得错绑到更早的话题；
    - segment 反查强制限定本会议：即便历史数据混入他会 ID，
      其他会议原文也绝不进入改写 prompt（会议隔离）。
    """
    last_assistant = next(
        (m for m in reversed(history) if m.role == "assistant"), None
    )
    if last_assistant is None or not last_assistant.cited_segment_ids:
        return []
    ids = [
        uuid.UUID(str(sid))
        for sid in last_assistant.cited_segment_ids[
            : settings.qa_rewrite_cited_max_segments
        ]
    ]
    segs = list(
        await session.scalars(
            select(TranscriptSegment).where(
                TranscriptSegment.id.in_(ids),
                TranscriptSegment.meeting_id == meeting_id,
            )
        )
    )
    by_id = {s.id: s for s in segs}
    return [by_id[i].text for i in ids if i in by_id]


async def handle_chat(
    session: AsyncSession, meeting: Meeting, question: str
) -> QaResult:
    """聊天入口：先取历史再落库用户消息，一次调用产出意图+改写后按意图路由。

    用户消息先于 LLM 调用单独 commit——created_at 严格早于 assistant 回复，
    且慢调用失败时问题本身不丢。
    """
    history = await _recent_history(session, meeting.id)
    session.add(ChatMessage(meeting_id=meeting.id, role="user", content=question))
    await session.commit()

    names = await active_speaker_names(session, meeting.id)
    speaker_list = sorted({nm for _, nm in names.values() if len(nm) >= 2})
    recent_user = [m.content for m in history if m.role == "user"]
    # 指代门控：命中才做上一轮引用原文的额外查询
    cited_texts = (
        await _last_cited_texts(session, meeting.id, history)
        if has_anaphora(question)
        else []
    )

    started = time.monotonic()
    intent_res = await classify_and_rewrite(
        question, recent_user, speaker_list, cited_texts
    )
    rewrite_ms = int((time.monotonic() - started) * 1000)

    if intent_res.intent == "edit":
        return await _handle_edit(session, meeting, question)
    return await _answer_question(
        session, meeting, question, intent_res, recent_user, cited_texts, rewrite_ms
    )


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
    session: AsyncSession,
    meeting: Meeting,
    question: str,
    intent_res: IntentResult,
    recent_user: list[str],
    cited_texts: list[str],
    rewrite_ms: int,
) -> QaResult:
    latency: dict[str, int] = {"intent_rewrite": rewrite_ms}
    segments = await _all_segments(session, meeting.id)
    seg_by_seq_all = {s.seq: s for s in segments}

    t0 = time.monotonic()
    embedder = get_embedder()
    # 改写后的问题用于 embedding（原问题只用于最终回答语气）
    qvec = (await embedder.embed([intent_res.standalone_query]))[0]
    await ensure_embeddings(session, meeting, segments, expected_dim=len(qvec))
    latency["embedding"] = int((time.monotonic() - t0) * 1000)

    names = await active_speaker_names(session, meeting.id)
    # 说话人匹配并集：改写臆造的人名不得成为强召回条件
    allowed_ctx = "\n".join([question, *recent_user, *cited_texts])
    person_ids, suspicious, _persons_truncated = match_speakers_union(
        names, question, intent_res.standalone_query, allowed_ctx
    )
    if suspicious:
        logger.warning(
            "rewrite introduced %d unknown speaker name(s); ignored", len(suspicious)
        )

    t0 = time.monotonic()
    vector = await retrieve_vector_anchors(
        session, meeting.id, qvec, settings.qa_top_k
    )
    speaker_by_person = await retrieve_speaker_anchors(
        session, meeting.id, qvec, person_ids, settings.qa_per_person_k
    )
    # 关键词/精确实体召回（§5.2）：从改写后问题提取精确 token
    tokens = extract_keywords(intent_res.standalone_query)
    keyword_by_token = await retrieve_keyword_anchors(
        session, meeting.id, tokens, settings.qa_keyword_top_k_per_token
    )
    # 三路 RRF 融合（§5.3）；受保护 speaker anchor 仍钉前（§4.1.1-E）
    anchors, candidates = fuse_anchors(
        vector,
        speaker_by_person,
        person_ids,
        settings.qa_min_speaker_anchors_per_person,
        keyword_by_token,
        tokens,
        settings.qa_rrf_k,
    )
    context, stats = budget_context(
        anchors,
        seg_by_seq_all,
        settings.qa_neighbor_window,
        settings.qa_max_context_segments,
    )
    latency["retrieval"] = int((time.monotonic() - t0) * 1000)

    seg_by_seq = {s.seq: s for s in context}
    citations: list[CitationOut] = []
    if not context:
        answer_text = _NO_ANSWER
        latency["generation"] = 0
    else:
        lines = "\n".join(
            f"[{s.seq}] [{_fmt_ts(s.start_time)}] "
            f"{names.get(s.speaker_label, (None, s.speaker_label))[1]}: {s.text}"
            for s in context
        )
        t0 = time.monotonic()
        try:
            # QA 只输入检索后的少量片段，免费 Flash 优先；Air 仅作末位兜底
            parsed, _ = await build_router(LLMTaskType.QA_ANSWER).generate_json(
                SYSTEM_QA, qa_prompt(question, lines), QaAnswer
            )
            # 溯源硬约束：只采信落在本轮 context 上的引用，反查真实 segment
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
            if parsed.insufficient_evidence:
                # 显式拒答：不携带引用
                citations = []
                answer_text = parsed.answer.strip() or _INSUFFICIENT
            elif not citations:
                # §4.1.1-F：事实性回答必须有 ≥1 条本轮 context 内的合法引用，
                # 否则回退为明确拒答，不让无据结论流出
                answer_text = _INSUFFICIENT
            else:
                answer_text = parsed.answer.strip() or _NO_ANSWER
        except LLMExhaustedError as exc:
            logger.warning("QA LLM exhausted for meeting %s: %s", meeting.id, exc)
            answer_text = _NO_ANSWER
            citations = []
        latency["generation"] = int((time.monotonic() - t0) * 1000)

    _log_retrieval(
        question,
        intent_res,
        matched_person_count=len(person_ids),
        suspicious_names=suspicious,
        source_counts={
            "vector": len(vector),
            "speaker": sum(len(v) for v in speaker_by_person.values()),
            "keyword": sum(len(v) for v in keyword_by_token.values()),
            "keyword_tokens": len(tokens),
        },
        stats=stats,
        latency_ms=latency,
        candidates=candidates,
    )

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
