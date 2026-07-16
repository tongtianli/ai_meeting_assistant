"""关键词/精确实体召回 + 三路候选 RRF 融合（RAG 设计 §5，Phase 2）。

- 只提取"精确形状"的 token（版本号/编号/金额/日期/文件名/英文词），
  **不提取裸数字**——裸数字配 `ILIKE '%…%'` 会产出大量低价值候选稀释 RRF；
- `ILIKE` 一律绑定参数 + LIKE 转义（%、_、\\），恒带 meeting_id 过滤，
  每个 token 独立候选限额（防高频词占满全局配额）；
  性能前置：transcript text 的 pg_trgm GIN 索引（迁移 0008）；
- 融合用 RRF（score = Σ 1/(k+rank)）替代 Phase 1 的简单 append；
  每人保底的受保护 speaker anchor 仍钉在 rank 最前（Phase 1 §4.1.1-E 保证不回退）。

文件组织：按评审 §6.3 采用单模块 `retrieval.py`，`retrieval/` package 化
待真实评估需要更多召回路（Reranker 等）时再做。
"""
import itertools
import logging
import re
import uuid
from dataclasses import dataclass, field

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import TranscriptSegment

logger = logging.getLogger(__name__)

# 精确实体模式（评审 §8.1：版本/编号/金额/日期豁免最短长度；金额须带单位）。
# version 须带 v 前缀或 ≥2 个点——否则 "29.5" 这类裸小数会被当版本号提走，
# 违背"不提裸数字"的约束
_ENTITY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("version", re.compile(r"\bv\d+(?:\.\d+)+\b|\b\d+(?:\.\d+){2,}\b", re.IGNORECASE)),
    ("code", re.compile(r"\b[A-Z]{2,10}-?\d+\b")),
    ("amount", re.compile(r"\d+(?:\.\d+)?\s*(?:万|亿|元|块|美元|USD|RMB)")),
    ("date", re.compile(r"\d{1,2}\s*[月/-]\s*\d{1,2}")),
    ("file", re.compile(r"\b[\w-]+\.(?:docx?|xlsx?|pptx?|pdf|txt|md|csv|json|yaml|yml)\b", re.IGNORECASE)),
]
# 普通英文 token（缩写/专名），受最短长度约束
_EN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def extract_keywords(query: str) -> list[str]:
    """从 standalone_query 提取精确 token，按出现序去重、总量封顶。

    实体模式命中的区间不再重复提取英文子串（如 "API-203" 不再吐出 "API"）。
    """
    max_tokens = settings.qa_keyword_max_tokens
    if max_tokens <= 0:
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    spans: list[tuple[int, int]] = []

    def _take(tok: str, start: int, end: int) -> None:
        key = tok.strip().lower()
        if key and key not in seen:
            seen.add(key)
            tokens.append(tok.strip())
            spans.append((start, end))

    hits: list[tuple[int, int, str]] = []
    for _, pattern in _ENTITY_PATTERNS:
        for m in pattern.finditer(query):
            hits.append((m.start(), m.end(), m.group(0)))
    # 同起点取最长匹配（"29.5 万" 优先于 "29.5"），重叠区间不重复提取
    for start, end, tok in sorted(hits, key=lambda h: (h[0], -h[1])):
        if any(s < end and start < e for s, e in spans):  # 与已取区间重叠
            continue
        _take(tok, start, end)

    for m in _EN_RE.finditer(query):
        if len(m.group(0)) < settings.qa_keyword_min_token_len:
            continue
        if any(s < m.end() and m.start() < e for s, e in spans):
            continue
        _take(m.group(0), m.start(), m.end())

    return tokens[:max_tokens]


def _escape_like(token: str) -> str:
    return (
        token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )


def _token_variants(token: str) -> list[str]:
    """空格变体："29.5 万" 也要命中原文里的 "29.5万"。"""
    stripped = token.replace(" ", "")
    return [token] if stripped == token else [token, stripped]


async def retrieve_keyword_anchors(
    session: AsyncSession,
    meeting_id: uuid.UUID,
    tokens: list[str],
    per_token_k: int,
) -> dict[str, list[TranscriptSegment]]:
    """每个 token 独立 ILIKE 召回（转义 + 绑定参数 + meeting 过滤 + 独立限额）。"""
    out: dict[str, list[TranscriptSegment]] = {}
    if per_token_k <= 0:
        return out
    for token in tokens:
        conds = [
            TranscriptSegment.text.ilike(f"%{_escape_like(v)}%", escape="\\")
            for v in _token_variants(token)
        ]
        rows = await session.scalars(
            select(TranscriptSegment)
            .where(TranscriptSegment.meeting_id == meeting_id, or_(*conds))
            .order_by(TranscriptSegment.seq)
            .limit(per_token_k)
        )
        out[token] = list(rows)
    return out


# ---------- RRF 融合（§5.3）----------


@dataclass
class RetrievalCandidate:
    segment: TranscriptSegment
    sources: set[str] = field(default_factory=set)
    vector_rank: int | None = None
    speaker_rank: int | None = None
    keyword_rank: int | None = None
    fused_score: float = 0.0

    def best_rank(self) -> int:
        return min(
            r
            for r in (self.vector_rank, self.speaker_rank, self.keyword_rank)
            if r is not None
        )


def _round_robin(lists: list[list[TranscriptSegment]]) -> list[TranscriptSegment]:
    """多组有序候选交错成单一 rank 序（跨人/跨 token 公平），去重保序。"""
    seen: set[uuid.UUID] = set()
    out: list[TranscriptSegment] = []
    for seg in itertools.chain.from_iterable(itertools.zip_longest(*lists)):
        if seg is not None and seg.id not in seen:
            seen.add(seg.id)
            out.append(seg)
    return out


def rrf_fuse(
    vector: list[TranscriptSegment],
    speaker_ranked: list[TranscriptSegment],
    keyword_ranked: list[TranscriptSegment],
    k: int,
) -> list[RetrievalCandidate]:
    """三路候选 RRF：score = Σ 1/(k+rank)，rank 从 1 起；统一去重。

    排序 tie-break 确定性：分数 → 各源最优 rank → seq。
    """
    by_id: dict[uuid.UUID, RetrievalCandidate] = {}

    def _feed(source: str, ranked: list[TranscriptSegment]) -> None:
        for rank, seg in enumerate(ranked, start=1):
            cand = by_id.setdefault(seg.id, RetrievalCandidate(segment=seg))
            cand.sources.add(source)
            setattr(cand, f"{source}_rank", rank)
            cand.fused_score += 1.0 / (k + rank)

    _feed("vector", vector)
    _feed("speaker", speaker_ranked)
    _feed("keyword", keyword_ranked)
    return sorted(
        by_id.values(),
        key=lambda c: (-c.fused_score, c.best_rank(), c.segment.seq),
    )


def fuse_anchors(
    vector: list[TranscriptSegment],
    speaker_by_person: dict[uuid.UUID, list[TranscriptSegment]],
    person_order: list[uuid.UUID],
    min_protected_per_person: int,
    keyword_by_token: dict[str, list[TranscriptSegment]],
    token_order: list[str],
    rrf_k: int,
) -> tuple[list[TranscriptSegment], list[RetrievalCandidate]]:
    """三路融合产出 anchors_by_rank（喂给 anchor 优先预算）。

    受保护 speaker anchor（每人前 N 条，§4.1.1-E）钉在最前，不参与 RRF 排位
    竞争——点名问题的保底不因融合而回退；其余按 RRF 分数排序。
    返回 (anchors, candidates)；candidates 含每个候选的召回来源（§5.4 可观测）。
    """
    speaker_ranked = _round_robin(
        [speaker_by_person.get(pid, []) for pid in person_order]
    )
    keyword_ranked = _round_robin(
        [keyword_by_token.get(tok, []) for tok in token_order]
    )
    candidates = rrf_fuse(vector, speaker_ranked, keyword_ranked, rrf_k)

    seen: set[uuid.UUID] = set()
    anchors: list[TranscriptSegment] = []
    for pid in person_order:  # 受保护配额，按人在问题中的出现序
        for seg in speaker_by_person.get(pid, [])[:min_protected_per_person]:
            if seg.id not in seen:
                seen.add(seg.id)
                anchors.append(seg)
    for cand in candidates:
        if cand.segment.id not in seen:
            seen.add(cand.segment.id)
            anchors.append(cand.segment)
    return anchors, candidates
