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

from sqlalchemy import nullslast, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import TranscriptSegment

logger = logging.getLogger(__name__)

# 精确实体模式（评审 §8.1：版本/编号/金额/日期豁免最短长度；金额须带单位）。
# version：带 v 前缀（v2 / v2.1.4 均可）或无前缀但 ≥2 个点（2.1.4）——
# "29.5" 这类裸小数不得被当版本号提走（"不提裸数字"约束）。
# code 大小写不敏感：用户输入 api-203 也应触发精确召回（ILIKE 本就不区分大小写）
_ENTITY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("version", re.compile(r"\bv\d+(?:\.\d+)*\b|\b\d+(?:\.\d+){2,}\b", re.IGNORECASE)),
    ("code", re.compile(r"\b[A-Za-z]{2,10}-?\d+\b")),
    ("amount", re.compile(r"\d+(?:\.\d+)?\s*(?:万|亿|元|块|美元|USD|RMB)")),
    ("date", re.compile(r"\d{1,2}\s*[月/-]\s*\d{1,2}")),
    ("file", re.compile(r"\b[\w-]+\.(?:docx?|xlsx?|pptx?|pdf|txt|md|csv|json|yaml|yml)\b", re.IGNORECASE)),
]
# 普通英文 token（缩写/专名），受最短长度约束
_EN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def extract_keywords(query: str) -> list[str]:
    """从 standalone_query 提取精确 token：**全局按出现位置排序**、去重、封顶。

    实体候选与英文候选统一收集后按 (start, 最长优先) 排序——封顶截断时
    保留文本中更早出现的 token（评审修复：不再实体全部先于英文）；
    重叠区间只取一次（"API-203" 不再吐出 "API"，"29.5 万" 优先于 "29.5"）。
    """
    max_tokens = settings.qa_keyword_max_tokens
    if max_tokens <= 0:
        return []
    candidates: list[tuple[int, int, str]] = []
    for _, pattern in _ENTITY_PATTERNS:
        for m in pattern.finditer(query):
            candidates.append((m.start(), m.end(), m.group(0)))
    for m in _EN_RE.finditer(query):
        if len(m.group(0)) >= settings.qa_keyword_min_token_len:
            candidates.append((m.start(), m.end(), m.group(0)))

    tokens: list[str] = []
    seen: set[str] = set()
    spans: list[tuple[int, int]] = []
    # 全局出现序；同起点取最长（实体天然长于其英文子串，故实体优先成立）。
    # "区间占用"与"token 去重"分离：重复出现的实体虽不重复输出，但仍要
    # 占用其命中区间——否则第二次出现的 "API-203" 挡不住子串 "API"（review 修复）
    for start, end, tok in sorted(candidates, key=lambda c: (c[0], -c[1])):
        if any(s < end and start < e for s, e in spans):  # 与已占区间重叠
            continue
        spans.append((start, end))
        key = tok.strip().lower()
        if key and key not in seen:
            seen.add(key)
            tokens.append(tok.strip())
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
    qvec: list[float] | None = None,
) -> dict[str, list[TranscriptSegment]]:
    """每个 token 独立 ILIKE 召回（转义 + 绑定参数 + meeting 过滤 + 独立限额）。

    限额内排序（评审修复）：给定 qvec 时按与问题的 embedding 余弦距离取
    最相关的 K 条——同一实体反复出现时不再"取最早几次"而漏掉真正回答
    问题的后段发言；无 qvec（如离线工具/测试）退回按 seq。
    """
    out: dict[str, list[TranscriptSegment]] = {}
    if per_token_k <= 0:
        return out
    for token in tokens:
        conds = [
            TranscriptSegment.text.ilike(f"%{_escape_like(v)}%", escape="\\")
            for v in _token_variants(token)
        ]
        stmt = select(TranscriptSegment).where(
            TranscriptSegment.meeting_id == meeting_id, or_(*conds)
        )
        if qvec is not None:
            # 空向量段排最后（热路径中 ensure_embeddings 已保证全量嵌入）
            stmt = stmt.order_by(
                nullslast(TranscriptSegment.embedding.cosine_distance(qvec)),
                TranscriptSegment.seq,
            )
        else:
            stmt = stmt.order_by(TranscriptSegment.seq)
        rows = await session.scalars(stmt.limit(per_token_k))
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
