"""会议问答离线评估（RAG 设计 §6.2，Phase 3）。

- 数据集：从真实会议整理的问题 + 期望证据 segment（seq 或 UUID），JSON 数组；
  样例见 eval/qa_dataset.example.json，格式校验失败逐条报错不静默跳过
- 指标：Recall@5/@10、MRR、context 命中率、（--with-answers 时）引用准确率、
  无答案拒答率、答案置信度分布；P50/P95 延迟；平均候选数与 >20 占比
  （§6.3 Reranker 引入条件的判断依据）
- 检索复用 qa.retrieve_context——评估与热路径同一条管线，不双写；
- 支持 RRF k 对比（评审 §8.3：10/30/60），逐 k 出一份报告。

运行：uv run python scripts/qa_eval.py --dataset eval/qa_dataset.json
"""
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select

from app.core.config import settings
from app.models import Meeting, TranscriptSegment

logger = logging.getLogger(__name__)

_REFUSALS = ("未在本次会议记录中找到相关内容", "会议原文中没有找到明确结论")

# 覆盖类型参考（§6.2）；no_answer 类的 expected_seqs 为空、期望拒答
CATEGORIES = {
    "speaker", "date", "amount", "action_item", "decision",
    "pronoun", "comparison", "cross_segment", "negative", "no_answer", "general",
}


@dataclass
class EvalItem:
    meeting_id: uuid.UUID
    question: str
    expected_seqs: list[int]
    category: str = "general"
    expected_answer: str | None = None
    speaker: str | None = None


@dataclass
class ItemResult:
    item: EvalItem
    # 本轮解析后的期望证据（seqs + 解析出的 ids，去重；不回写 EvalItem——
    # 多组 rrf-k 连续评估时 items 复用，写回会跨轮污染指标）
    expected_seqs: list[int] = field(default_factory=list)
    anchor_seqs: list[int] = field(default_factory=list)
    context_seqs: list[int] = field(default_factory=list)
    candidate_count: int = 0
    latency_ms: int = 0
    # --with-answers 时填充
    answered: bool | None = None
    refused: bool | None = None
    cited_seqs: list[int] | None = None
    confidence: str | None = None


def load_dataset(path: str | Path) -> list[EvalItem]:
    """加载并校验数据集；支持 expected_segment_seqs（推荐）或 expected_segment_ids。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("数据集必须是 JSON 数组")
    items: list[EvalItem] = []
    errors: list[str] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict) or str(row.get("_comment", "")).startswith("样例"):
            continue
        try:
            category = row.get("category", "general")
            if category not in CATEGORIES:
                raise ValueError(f"未知 category: {category}")
            seqs = row.get("expected_segment_seqs")
            ids = row.get("expected_segment_ids")
            if seqs is None and ids is None and category != "no_answer":
                raise ValueError("缺少 expected_segment_seqs / expected_segment_ids")
            items.append(
                EvalItem(
                    meeting_id=uuid.UUID(str(row["meeting_id"])),
                    question=str(row["question"]).strip(),
                    expected_seqs=[int(s) for s in (seqs or [])],
                    category=category,
                    expected_answer=row.get("expected_answer"),
                    speaker=row.get("speaker"),
                )
            )
            if ids:  # UUID 形式在 evaluate 内解析为 seq（需查库，见 _resolve_expected_ids）
                setattr(items[-1], "_expected_ids", [uuid.UUID(str(x)) for x in ids])
        except Exception as exc:
            errors.append(f"第 {i} 条：{exc}")
    if errors:
        raise ValueError("数据集校验失败：\n" + "\n".join(errors))
    if not items:
        raise ValueError("数据集为空")
    return items


def recall_at_k(anchor_seqs: list[int], expected: list[int], k: int) -> float | None:
    """top-K anchors 命中期望证据的比例；无期望（no_answer）返回 None 不计入。"""
    if not expected:
        return None
    top = set(anchor_seqs[:k])
    return len(top & set(expected)) / len(expected)


def mrr(anchor_seqs: list[int], expected: list[int]) -> float | None:
    if not expected:
        return None
    exp = set(expected)
    for rank, seq in enumerate(anchor_seqs, start=1):
        if seq in exp:
            return 1.0 / rank
    return 0.0


def percentile(values: list[int | float], p: float) -> float:
    """最近邻插值分位数；空列表返回 0。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return float(ordered[idx])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def is_refusal(text: str) -> bool:
    return any(marker in text for marker in _REFUSALS)


async def _resolve_expected(
    session, item: EvalItem
) -> tuple[list[int], list[str]]:
    """本轮的期望证据（seqs ∪ 解析后的 ids，去重）；**不修改 EvalItem**。

    未解析的 ID（不存在或属其他会议）显式返回错误说明，不静默降级——
    否则该 item 会从 scored_items 消失，报告虚高。
    """
    expected = list(dict.fromkeys(item.expected_seqs))
    notes: list[str] = []
    ids = getattr(item, "_expected_ids", None)
    if ids:
        rows = list(
            await session.scalars(
                select(TranscriptSegment).where(
                    TranscriptSegment.id.in_(ids),
                    TranscriptSegment.meeting_id == item.meeting_id,
                )
            )
        )
        found = {s.id: s.seq for s in rows}
        missing = [str(i) for i in ids if i not in found]
        if missing:
            notes.append(
                f"{item.meeting_id}「{item.question[:20]}」: "
                f"{len(missing)} 个 expected_segment_ids 未解析（不存在或属其他会议）"
            )
        for i in ids:
            if i in found and found[i] not in expected:
                expected.append(found[i])
    return expected, notes


async def evaluate(
    items: list[EvalItem],
    rrf_k: int | None = None,
    with_answers: bool = False,
) -> dict:
    """跑一轮评估；rrf_k 覆盖 settings.qa_rrf_k（结束后恢复）。

    with_answers=False（默认）只评检索（零 LLM 消耗）；True 时完整跑
    handle_chat 的回答段（用当前配置的 LLM provider，注意额度）。
    """
    # 延迟导入：qa 依赖链较重，脚本 --help 不应加载
    from app.db.session import SessionLocal
    from app.services.qa import generate_answer, retrieve_context

    old_k = settings.qa_rrf_k
    if rrf_k is not None:
        settings.qa_rrf_k = rrf_k
    results: list[ItemResult] = []
    skipped: list[str] = []
    try:
        async with SessionLocal() as session:
            for item in items:
                meeting = await session.get(Meeting, item.meeting_id)
                if meeting is None:
                    skipped.append(f"{item.meeting_id}: meeting 不存在")
                    continue
                expected, notes = await _resolve_expected(session, item)
                skipped.extend(notes)
                if not expected and item.category != "no_answer":
                    skipped.append(
                        f"{item.meeting_id}「{item.question[:20]}」: "
                        "期望证据全部未解析，跳过该题"
                    )
                    continue
                t0 = time.monotonic()
                # 每题只检索一次；--with-answers 复用同一份检索结果生成回答
                ro = await retrieve_context(
                    session, meeting, item.question, item.question
                )
                res = ItemResult(
                    item=item,
                    expected_seqs=expected,
                    anchor_seqs=[s.seq for s in ro.anchors],
                    context_seqs=[s.seq for s in ro.context],
                    candidate_count=len(ro.candidates),
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
                if with_answers:
                    # 纯生成：不写 ChatMessage——评估工具对业务数据只读
                    ans = await generate_answer(item.question, ro)
                    res.answered = not is_refusal(ans.text)
                    res.refused = not res.answered
                    res.cited_seqs = [c.seq for c in ans.citations]
                    res.confidence = ans.confidence
                    res.latency_ms = int((time.monotonic() - t0) * 1000)
                results.append(res)
    finally:
        settings.qa_rrf_k = old_k
    return _build_report(results, skipped, rrf_k or old_k, with_answers)


def _build_report(
    results: list[ItemResult], skipped: list[str], rrf_k: int, with_answers: bool
) -> dict:
    scored = [r for r in results if r.expected_seqs]
    r5 = [recall_at_k(r.anchor_seqs, r.expected_seqs, 5) for r in scored]
    r10 = [recall_at_k(r.anchor_seqs, r.expected_seqs, 10) for r in scored]
    mrrs = [mrr(r.anchor_seqs, r.expected_seqs) for r in scored]
    ctx_hits = [
        bool(set(r.expected_seqs) & set(r.context_seqs)) for r in scored
    ]
    cand_counts = [r.candidate_count for r in results]
    latencies = [r.latency_ms for r in results]

    report: dict = {
        "rrf_k": rrf_k,
        "items": len(results),
        "scored_items": len(scored),
        "skipped": skipped,
        "retrieval": {
            "recall@5": round(_mean([x for x in r5 if x is not None]), 4),
            "recall@10": round(_mean([x for x in r10 if x is not None]), 4),
            "mrr": round(_mean([x for x in mrrs if x is not None]), 4),
            "context_hit_rate": round(_mean([1.0 if h else 0.0 for h in ctx_hits]), 4),
            # §6.3 Reranker 引入条件的判断依据
            "avg_candidates": round(_mean([float(c) for c in cand_counts]), 1),
            "candidates_gt_20_rate": round(
                _mean([1.0 if c > 20 else 0.0 for c in cand_counts]), 4
            ),
            "latency_ms": {
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
            },
        },
    }
    if with_answers:
        answered = [r for r in results if r.answered is not None]
        no_answer = [r for r in answered if r.item.category == "no_answer"]
        cited_prec: list[float] = []
        for r in answered:
            if r.expected_seqs and r.cited_seqs:
                hit = len(set(r.cited_seqs) & set(r.expected_seqs))
                cited_prec.append(hit / len(r.cited_seqs))
        report["answers"] = {
            "answered_rate": round(
                _mean([1.0 if r.answered else 0.0 for r in answered]), 4
            ),
            # 无答案问题的正确拒答率（§6.2 指标）
            "no_answer_refusal_rate": round(
                _mean([1.0 if r.refused else 0.0 for r in no_answer]), 4
            )
            if no_answer
            else None,
            "citation_precision": round(_mean(cited_prec), 4) if cited_prec else None,
            "confidence_dist": {
                lvl: sum(1 for r in answered if r.confidence == lvl)
                for lvl in ("high", "medium", "low")
            },
        }
    # 按类型细分（§6.2 覆盖类型）
    by_cat: dict[str, dict] = {}
    for cat in sorted({r.item.category for r in results}):
        cat_scored = [r for r in scored if r.item.category == cat]
        if cat_scored:
            by_cat[cat] = {
                "items": len(cat_scored),
                "recall@5": round(
                    _mean(
                        [
                            x
                            for x in (
                                recall_at_k(r.anchor_seqs, r.expected_seqs, 5)
                                for r in cat_scored
                            )
                            if x is not None
                        ]
                    ),
                    4,
                ),
            }
    report["by_category"] = by_cat
    return report
