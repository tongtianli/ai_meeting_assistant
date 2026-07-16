"""RAG Phase 3（§6）：confidence 拒答完整化、离线评估 harness、RRF k 对比。"""
import asyncio
import json
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models import DEFAULT_USER_ID, Meeting, MeetingStatus, TranscriptSegment
from app.services.qa_eval import (
    EvalItem,
    evaluate,
    is_refusal,
    load_dataset,
    mrr,
    percentile,
    recall_at_k,
)
from tests.conftest import auth_headers, requires_db

pytestmark = requires_db


# ---------- §6.1 confidence 传递 ----------


async def _seed_meeting(texts: list[str]) -> uuid.UUID:
    async with SessionLocal() as session:
        meeting = Meeting(
            user_id=DEFAULT_USER_ID,
            title="P3",
            status=MeetingStatus.done,
            audio_url="local://p3.wav",
        )
        session.add(meeting)
        await session.flush()
        for i, text in enumerate(texts):
            session.add(
                TranscriptSegment(
                    meeting_id=meeting.id,
                    seq=i,
                    start_time=float(i),
                    end_time=float(i) + 1,
                    speaker_label="speaker_001",
                    text=text,
                )
            )
        await session.commit()
        return meeting.id


def _post_chat(client, headers, mid, question):
    return client.post(
        f"/api/meetings/{mid}/chat", json={"question": question}, headers=headers
    )


def test_confidence_in_post_response(tmp_path, monkeypatch) -> None:
    """正常回答：mock 带引用 → confidence=high 随 POST 响应返回。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    mid = asyncio.run(_seed_meeting(["API-203 下周三上线", "其他议题"]))
    with TestClient(app) as client:
        headers = auth_headers(client)
        body = _post_chat(client, headers, mid, "API-203 何时上线").json()
    assert body["confidence"] == "high"
    assert body["citations"]


def test_refusal_paths_force_low_confidence(tmp_path, monkeypatch) -> None:
    """事实性回答无合法引用 → 回退拒答且 confidence 恒 low（即便模型自评 high）。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    import app.services.qa as qa_mod
    from app.services.llm.base import LLMProvider, LLMResponse, LLMTaskType
    from app.services.llm.router import LLMRouter

    class _Overclaim(LLMProvider):
        name, model = "overclaim", "overclaim-1"

        async def complete(self, system, user, json_mode=True, temperature=0.2):
            return LLMResponse(
                text='{"answer": "上线日期是下周三", "cited_segment_seqs": [], '
                '"insufficient_evidence": false, "confidence": "high"}',
                provider=self.name,
                model=self.model,
            )

    real = qa_mod.build_router
    monkeypatch.setattr(
        qa_mod,
        "build_router",
        lambda task: LLMRouter([_Overclaim()])
        if task == LLMTaskType.QA_ANSWER
        else real(task),
    )
    mid = asyncio.run(_seed_meeting(["讨论了一些别的事", "散会"]))
    with TestClient(app) as client:
        headers = auth_headers(client)
        body = _post_chat(client, headers, mid, "上线日期？").json()
    assert "没有找到明确结论" in body["content"]
    assert body["confidence"] == "low"  # 程序规则压制模型的自评 high


# ---------- §6.2 指标纯函数 ----------


def test_recall_and_mrr_math() -> None:
    anchors = [7, 3, 9, 1, 5, 2]
    assert recall_at_k(anchors, [3, 9], 5) == 1.0
    assert recall_at_k(anchors, [3, 99], 5) == 0.5
    assert recall_at_k(anchors, [99], 5) == 0.0
    assert recall_at_k(anchors, [], 5) is None  # no_answer 不计入
    assert mrr(anchors, [9]) == pytest.approx(1 / 3)
    assert mrr(anchors, [99]) == 0.0
    assert mrr(anchors, []) is None


def test_percentile_and_refusal() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([10], 50) == 10.0
    vals = list(range(1, 101))
    assert percentile(vals, 50) == pytest.approx(50, abs=1)
    assert percentile(vals, 95) == pytest.approx(95, abs=1)
    assert is_refusal("会议原文中没有找到明确结论。")
    assert is_refusal("未在本次会议记录中找到相关内容。")
    assert not is_refusal("上线日期是下周三")


# ---------- §6.2 数据集加载与校验 ----------


def test_load_dataset_valid_and_invalid(tmp_path) -> None:
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            [
                {"_comment": "样例说明行被跳过"},
                {
                    "meeting_id": str(uuid.uuid4()),
                    "question": "预算多少？",
                    "expected_segment_seqs": [3],
                    "category": "amount",
                },
                {
                    "meeting_id": str(uuid.uuid4()),
                    "question": "讨论过搬迁吗？",
                    "category": "no_answer",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    items = load_dataset(good)
    assert len(items) == 2
    assert items[0].expected_seqs == [3] and items[1].expected_seqs == []

    bad_cat = tmp_path / "bad_cat.json"
    bad_cat.write_text(
        json.dumps(
            [{"meeting_id": str(uuid.uuid4()), "question": "q",
              "expected_segment_seqs": [1], "category": "nonsense"}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="category"):
        load_dataset(bad_cat)

    missing = tmp_path / "missing.json"
    missing.write_text(
        json.dumps([{"meeting_id": str(uuid.uuid4()), "question": "q"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected_segment"):
        load_dataset(missing)


def test_example_dataset_file_loads() -> None:
    items = load_dataset("eval/qa_dataset.example.json")
    assert len(items) == 3
    assert {i.category for i in items} == {"date", "amount", "no_answer"}


# ---------- §6.2 evaluate 端到端（mock embedder，零 LLM）----------


def _eval_items(mid: uuid.UUID) -> list[EvalItem]:
    return [
        EvalItem(meeting_id=mid, question="API-203 何时上线", expected_seqs=[1],
                 category="date"),
        EvalItem(meeting_id=mid, question="讨论过搬迁吗", expected_seqs=[],
                 category="no_answer"),
        EvalItem(meeting_id=uuid.uuid4(), question="不存在的会议", expected_seqs=[0]),
    ]


def test_evaluate_retrieval_only(monkeypatch) -> None:
    mid = asyncio.run(
        _seed_meeting(["先同步预算", "API-203 定于下周三上线", "散会"])
    )
    report = asyncio.run(evaluate(_eval_items(mid)))
    assert report["items"] == 2  # 不存在的会议被跳过并记录
    assert len(report["skipped"]) == 1
    assert report["scored_items"] == 1  # no_answer 不计入检索指标
    # 关键词召回保证 seq=1 进入 anchors → 全指标为正
    assert report["retrieval"]["recall@5"] == 1.0
    assert report["retrieval"]["mrr"] > 0
    assert report["retrieval"]["context_hit_rate"] == 1.0
    assert "avg_candidates" in report["retrieval"]  # §6.3 条件数据
    assert report["by_category"]["date"]["recall@5"] == 1.0


def test_evaluate_rrf_k_override_restores(monkeypatch) -> None:
    mid = asyncio.run(_seed_meeting(["API-203 上线", "其他"]))
    before = settings.qa_rrf_k
    report = asyncio.run(
        evaluate([_eval_items(mid)[0]], rrf_k=10)
    )
    assert report["rrf_k"] == 10
    assert settings.qa_rrf_k == before  # 评估后恢复配置


def test_evaluate_with_answers_mock(tmp_path, monkeypatch) -> None:
    """--with-answers：mock LLM 下产出回答率/引用准确率/置信度分布。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    # 收窄召回让 context 恰为期望片段：引用准确率可精确断言
    monkeypatch.setattr(settings, "qa_top_k", 0)
    monkeypatch.setattr(settings, "qa_neighbor_window", 0)
    mid = asyncio.run(_seed_meeting(["先同步预算", "API-203 定于下周三上线", "散会"]))
    items = [
        EvalItem(meeting_id=mid, question="API-203 何时上线", expected_seqs=[1],
                 category="date"),
    ]
    report = asyncio.run(evaluate(items, with_answers=True))
    answers = report["answers"]
    assert answers["answered_rate"] == 1.0
    assert answers["citation_precision"] == 1.0  # mock 引用 context 最小 seq = 期望 seq
    assert answers["confidence_dist"]["high"] == 1
