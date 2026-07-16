"""RAG Phase 2（§5 + 评审 §8）：关键词/精确实体提取与召回、LIKE 转义、
每 token 限额、RRF 融合与可观测性。"""
import asyncio
import uuid
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models import DEFAULT_USER_ID, Meeting, MeetingStatus, TranscriptSegment
from app.services.retrieval import (
    _escape_like,
    extract_keywords,
    retrieve_keyword_anchors,
    rrf_fuse,
)
from tests.conftest import auth_headers, requires_db

pytestmark = requires_db


# ---------- 关键词提取（§5.2 / 评审 §8.1）----------


def test_extract_entities_bypass_min_length() -> None:
    q = "API-203 在 v2.1.4 上线，预算 29.5 万，7月18 验收，方案见 plan.docx"
    tokens = extract_keywords(q)
    assert "API-203" in tokens
    assert "v2.1.4" in tokens
    assert any(t.replace(" ", "") == "29.5万" for t in tokens)
    assert any("7月18" in t.replace(" ", "") for t in tokens)
    assert "plan.docx" in tokens


def test_extract_no_bare_numbers() -> None:
    # 裸数字（seq/页码/时长类噪声）不提取；带单位的金额才算
    tokens = extract_keywords("第 42 页提到 300 人参加，费用 300 元")
    assert "42" not in tokens and "300" not in tokens
    assert any(t.replace(" ", "") == "300元" for t in tokens)


def test_extract_english_min_length_and_no_substring_of_entity() -> None:
    tokens = extract_keywords("API-203 的 ab 接口用 OAuth 鉴权")
    assert "OAuth" in tokens
    assert "ab" not in tokens  # 短于最短长度
    assert "API" not in tokens  # 已被实体 API-203 覆盖，不再吐子串


def test_extract_caps_and_dedupes(monkeypatch) -> None:
    monkeypatch.setattr(settings, "qa_keyword_max_tokens", 3)
    q = "AAA-1 BBB-2 CCC-3 DDD-4 AAA-1"
    tokens = extract_keywords(q)
    assert len(tokens) == 3 and tokens == ["AAA-1", "BBB-2", "CCC-3"]
    monkeypatch.setattr(settings, "qa_keyword_max_tokens", 0)
    assert extract_keywords(q) == []


def test_escape_like_literals() -> None:
    assert _escape_like("100%") == "100\\%"
    assert _escape_like("a_b") == "a\\_b"
    assert _escape_like("c\\d") == "c\\\\d"


# ---------- 关键词召回（评审 §8.2：转义/限额/会议隔离）----------


async def _seed_meeting(texts: list[str]) -> uuid.UUID:
    async with SessionLocal() as session:
        meeting = Meeting(
            user_id=DEFAULT_USER_ID,
            title="P2",
            status=MeetingStatus.done,
            audio_url="local://p2.wav",
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


def test_keyword_recall_exact_and_escaped() -> None:
    async def _run():
        mid = await _seed_meeting(
            ["进度占比 100% 完成", "编号 1004 的工单", "API-203 下周上线"]
        )
        async with SessionLocal() as session:
            hits = await retrieve_keyword_anchors(session, mid, ["100%", "API-203"], 4)
            return {tok: [s.seq for s in segs] for tok, segs in hits.items()}

    hits = asyncio.run(_run())
    assert hits["100%"] == [0]  # % 被转义为字面量：不通配命中 "1004"
    assert hits["API-203"] == [2]


def test_keyword_recall_space_variant() -> None:
    async def _run():
        mid = await _seed_meeting(["预算定为29.5万元，下季度执行"])
        async with SessionLocal() as session:
            hits = await retrieve_keyword_anchors(session, mid, ["29.5 万"], 4)
            return [s.seq for s in hits["29.5 万"]]

    assert asyncio.run(_run()) == [0]  # 查询带空格也命中原文无空格写法


def test_keyword_recall_per_token_limit_and_meeting_scope() -> None:
    async def _run():
        mid_a = await _seed_meeting([f"OKR 复盘第 {i} 项" for i in range(6)])
        mid_b = await _seed_meeting(["OKR 另一场会议"])
        async with SessionLocal() as session:
            hits_a = await retrieve_keyword_anchors(session, mid_a, ["OKR"], 2)
            return [s.meeting_id for s in hits_a["OKR"]], len(hits_a["OKR"]), mid_a, mid_b

    mids, n, mid_a, _ = asyncio.run(_run())
    assert n == 2  # 每 token 独立限额
    assert all(m == mid_a for m in mids)  # 恒带 meeting_id 过滤


# ---------- RRF 融合（§5.3 / 评审 §8.3）----------


@dataclass
class _Seg:
    seq: int
    id: uuid.UUID = field(default_factory=uuid.uuid4)


def test_rrf_multi_source_beats_single_source_same_rank() -> None:
    both = _Seg(seq=1)
    only_v = _Seg(seq=2)
    cands = rrf_fuse([both, only_v], [both], [], k=60)
    assert cands[0].segment is both  # 双来源 1/(60+1)*2 > 单来源
    assert cands[0].sources == {"vector", "speaker"}
    assert cands[0].vector_rank == 1 and cands[0].speaker_rank == 1
    assert cands[0].fused_score == pytest.approx(2 / 61)


def test_rrf_dedupes_and_tie_breaks_deterministically() -> None:
    a, b = _Seg(seq=9), _Seg(seq=3)
    # 同分（各自 vector rank 相同不可能——用两源对称构造）：a vector#1，b speaker#1
    cands = rrf_fuse([a], [b], [], k=60)
    assert [c.segment.seq for c in cands] == [3, 9]  # 同分同 best_rank → seq 小者先
    assert len(cands) == 2


def test_rrf_k_configurable_changes_spread() -> None:
    a, b = _Seg(seq=1), _Seg(seq=2)
    low_k = rrf_fuse([a, b], [], [], k=10)
    high_k = rrf_fuse([a, b], [], [], k=60)
    gap_low = low_k[0].fused_score - low_k[1].fused_score
    gap_high = high_k[0].fused_score - high_k[1].fused_score
    assert gap_low > gap_high  # k 越小 rank 区分度越大（评审 §8.3 对比 10/30/60 的依据）


# ---------- e2e：精确实体查询命中原文（§5.4 验收）----------


def _post_chat(client, headers, mid, question):
    return client.post(
        f"/api/meetings/{mid}/chat", json={"question": question}, headers=headers
    )


def test_keyword_query_hits_exact_segment(tmp_path, monkeypatch) -> None:
    """向量召回关闭、无说话人绑定：API-203 仅靠关键词召回命中原文。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "qa_top_k", 0)
    monkeypatch.setattr(settings, "qa_neighbor_window", 0)
    mid = asyncio.run(
        _seed_meeting(["先同步预算情况", "API-203 接口定于下周三上线", "散会"])
    )
    with TestClient(app) as client:
        headers = auth_headers(client)
        body = _post_chat(client, headers, mid, "API-203 什么时候上线？").json()
    assert body["citations"], body
    assert body["citations"][0]["seq"] == 1
    assert "API-203" in body["citations"][0]["text"]


def test_keyword_sources_visible_in_log(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "qa_top_k", 0)
    mid = asyncio.run(_seed_meeting(["API-203 下周上线", "其他议题"]))
    with TestClient(app) as client:
        headers = auth_headers(client)
        with caplog.at_level("INFO", logger="app.services.qa"):
            _post_chat(client, headers, mid, "API-203 进展如何")
    logged = "\n".join(
        r.getMessage() for r in caplog.records if "qa retrieval" in r.getMessage()
    )
    assert '"keyword": 1' in logged  # 策略计数含 keyword
    assert '"candidate_sources"' in logged and '"keyword"' in logged  # 候选来源可见
    assert "API-203" not in logged.replace("candidate_sources", "")  # 默认仍无明文
