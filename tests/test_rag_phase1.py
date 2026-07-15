"""RAG Phase 1（TECH_DESIGN_MEETING_RAG_V1 §4.1.1）：
合并意图+改写、anchor 优先预算、保底 speaker anchor、说话人并集、
最小拒答、embedding 模型身份、脱敏检索日志。"""
import asyncio
import io
import uuid
import wave
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models import Meeting, TranscriptSegment
from app.schemas.chat import ChatIntent
from app.services.llm.base import LLMProvider, LLMResponse
from app.services.qa import (
    budget_context,
    ensure_embeddings,
    match_speakers_union,
    merge_anchors,
)
from app.services.query_intent import classify_and_rewrite, has_anaphora
from tests.conftest import auth_headers, requires_db

pytestmark = requires_db


# ---------- 纯逻辑单元测试用的假 segment ----------


@dataclass
class _Seg:
    seq: int
    id: uuid.UUID = field(default_factory=uuid.uuid4)


def _segs(*seqs: int) -> list[_Seg]:
    return [_Seg(seq=s) for s in seqs]


def _by_seq(segs: list[_Seg]) -> dict[int, _Seg]:
    return {s.seq: s for s in segs}


# ---------- anchor 优先预算（§4.1.1-D）----------


def test_budget_high_rank_late_anchor_survives_truncation() -> None:
    """高排名 anchor 位于会议末尾（seq=500），不能因按 seq 截断而丢失。"""
    universe = _segs(*range(0, 30), *range(495, 506))
    m = _by_seq(universe)
    anchors = [m[500], m[20], m[5], m[10]]  # rank 0 = seq500
    ctx, stats = budget_context(anchors, m, window=2, max_segments=6)
    assert m[500] in ctx  # 最高排名 anchor 恒保留
    assert [s.seq for s in ctx] == sorted(s.seq for s in ctx)  # 展示按 seq 排序
    assert stats.context_segment_count == 6 and stats.truncated


def test_budget_anchor_priority_over_neighbors() -> None:
    """预算 = anchor 数时，全部 anchor 入选、邻居一个不进。"""
    universe = _segs(*range(0, 60))
    m = _by_seq(universe)
    anchors = [m[50], m[10], m[30]]
    ctx, stats = budget_context(anchors, m, window=2, max_segments=3)
    assert {s.seq for s in ctx} == {10, 30, 50}
    assert stats.kept_anchor_count == 3 and stats.evicted_anchor_count == 0


def test_budget_evicts_lowest_rank_anchor_when_over_cap() -> None:
    universe = _segs(*range(0, 60))
    m = _by_seq(universe)
    anchors = [m[50], m[10], m[30], m[40]]  # rank 序
    ctx, stats = budget_context(anchors, m, window=1, max_segments=3)
    assert {s.seq for s in ctx} == {50, 10, 30}  # 淘汰 rank 最低的 seq=40
    assert stats.evicted_anchor_count == 1 and stats.truncated


def test_budget_neighbors_fill_by_rounds_and_dedupe() -> None:
    """邻居按 ±1、±2 逐圈补齐；重叠邻居去重、不重复计数。"""
    universe = _segs(*range(0, 20))
    m = _by_seq(universe)
    anchors = [m[5], m[7]]  # 邻居 6 重叠
    ctx, stats = budget_context(anchors, m, window=2, max_segments=20)
    assert [s.seq for s in ctx] == [3, 4, 5, 6, 7, 8, 9]
    assert stats.context_segment_count == 7 and not stats.truncated


def test_budget_first_ring_before_second_ring() -> None:
    """±1 圈优先于任何 anchor 的 ±2 圈（不是一个 anchor 一次吃满窗口）。"""
    universe = _segs(*range(0, 40))
    m = _by_seq(universe)
    anchors = [m[10], m[30]]
    # 预算 6 = 2 anchor + 4 邻居：应是两个 anchor 各自的 ±1，绝无 ±2
    ctx, _ = budget_context(anchors, m, window=2, max_segments=6)
    assert {s.seq for s in ctx} == {9, 10, 11, 29, 30, 31}


# ---------- 保底 speaker anchor（§4.1.1-E）----------


def test_merge_anchors_protected_speaker_first() -> None:
    p1, p2 = uuid.uuid4(), uuid.uuid4()
    v = _segs(1, 2, 3)
    s1, s2 = _segs(11, 12), _segs(21, 22)
    ordered = merge_anchors(
        v, {p1: s1, p2: s2}, person_order=[p1, p2], min_protected_per_person=1
    )
    # 受保护（每人第 1 条）最先，其后 vector，最后剩余 speaker
    assert [s.seq for s in ordered] == [11, 21, 1, 2, 3, 12, 22]


def test_merge_anchors_dedupes_across_sources() -> None:
    p1 = uuid.uuid4()
    shared = _Seg(seq=5)
    ordered = merge_anchors(
        [shared, _Seg(seq=6)], {p1: [shared]}, [p1], min_protected_per_person=1
    )
    assert [s.seq for s in ordered] == [5, 6]  # 同 segment 只保留一次（受保护位）


def test_named_person_anchor_survives_vector_flood() -> None:
    """点名问题：vector 候选占满预算时，受保护 speaker anchor 仍保留。"""
    p1 = uuid.uuid4()
    vector = _segs(*range(0, 10))
    speaker = _segs(99)
    universe = vector + speaker
    ordered = merge_anchors(vector, {p1: speaker}, [p1], 1)
    ctx, _ = budget_context(ordered, _by_seq(universe), window=0, max_segments=5)
    assert 99 in {s.seq for s in ctx}  # 保底在 rank 首位，预算再紧也在


# ---------- 说话人并集与臆造防护（§4.1.1-C）----------


def test_speaker_union_keeps_name_dropped_by_rewrite() -> None:
    pid1, pid2 = uuid.uuid4(), uuid.uuid4()
    names = {"speaker_001": (pid1, "张三"), "speaker_002": (pid2, "李四")}
    # 原问题含李四，改写把李四弄丢了 → 并集仍保留
    pids, suspicious, _ = match_speakers_union(
        names, "张三和李四分别说了什么", "张三说了什么", "张三和李四分别说了什么"
    )
    assert pids == [pid1, pid2] and suspicious == []


def test_speaker_union_ignores_invented_name() -> None:
    pid1, pid2 = uuid.uuid4(), uuid.uuid4()
    names = {"speaker_001": (pid1, "张三"), "speaker_002": (pid2, "王五")}
    # 改写臆造王五（原问题/历史/引用均无该名字）→ 不做强召回，记可疑
    pids, suspicious, _ = match_speakers_union(
        names, "张三说了什么", "张三和王五说了什么", "张三说了什么\n上一轮问的是风险"
    )
    assert pids == [pid1] and suspicious == ["王五"]


def test_speaker_union_caps_persons_by_first_occurrence(monkeypatch) -> None:
    monkeypatch.setattr(settings, "qa_max_speaker_persons", 2)
    ids = [uuid.uuid4() for _ in range(3)]
    names = {
        f"speaker_{i}": (ids[i], nm) for i, nm in enumerate(["甲乙", "丙丁", "戊己"])
    }
    q = "甲乙、丙丁和戊己都说了什么"
    pids, _, truncated = match_speakers_union(names, q, q, q)
    assert pids == [ids[0], ids[1]] and truncated  # 按首次出现序保留前 2 人


# ---------- 合并意图+改写（§4.1.1-A/B）----------


class _CapturedRouter:
    """捕获 prompt 的假 router；按脚本返回 ChatIntent。"""

    def __init__(self, result: ChatIntent) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    async def generate_json(self, system, user, schema):
        self.calls.append((system, user))
        return self.result, LLMResponse(text="", provider="fake", model="fake")


def _patch_router(monkeypatch, router) -> None:
    import app.services.query_intent as qi

    monkeypatch.setattr(qi, "build_router", lambda task: router)


def test_first_turn_no_history_block_and_original_standalone(monkeypatch) -> None:
    router = _CapturedRouter(ChatIntent(intent="query", standalone_query="被改过"))
    _patch_router(monkeypatch, router)
    res = asyncio.run(classify_and_rewrite("张三说了什么", [], ["张三"], []))
    assert len(router.calls) == 1  # 仅一次调用（意图分类），无独立改写调用
    assert "最近的用户提问" not in router.calls[0][1]  # 首轮不注入历史块
    assert res.standalone_query == "张三说了什么"  # 无历史时改写不生效
    assert not res.rewritten


def test_multi_turn_single_call_returns_intent_and_standalone(monkeypatch) -> None:
    router = _CapturedRouter(
        ChatIntent(intent="query", standalone_query="张三后来给方案了吗")
    )
    _patch_router(monkeypatch, router)
    res = asyncio.run(
        classify_and_rewrite("他后来给方案了吗", ["张三提了什么风险"], ["张三"], [])
    )
    assert len(router.calls) == 1  # 意图+改写一次调用
    assert "最近的用户提问" in router.calls[0][1]
    assert res.intent == "query" and res.standalone_query == "张三后来给方案了吗"
    assert res.rewritten


def test_rewrite_failure_degrades_to_original(monkeypatch) -> None:
    from app.services.llm import LLMExhaustedError

    class _Broken:
        async def generate_json(self, *a, **k):
            raise LLMExhaustedError("down")

    _patch_router(monkeypatch, _Broken())
    res = asyncio.run(classify_and_rewrite("把标题改一下", ["历史"], [], []))
    assert res.intent == "query" and res.standalone_query == "把标题改一下"


def test_cited_context_gated_by_anaphora(monkeypatch) -> None:
    router = _CapturedRouter(ChatIntent(intent="query", standalone_query="x"))
    _patch_router(monkeypatch, router)
    cited = ["王建国：我提议延期到下周三。"]
    # 无指代 → 不附带引用原文
    asyncio.run(classify_and_rewrite("会议结论是什么", ["历史问题"], [], cited))
    assert "王建国" not in router.calls[-1][1]
    # 有指代 → 附带
    asyncio.run(classify_and_rewrite("他给的原因是什么", ["谁提议延期"], [], cited))
    assert "王建国" in router.calls[-1][1]
    assert has_anaphora("他给的原因是什么") and not has_anaphora("会议结论是什么")


def test_cited_context_budget(monkeypatch) -> None:
    monkeypatch.setattr(settings, "qa_rewrite_cited_max_segments", 2)
    monkeypatch.setattr(settings, "qa_rewrite_cited_max_chars", 10)
    router = _CapturedRouter(ChatIntent(intent="query", standalone_query="x"))
    _patch_router(monkeypatch, router)
    cited = ["短句一。", "这是一条超过十个字符预算的很长引用原文", "短句二。"]
    asyncio.run(classify_and_rewrite("他呢", ["历史"], [], cited))
    user = router.calls[-1][1]
    assert "短句一" in user
    assert "很长引用原文" not in user  # 超字符预算截断
    assert "短句二" not in user  # 超条数预算


# ---------- e2e：多轮指代改写驱动检索 + 历史时序 ----------


def _wav_bytes(seconds: float = 3.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


def _done_meeting(client, headers, title="RAG P1") -> str:
    resp = client.post(
        "/api/meetings",
        files={"file": ("r.wav", _wav_bytes(), "audio/wav")},
        data={"title": title},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    mid = resp.json()["id"]
    assert client.get(f"/api/meetings/{mid}", headers=headers).json()["status"] == "done"
    return mid


def test_multi_turn_pronoun_resolved_via_rewrite(tmp_path, monkeypatch) -> None:
    """第二轮"他…"经 mock 确定性改写为"张三…"，speaker 召回生效。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "qa_top_k", 0)  # 关向量召回，命中只能来自改写
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)
        client.post(
            f"/api/meetings/{mid}/speaker-bindings",
            json={"speaker_label": "speaker_001", "name": "张三"},
            headers=headers,
        )
        r1 = client.post(
            f"/api/meetings/{mid}/chat",
            json={"question": "张三说了什么"},
            headers=headers,
        )
        assert r1.json()["citations"], r1.text
        # 第二轮：原问题无"张三"，仅指代——改写后仍召回张三片段
        r2 = client.post(
            f"/api/meetings/{mid}/chat",
            json={"question": "他还说过别的吗"},
            headers=headers,
        )
        body = r2.json()
        assert body["citations"], body
        assert body["citations"][0]["speaker_name"] == "张三"


def test_history_excludes_current_message(tmp_path, monkeypatch) -> None:
    """时序修复：改写输入的历史不含本轮问题、不含 assistant 文本。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    captured: list[list[str]] = []

    import app.services.qa as qa_mod
    from app.services.query_intent import IntentResult

    async def _spy(question, recent, speakers, cited=None):
        captured.append(list(recent))
        return IntentResult("query", question, False, False)

    monkeypatch.setattr(qa_mod, "classify_and_rewrite", _spy)
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)
        client.post(
            f"/api/meetings/{mid}/chat", json={"question": "第一问"}, headers=headers
        )
        client.post(
            f"/api/meetings/{mid}/chat", json={"question": "第二问"}, headers=headers
        )
    assert captured[0] == []  # 首轮无历史
    assert captured[1] == ["第一问"]  # 不含"第二问"本身，也不含 assistant 回复


# ---------- 最小拒答（§4.1.1-F）----------


class _QAProvider(LLMProvider):
    def __init__(self, payload: str) -> None:
        self.payload = payload

    name = "fakeqa"
    model = "fakeqa-model"

    async def complete(self, system, user, json_mode=True, temperature=0.2):
        return LLMResponse(text=self.payload, provider=self.name, model=self.model)


def _force_qa_payload(monkeypatch, payload: str) -> None:
    import app.services.qa as qa_mod
    from app.services.llm.router import LLMRouter

    real = qa_mod.build_router

    def _router(task):
        from app.services.llm.base import LLMTaskType

        if task == LLMTaskType.QA_ANSWER:
            return LLMRouter([_QAProvider(payload)])
        return real(task)

    monkeypatch.setattr(qa_mod, "build_router", _router)


def test_factual_answer_without_citation_becomes_refusal(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    _force_qa_payload(
        monkeypatch,
        '{"answer": "上线日期是下周三", "cited_segment_seqs": [], '
        '"insufficient_evidence": false}',
    )
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)
        body = client.post(
            f"/api/meetings/{mid}/chat", json={"question": "上线日期？"}, headers=headers
        ).json()
    # 无据事实性回答不得流出 → 回退为明确拒答且无引用
    assert body["content"] == "会议原文中没有找到明确结论。"
    assert body["citations"] == []


def test_insufficient_evidence_refusal_honored(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    _force_qa_payload(
        monkeypatch,
        '{"answer": "会议原文中没有找到明确结论。", "cited_segment_seqs": [3], '
        '"insufficient_evidence": true}',
    )
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)
        body = client.post(
            f"/api/meetings/{mid}/chat", json={"question": "预算多少？"}, headers=headers
        ).json()
    assert "没有找到明确结论" in body["content"]
    assert body["citations"] == []  # 拒答不携带引用（即便模型给了）


# ---------- embedding 模型身份（§4.1.1-G）----------


def test_model_key_change_same_dim_triggers_reembed(tmp_path, monkeypatch) -> None:
    """同维度但模型身份不同 → 整场重嵌并回填 key（只比维度会静默漏过）。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)
        client.post(
            f"/api/meetings/{mid}/chat", json={"question": "问题一"}, headers=headers
        )

    async def _tamper_key() -> str:
        async with SessionLocal() as session:
            meeting = await session.get(Meeting, uuid.UUID(mid))
            old = meeting.embedding_model_key
            assert old == "mock/mock-embed:64"  # 首问已回填身份键
            meeting.embedding_model_key = "legacy/other-model:64"  # 同维不同模型
            await session.commit()
            return old

    expected_key = asyncio.run(_tamper_key())

    calls = {"n": 0}

    async def _counting(session, meeting, segments, expected_dim):
        calls["n"] += 1
        return await ensure_embeddings(session, meeting, segments, expected_dim)

    import app.services.qa as qa_mod

    monkeypatch.setattr(qa_mod, "ensure_embeddings", _counting)
    with TestClient(app) as client:
        headers = auth_headers(client)
        resp = client.post(
            f"/api/meetings/{mid}/chat", json={"question": "问题二"}, headers=headers
        )
        assert resp.status_code == 201

    async def _check() -> None:
        async with SessionLocal() as session:
            meeting = await session.get(Meeting, uuid.UUID(mid))
            assert meeting.embedding_model_key == expected_key  # 重嵌后回填当前身份
            rows = list(
                await session.scalars(
                    select(TranscriptSegment).where(
                        TranscriptSegment.meeting_id == uuid.UUID(mid)
                    )
                )
            )
            assert all(len(s.embedding) == 64 for s in rows)

    asyncio.run(_check())


def test_same_model_same_dim_no_reembed(tmp_path, monkeypatch) -> None:
    """模型与维度均一致时不重复重嵌（embed 只为问题向量被调用）。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)
        client.post(
            f"/api/meetings/{mid}/chat", json={"question": "问题一"}, headers=headers
        )

    embed_batches: list[int] = []
    from app.services.embeddings.mock import MockEmbedder

    real_embed = MockEmbedder.embed

    async def _spy(self, texts):
        embed_batches.append(len(texts))
        return await real_embed(self, texts)

    monkeypatch.setattr(MockEmbedder, "embed", _spy)
    with TestClient(app) as client:
        headers = auth_headers(client)
        client.post(
            f"/api/meetings/{mid}/chat", json={"question": "问题二"}, headers=headers
        )
    assert embed_batches == [1]  # 仅问题向量；未触发整场重嵌（7 段会是一批 7）


# ---------- 检索日志默认脱敏（§4.1.1-H）----------


def test_retrieval_log_no_plaintext_by_default(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    question = "上传接口谁负责这件绝密事项"
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)
        with caplog.at_level("INFO", logger="app.services.qa"):
            client.post(
                f"/api/meetings/{mid}/chat", json={"question": question}, headers=headers
            )
    logged = "\n".join(r.getMessage() for r in caplog.records if "qa retrieval" in r.getMessage())
    assert logged  # 有结构化检索日志
    assert question not in logged  # 默认不含明文
    assert "question_hash" in logged and "sha256:" in logged


def test_retrieval_log_plaintext_with_debug_flag(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "qa_retrieval_debug_text", True)
    question = "上传接口谁负责"
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _done_meeting(client, headers)
        with caplog.at_level("INFO", logger="app.services.qa"):
            client.post(
                f"/api/meetings/{mid}/chat", json={"question": question}, headers=headers
            )
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert question in logged  # 显式开启才记录明文
