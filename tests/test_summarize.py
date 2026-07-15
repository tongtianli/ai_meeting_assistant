"""摘要服务：chunking、端到端入库、溯源反查、降级路径。"""
import asyncio

import pytest
from sqlalchemy import select

from app.core.config import settings
from app.db.session import SessionLocal
from app.models import ActionItem, DEFAULT_USER_ID, Meeting, Summary
from app.services.llm.base import LLMError, LLMProvider, LLMResponse
from app.services.summarize import chunk_lines, summarize_meeting
from tests.conftest import make_wav, requires_db


def test_chunk_lines_respects_budget_and_keeps_all() -> None:
    lines = [f"[{i}] 内容" + "x" * 50 for i in range(100)]
    # chars_per_token=1 → est_tokens==字符数，等价于旧的字符预算；无重叠时不丢不重
    chunks = chunk_lines(
        lines, target_tokens=500, chars_per_token=1, overlap_segments=0
    )
    assert len(chunks) > 1
    assert sum(c.count("[") for c in chunks) == len(lines)


def test_chunk_lines_single_chunk_when_small() -> None:
    assert (
        len(chunk_lines(["短行"] * 3, target_tokens=1000, chars_per_token=1)) == 1
    )


def test_chunk_lines_overlap_carries_tail() -> None:
    lines = [f"[{i}] line{i}" for i in range(12)]
    chunks = chunk_lines(
        lines, target_tokens=20, max_tokens=30, overlap_segments=2, chars_per_token=1
    )
    assert len(chunks) > 1
    # 相邻块重叠：上一块末尾 2 行 == 下一块开头 2 行
    for a, b in zip(chunks, chunks[1:]):
        assert a.split("\n")[-2:] == b.split("\n")[:2]
    # 不丢内容：每行至少出现在某块
    joined = "\n".join(chunks)
    assert all(line in joined for line in lines)


@requires_db
def test_summarize_persists_summary_and_action_items(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    async def scenario() -> None:
        from app.services.pipeline import run_pipeline
        from app.services.storage import get_audio_storage

        src = make_wav(tmp_path / "s.wav", seconds=3.0)
        with src.open("rb") as f:
            audio_url = get_audio_storage().save(f, "s.wav")
        async with SessionLocal() as session:
            meeting = Meeting(
                user_id=DEFAULT_USER_ID, title="摘要测试", audio_url=audio_url
            )
            session.add(meeting)
            await session.commit()
            mid = meeting.id

        try:
            await run_pipeline(mid)
            async with SessionLocal() as session:
                meeting = await session.get(Meeting, mid)
                assert meeting.status.value == "done"

                summary = await session.scalar(
                    select(Summary).where(Summary.meeting_id == mid)
                )
                assert summary is not None
                assert summary.version == 1
                content = summary.content_json
                assert content["title"]
                assert content["_meta"]["model"].startswith("mock/")
                assert content["_meta"]["degraded"] is False

                items = list(
                    await session.scalars(
                        select(ActionItem).where(ActionItem.meeting_id == mid)
                    )
                )
                assert len(items) == 1
                item = items[0]
                assert item.task == "完成上传接口的开发"
                # 溯源：mock LLM 引用 seq=3，程序反查到真实 segment id
                assert item.source_segment_id is not None
                assert item.owner_text == "speaker_001"

                # 重跑 → Summary 版本递增、ActionItem 不重复
                await run_pipeline(mid)
            async with SessionLocal() as session:
                versions = sorted(
                    await session.scalars(
                        select(Summary.version).where(Summary.meeting_id == mid)
                    )
                )
                assert versions == [1, 2]
                n_items = len(
                    list(
                        await session.scalars(
                            select(ActionItem).where(ActionItem.meeting_id == mid)
                        )
                    )
                )
                assert n_items == 1
        finally:
            async with SessionLocal() as session:
                meeting = await session.get(Meeting, mid)
                if meeting is not None:
                    await session.delete(meeting)
                    await session.commit()

    asyncio.run(scenario())


@requires_db
def test_long_meeting_map_reduce_dedup(tmp_path, monkeypatch) -> None:
    """长会议走 map-reduce：map 多次调用（MapFacts）、跨块 TODO 不丢且不重复。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    # 极小分块，让 7 段 mock 逐字稿分成多块，触发 map-reduce
    monkeypatch.setattr(settings, "summary_chunk_target_tokens", 30)
    monkeypatch.setattr(settings, "summary_chunk_max_tokens", 40)
    monkeypatch.setattr(settings, "summary_chunk_overlap_segments", 2)
    monkeypatch.setattr(settings, "summary_chars_per_token", 1.0)

    import app.services.summarize as summarize_mod

    calls = {"map": 0}
    real_map_prompt = summarize_mod.map_prompt

    def counting_map_prompt(*a, **k):
        calls["map"] += 1
        return real_map_prompt(*a, **k)

    monkeypatch.setattr(summarize_mod, "map_prompt", counting_map_prompt)

    async def scenario() -> None:
        from app.services.pipeline import run_pipeline
        from app.services.storage import get_audio_storage

        src = make_wav(tmp_path / "long.wav", seconds=3.0)
        with src.open("rb") as f:
            audio_url = get_audio_storage().save(f, "long.wav")
        async with SessionLocal() as session:
            meeting = Meeting(
                user_id=DEFAULT_USER_ID, title="长会议", audio_url=audio_url
            )
            session.add(meeting)
            await session.commit()
            mid = meeting.id
        try:
            await run_pipeline(mid)
            assert calls["map"] > 1  # 多块 → 走了 map-reduce（而非 single-pass）
            async with SessionLocal() as session:
                meeting = await session.get(Meeting, mid)
                assert meeting.status.value == "done"
                summary = await session.scalar(
                    select(Summary).where(Summary.meeting_id == mid)
                )
                assert summary.content_json["title"]  # reduce 产出合法 SummaryContent
                # 清洁纪要不写 quality 字段
                assert "quality" not in summary.content_json["_meta"]
                items = list(
                    await session.scalars(
                        select(ActionItem).where(ActionItem.meeting_id == mid)
                    )
                )
                assert len(items) == 1  # 跨块 TODO 合并为一，不因重叠重复入库
        finally:
            async with SessionLocal() as session:
                meeting = await session.get(Meeting, mid)
                if meeting is not None:
                    await session.delete(meeting)
                    await session.commit()

    asyncio.run(scenario())


_VALID_SUMMARY = {
    "title": "会议",
    "participants": ["speaker_001"],
    "summary": "讨论了上传接口开发。",
    "topics": [{"title": "进展", "owner": "speaker_001", "items": ["接口推进中。"]}],
    "decisions": [],
    "todos": [
        {"task": "完成上传接口", "owner": "speaker_001", "deadline": "周五",
         "source_segment_seq": 4}
    ],
}


@requires_db
def test_high_risk_regenerates_once(tmp_path, monkeypatch) -> None:
    """首次纪要命中高风险（无效引用 seq）→ 自动重跑一次、标 regenerated。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    import copy

    import app.services.summarize as summarize_mod

    calls = {"n": 0}

    async def fake_generate(lines, examples):
        calls["n"] += 1
        content = copy.deepcopy(_VALID_SUMMARY)
        if calls["n"] == 1:
            content["todos"][0]["source_segment_seq"] = 999  # 不存在 → 高风险
        return content, "mock/x", False

    monkeypatch.setattr(summarize_mod, "_generate_content", fake_generate)

    async def scenario() -> None:
        from app.services.pipeline import run_pipeline
        from app.services.storage import get_audio_storage

        src = make_wav(tmp_path / "hr.wav", seconds=2.0)
        with src.open("rb") as f:
            audio_url = get_audio_storage().save(f, "hr.wav")
        async with SessionLocal() as session:
            meeting = Meeting(
                user_id=DEFAULT_USER_ID, title="高风险重跑", audio_url=audio_url
            )
            session.add(meeting)
            await session.commit()
            mid = meeting.id
        try:
            await run_pipeline(mid)
            assert calls["n"] == 2  # 重跑一次
            async with SessionLocal() as session:
                summary = await session.scalar(
                    select(Summary).where(Summary.meeting_id == mid)
                )
                q = summary.content_json["_meta"].get("quality", {})
                assert q.get("regenerated") is True
                # 第二次为干净结果，无残留高风险项
                assert "invalid_todo_seqs" not in q
        finally:
            async with SessionLocal() as session:
                meeting = await session.get(Meeting, mid)
                if meeting is not None:
                    await session.delete(meeting)
                    await session.commit()

    asyncio.run(scenario())


class _BrokenJSONProvider(LLMProvider):
    """结构化输出永远失败、纯文本成功——驱动降级路径。"""

    name = "broken"
    model = "broken-model"

    async def complete(self, system, user, json_mode=True, temperature=0.2):
        if json_mode:
            return LLMResponse(text="这不是 JSON", provider=self.name, model=self.model)
        return LLMResponse(
            text="纯文本纪要：讨论了项目进展。", provider=self.name, model=self.model
        )


@requires_db
def test_summarize_degrades_to_plain_text(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    import app.services.summarize as summarize_mod
    from app.services.llm.router import LLMRouter

    monkeypatch.setattr(
        summarize_mod, "build_router", lambda *a, **k: LLMRouter([_BrokenJSONProvider()])
    )

    async def scenario() -> None:
        from app.services.pipeline import run_pipeline
        from app.services.storage import get_audio_storage

        src = make_wav(tmp_path / "d.wav", seconds=2.0)
        with src.open("rb") as f:
            audio_url = get_audio_storage().save(f, "d.wav")
        async with SessionLocal() as session:
            meeting = Meeting(
                user_id=DEFAULT_USER_ID, title="降级测试", audio_url=audio_url
            )
            session.add(meeting)
            await session.commit()
            mid = meeting.id
        try:
            await run_pipeline(mid)
            async with SessionLocal() as session:
                meeting = await session.get(Meeting, mid)
                assert meeting.status.value == "done"  # 降级仍算成功
                summary = await session.scalar(
                    select(Summary).where(Summary.meeting_id == mid)
                )
                assert summary.content_json["_meta"]["degraded"] is True
                assert "纯文本纪要" in summary.content_json["text"]
                items = list(
                    await session.scalars(
                        select(ActionItem).where(ActionItem.meeting_id == mid)
                    )
                )
                assert items == []  # 降级时无结构化 TODO
        finally:
            async with SessionLocal() as session:
                meeting = await session.get(Meeting, mid)
                if meeting is not None:
                    await session.delete(meeting)
                    await session.commit()

    asyncio.run(scenario())
