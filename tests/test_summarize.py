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
    chunks = chunk_lines(lines, budget=500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 + 60 for c in chunks)
    assert "\n".join(lines) == "\n".join(chunks).replace("\n\n", "\n") or sum(
        c.count("[") for c in chunks
    ) == len(lines)


def test_chunk_lines_single_chunk_when_small() -> None:
    assert len(chunk_lines(["短行"] * 3, budget=1000)) == 1


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
        summarize_mod, "build_router", lambda: LLMRouter([_BrokenJSONProvider()])
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
