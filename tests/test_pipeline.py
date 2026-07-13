"""管道端到端集成测试（真实 PostgreSQL + ffmpeg + Mock ASR）。"""
import asyncio

from sqlalchemy import select

from app.core.config import settings
from app.db.session import SessionLocal
from app.models import (
    DEFAULT_USER_ID,
    Meeting,
    MeetingStatus,
    TranscriptSegment,
    VoiceSample,
)
from app.services.pipeline import run_pipeline
from app.services.storage import get_audio_storage
from tests.conftest import make_wav, requires_db

pytestmark = requires_db


def test_pipeline_end_to_end_and_idempotent_retry(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    async def scenario() -> None:
        src = make_wav(tmp_path / "src.wav", seconds=3.0)
        with src.open("rb") as f:
            audio_url = get_audio_storage().save(f, "src.wav")

        async with SessionLocal() as session:
            meeting = Meeting(
                user_id=DEFAULT_USER_ID, title="pipeline-test", audio_url=audio_url
            )
            session.add(meeting)
            await session.commit()
            meeting_id = meeting.id

        try:
            await run_pipeline(meeting_id)

            async with SessionLocal() as session:
                meeting = await session.get(Meeting, meeting_id)
                assert meeting.status == MeetingStatus.done
                assert meeting.error_message is None
                assert meeting.duration and abs(meeting.duration - 3.0) < 0.1

                segs = (
                    await session.scalars(
                        select(TranscriptSegment)
                        .where(TranscriptSegment.meeting_id == meeting_id)
                        .order_by(TranscriptSegment.seq)
                    )
                ).all()
                assert len(segs) > 0
                assert [s.seq for s in segs] == list(range(len(segs)))
                assert all(s.text for s in segs)
                assert all(s.person_id is None for s in segs)  # 暗桩字段留空

                samples = (
                    await session.scalars(
                        select(VoiceSample).where(
                            VoiceSample.source_meeting_id == meeting_id
                        )
                    )
                ).all()
                assert len(samples) == 2  # mock 剧本有两个说话人
                for vs in samples:
                    assert vs.person_id is None  # 暗桩：一期不绑定
                    assert list(vs.embedding)  # embedding 已留存
                    assert vs.model_name == "mock-diarizer"
                    assert vs.model_version

            # 重试幂等：再跑一遍不产生重复数据
            await run_pipeline(meeting_id)
            async with SessionLocal() as session:
                n_segs = len(
                    (
                        await session.scalars(
                            select(TranscriptSegment).where(
                                TranscriptSegment.meeting_id == meeting_id
                            )
                        )
                    ).all()
                )
                n_samples = len(
                    (
                        await session.scalars(
                            select(VoiceSample).where(
                                VoiceSample.source_meeting_id == meeting_id
                            )
                        )
                    ).all()
                )
                assert n_segs == len(segs)
                assert n_samples == 2
        finally:
            async with SessionLocal() as session:
                meeting = await session.get(Meeting, meeting_id)
                if meeting is not None:
                    await session.delete(meeting)  # 级联清理 segments/voice_samples
                    await session.commit()

    asyncio.run(scenario())


def test_pipeline_failure_sets_failed_status(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    async def scenario() -> None:
        # 伪造一个损坏的"音频"文件，转码阶段必然失败
        bad = tmp_path / "audio" / "broken.mp3"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"not-audio")

        async with SessionLocal() as session:
            meeting = Meeting(
                user_id=DEFAULT_USER_ID,
                title="pipeline-fail-test",
                audio_url="local://broken.mp3",
            )
            session.add(meeting)
            await session.commit()
            meeting_id = meeting.id

        try:
            await run_pipeline(meeting_id)
            async with SessionLocal() as session:
                meeting = await session.get(Meeting, meeting_id)
                assert meeting.status == MeetingStatus.failed
                assert meeting.error_message
        finally:
            async with SessionLocal() as session:
                meeting = await session.get(Meeting, meeting_id)
                if meeting is not None:
                    await session.delete(meeting)
                    await session.commit()

    asyncio.run(scenario())


def test_retry_after_summarize_failure_reuses_transcript(
    tmp_path, monkeypatch
) -> None:
    """只有摘要失败时，重试从已入库的转写续跑，不重新调 ASR。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    import app.services.pipeline as pipeline_mod
    import app.services.summarize as summarize_mod
    from app.services.asr import MockASRProvider
    from app.services.llm.base import LLMError, LLMProvider
    from app.services.llm.mock import MockLLMProvider
    from app.services.llm.router import LLMRouter

    asr_calls = {"n": 0}
    real_asr = MockASRProvider()

    class CountingASR(MockASRProvider):
        async def transcribe(self, audio_path, hotwords=None, voiceprint_ids=None):
            asr_calls["n"] += 1
            return await real_asr.transcribe(audio_path, hotwords)

    class DownLLM(LLMProvider):
        name = "down"
        model = "down"

        async def complete(self, system, user, json_mode=True, temperature=0.2):
            raise LLMError("llm outage")

    monkeypatch.setattr(pipeline_mod, "get_asr_provider", lambda: CountingASR())
    monkeypatch.setattr(
        summarize_mod, "build_router", lambda: LLMRouter([DownLLM()])
    )

    async def scenario() -> None:
        from app.services.pipeline import run_pipeline
        from app.services.storage import get_audio_storage

        src = make_wav(tmp_path / "r.wav", seconds=2.0)
        with src.open("rb") as f:
            audio_url = get_audio_storage().save(f, "r.wav")
        async with SessionLocal() as session:
            meeting = Meeting(
                user_id=DEFAULT_USER_ID, title="续跑测试", audio_url=audio_url
            )
            session.add(meeting)
            await session.commit()
            mid = meeting.id

        try:
            # 第一次：转写成功、摘要失败
            await run_pipeline(mid)
            async with SessionLocal() as session:
                meeting = await session.get(Meeting, mid)
                assert meeting.status == MeetingStatus.failed
                n_segs = len(
                    (
                        await session.scalars(
                            select(TranscriptSegment).where(
                                TranscriptSegment.meeting_id == mid
                            )
                        )
                    ).all()
                )
                assert n_segs > 0  # 转写结果已持久化
            assert asr_calls["n"] == 1

            # LLM 恢复后重试：应跳过 ASR，直接续跑摘要
            monkeypatch.setattr(
                summarize_mod,
                "build_router",
                lambda: LLMRouter([MockLLMProvider()]),
            )
            await run_pipeline(mid)
            async with SessionLocal() as session:
                meeting = await session.get(Meeting, mid)
                assert meeting.status == MeetingStatus.done
            assert asr_calls["n"] == 1  # ASR 没有被重新调用
        finally:
            async with SessionLocal() as session:
                meeting = await session.get(Meeting, mid)
                if meeting is not None:
                    await session.delete(meeting)
                    await session.commit()

    asyncio.run(scenario())
