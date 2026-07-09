"""异步处理管道：转码归一化 → 转写+分离+对齐 → segments/embedding 入库。

- 每阶段独立，状态实时写回 Meeting（uploaded → transcoding → transcribing → done/failed）
- 任一阶段失败置 failed + error_message；重试从头执行但各阶段幂等
  （已转码的中间产物直接复用，segments/voice_samples 先清后写）
- MVP 由 FastAPI BackgroundTasks 驱动；二期换 Celery 时阶段划分不变（PRD §4）
- LLM 总结阶段（任务 4）接入后插到 transcribe 之后：summarizing → done
"""
import logging
import wave
from pathlib import Path
from uuid import UUID

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import SessionLocal
from app.models import Meeting, MeetingStatus, TranscriptSegment, VoiceSample
from app.services.asr import get_asr_provider
from app.services.storage import get_audio_storage
from app.services.transcode import transcode_to_wav16k_mono

logger = logging.getLogger(__name__)


async def run_pipeline(meeting_id: UUID) -> None:
    async with SessionLocal() as session:
        meeting = await session.get(Meeting, meeting_id)
        if meeting is None:
            logger.error("pipeline: meeting %s not found", meeting_id)
            return
        try:
            wav_path = await _stage_transcode(session, meeting)
            await _stage_transcribe(session, meeting, wav_path)
        except Exception as exc:
            logger.exception("pipeline failed for meeting %s", meeting_id)
            await session.rollback()
            meeting.status = MeetingStatus.failed
            meeting.error_message = f"{type(exc).__name__}: {exc}"[:2000]
            await session.commit()
            return
        meeting.status = MeetingStatus.done
        meeting.error_message = None
        await session.commit()


async def _set_status(
    session: AsyncSession, meeting: Meeting, status: MeetingStatus
) -> None:
    meeting.status = status
    await session.commit()


async def _stage_transcode(session: AsyncSession, meeting: Meeting) -> Path:
    await _set_status(session, meeting, MeetingStatus.transcoding)
    src = get_audio_storage().resolve(meeting.audio_url)
    dest_dir = settings.data_dir / "transcoded"
    dest = dest_dir / f"{src.stem}.wav"
    if not dest.exists():  # 幂等：重试时复用已完成的转码产物
        dest = await transcode_to_wav16k_mono(src, dest_dir)
    with wave.open(str(dest), "rb") as w:
        meeting.duration = round(w.getnframes() / w.getframerate(), 3)
    await session.commit()
    return dest


async def _stage_transcribe(
    session: AsyncSession, meeting: Meeting, wav_path: Path
) -> None:
    await _set_status(session, meeting, MeetingStatus.transcribing)
    result = await get_asr_provider().transcribe(wav_path)
    # 幂等：重试时清掉本会议旧的派生数据再写入
    await session.execute(
        delete(TranscriptSegment).where(TranscriptSegment.meeting_id == meeting.id)
    )
    await session.execute(
        delete(VoiceSample).where(VoiceSample.source_meeting_id == meeting.id)
    )
    session.add_all(
        TranscriptSegment(
            meeting_id=meeting.id,
            seq=i,
            start_time=seg.start_time,
            end_time=seg.end_time,
            speaker_label=seg.speaker_label,
            text=seg.text,
        )
        for i, seg in enumerate(result.segments)
    )
    session.add_all(
        VoiceSample(
            source_meeting_id=meeting.id,  # person_id 留空：二期声纹绑定后物化（暗桩）
            embedding=e.embedding,
            model_name=e.model_name,
            model_version=e.model_version,
            sample_start=e.sample_start,
            sample_end=e.sample_end,
        )
        for e in result.speaker_embeddings
    )
    await session.commit()
