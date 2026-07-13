"""异步处理管道：转码归一化 → 转写+分离+对齐 → segments/embedding 入库。

- 每阶段独立，状态实时写回 Meeting（uploaded → transcoding → transcribing → done/failed）
- 任一阶段失败置 failed + error_message；重试从头执行但各阶段幂等
  （已转码的中间产物直接复用，segments/voice_samples 先清后写）
- MVP 由 FastAPI BackgroundTasks 驱动；二期换 Celery 时阶段划分不变（PRD §4）
"""
import logging
import wave
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import SessionLocal
from app.models import Meeting, MeetingStatus, Person, TranscriptSegment, VoiceSample
from app.services.asr import get_asr_provider
from app.services.speakers import auto_bind_voiceprints
from app.services.summarize import summarize_meeting
from app.services.storage import get_audio_storage
from app.services.transcode import transcode_to_wav16k_mono

logger = logging.getLogger(__name__)


async def run_pipeline(meeting_id: UUID) -> None:
    async with SessionLocal() as session:
        meeting = await session.get(Meeting, meeting_id)
        if meeting is None:
            logger.error("pipeline: meeting %s not found", meeting_id)
            return
        meeting.error_message = None  # 重试时清掉上次失败的残留文案
        try:
            wav_path = await _stage_transcode(session, meeting)
            await _stage_transcribe(session, meeting, wav_path)
            await _stage_summarize(session, meeting)
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
    # 阶段级重试（PRD §2）：转写结果已入库则直接跳过，不重跑 ASR——
    # segments 与 voice_samples 在同一事务提交，存在即代表该阶段完整成功；
    # 已有的 SpeakerBinding 与物化的 person_id 也因此得以保留
    existing = await session.scalar(
        select(func.count())
        .select_from(TranscriptSegment)
        .where(TranscriptSegment.meeting_id == meeting.id)
    )
    if existing:
        logger.info(
            "transcribe stage skipped for %s: %d segments already persisted",
            meeting.id,
            existing,
        )
        return
    # 已登记云端声纹的 Person → 随任务下发做声纹匹配（支持的 provider 生效）
    vp_ids = list(
        await session.scalars(
            select(Person.voiceprint_id).where(
                Person.user_id == meeting.user_id,
                Person.voiceprint_id.is_not(None),
            )
        )
    )
    result = await get_asr_provider().transcribe(
        wav_path, voiceprint_ids=vp_ids or None
    )
    if not result.segments:
        raise RuntimeError(
            "ASR produced no segments; check provider logs before retrying"
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
    # 声纹命中 → 自动绑定（与 segments 同事务：要么全成，要么重试时整体重来）
    await session.flush()
    bound = await auto_bind_voiceprints(session, meeting, result.segments)
    if bound:
        logger.info("voiceprint auto-binding: %d speakers bound", bound)
    await session.commit()


async def _stage_summarize(session: AsyncSession, meeting: Meeting) -> None:
    await _set_status(session, meeting, MeetingStatus.summarizing)
    await summarize_meeting(session, meeting)
