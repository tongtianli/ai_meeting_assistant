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
from app.models import (
    GlossaryTerm,
    Meeting,
    MeetingStatus,
    Person,
    TranscriptSegment,
    VoiceSample,
)
from app.services.asr import get_asr_provider
from app.services.llm.usage import usage_context
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
            # 管道内所有 LLM/embedding 调用的用量审计都归属本会议
            with usage_context(meeting_id=meeting.id, user_id=meeting.user_id):
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


async def run_resummarize(meeting_id: UUID) -> None:
    """仅重跑摘要阶段（已完成会议重新生成纪要，追加新 Summary 版本）。

    失败语义与 run_pipeline 不同：会议已有可用转录与旧版纪要，失败时
    回到 done 而非 failed，避免旧纪要在界面上消失；错误存 error_message。
    """
    async with SessionLocal() as session:
        meeting = await session.get(Meeting, meeting_id)
        if meeting is None:
            logger.error("resummarize: meeting %s not found", meeting_id)
            return
        try:
            with usage_context(meeting_id=meeting.id, user_id=meeting.user_id):
                await _stage_summarize(session, meeting, origin="resummarize")
        except Exception as exc:
            logger.exception("resummarize failed for meeting %s", meeting_id)
            await session.rollback()
            meeting.error_message = f"{type(exc).__name__}: {exc}"[:2000]
        else:
            meeting.error_message = None
        meeting.status = MeetingStatus.done
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
    # 全局术语表 → 作为 hotwords 注入（provider 无关，PRD Feature 1）；
    # FunASR 不做数量上限，在此统一按 glossary_max_terms 截断
    hotwords = list(
        await session.scalars(
            select(GlossaryTerm.term)
            .where(
                GlossaryTerm.user_id == meeting.user_id,
                GlossaryTerm.enabled.is_(True),
            )
            .order_by(GlossaryTerm.updated_at.desc())
            .limit(settings.glossary_max_terms)
        )
    )
    result = await get_asr_provider().transcribe(
        wav_path, voiceprint_ids=vp_ids or None, hotwords=hotwords or None
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


async def _stage_summarize(
    session: AsyncSession, meeting: Meeting, origin: str = "pipeline"
) -> None:
    await _set_status(session, meeting, MeetingStatus.summarizing)
    await summarize_meeting(session, meeting, origin=origin)
