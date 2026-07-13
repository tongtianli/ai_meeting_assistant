import logging
from pathlib import Path
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import PlainTextResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import create_audio_token, require_user
from app.db.session import get_db
from app.models import ActionItem, Meeting, MeetingStatus, Person, Summary, TranscriptSegment
from app.schemas.summary import SummaryOut
from app.schemas.meeting import (
    AudioUrlOut,
    MeetingOut,
    SegmentOut,
    SpeakerBindingOut,
    SpeakerRenameIn,
    TranscriptOut,
)
from app.services.pipeline import run_pipeline
from app.services.speakers import (
    UnknownSpeakerLabel,
    active_speaker_names,
    bind_speaker,
)
from app.services.storage import get_audio_storage
from app.services.transcode import delete_transcoded_artifacts
from app.services.word_export import build_context, render_summary_docx

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/meetings", tags=["meetings"])

ALLOWED_SUFFIXES = {".mp3", ".wav", ".m4a", ".mp4"}  # PRD §2 支持格式


async def _get_meeting_or_404(
    db: AsyncSession, meeting_id: UUID, user_id: UUID
) -> Meeting:
    meeting = await db.get(Meeting, meeting_id)
    if meeting is None or meeting.user_id != user_id:
        raise HTTPException(status_code=404, detail="meeting not found")
    return meeting


@router.post("", response_model=MeetingOut, status_code=201)
async def create_meeting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    location: str | None = Form(None),
    host: str | None = Form(None),
    recorder: str | None = Form(None),
    importance: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> Meeting:
    """上传录音并触发处理管道。

    MVP 为服务端直传本地存储；预签名 URL 分片直传对象存储（PRD §9.2）
    在对象存储接入时替换此入口，管道不变。
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported audio format {suffix!r}, allowed: {sorted(ALLOWED_SUFFIXES)}",
        )
    audio_url = get_audio_storage().save(file.file, file.filename or f"audio{suffix}")
    meeting = Meeting(
        user_id=user_id,
        title=title,
        status=MeetingStatus.uploaded,
        audio_url=audio_url,
        location=location or None,
        host=host or None,
        recorder=recorder or None,
        importance=importance or None,
    )
    db.add(meeting)
    await db.commit()
    await db.refresh(meeting)
    background_tasks.add_task(run_pipeline, meeting.id)
    return meeting


@router.get("", response_model=list[MeetingOut])
async def list_meetings(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> list[Meeting]:
    result = await db.scalars(
        select(Meeting)
        .where(Meeting.user_id == user_id)
        .order_by(Meeting.created_at.desc())
        .limit(min(limit, 200))
        .offset(offset)
    )
    return list(result)


@router.get("/{meeting_id}", response_model=MeetingOut)
async def get_meeting(
    meeting_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> Meeting:
    return await _get_meeting_or_404(db, meeting_id, user_id)


@router.post("/{meeting_id}/retry", response_model=MeetingOut)
async def retry_meeting(
    meeting_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> Meeting:
    """失败后重跑管道；各阶段幂等，已完成的转码产物会被复用。"""
    meeting = await _get_meeting_or_404(db, meeting_id, user_id)
    if meeting.status != MeetingStatus.failed:
        raise HTTPException(
            status_code=409,
            detail=f"meeting is {meeting.status.value}, only failed meetings can be retried",
        )
    background_tasks.add_task(run_pipeline, meeting.id)
    return meeting


@router.delete("/{meeting_id}", status_code=204)
async def delete_meeting(
    meeting_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> Response:
    """删除会议（PRD §9.4：删除仅由用户显式触发）。

    DB 侧靠外键级联清理 segments/bindings/summaries/action_items/
    chat_messages/voice_samples（summary_examples.source_meeting_id 置空，
    范例保留）；随后 best-effort 清理磁盘上的音频与转码中间产物。
    """
    meeting = await _get_meeting_or_404(db, meeting_id, user_id)
    audio_url = meeting.audio_url

    await db.delete(meeting)
    await db.commit()

    # DB 记录已是权威状态；文件清理失败只告警，不让删除请求整体报错
    if audio_url:
        try:
            storage = get_audio_storage()
            src = storage.resolve(audio_url)
            storage.delete(audio_url)
            delete_transcoded_artifacts(src, settings.data_dir / "transcoded")
        except Exception:
            logger.warning(
                "meeting %s deleted, but on-disk cleanup failed for %s",
                meeting_id,
                audio_url,
                exc_info=True,
            )

    return Response(status_code=204)


async def _load_segments(
    db: AsyncSession, meeting_id: UUID
) -> list[TranscriptSegment]:
    result = await db.scalars(
        select(TranscriptSegment)
        .where(TranscriptSegment.meeting_id == meeting_id)
        .order_by(TranscriptSegment.seq)
    )
    return list(result)


@router.get("/{meeting_id}/segments", response_model=TranscriptOut)
async def get_transcript(
    meeting_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> TranscriptOut:
    """完整转录；segments 入库即可用，不依赖后续阶段（PRD §2）。"""
    meeting = await _get_meeting_or_404(db, meeting_id, user_id)
    segments = await _load_segments(db, meeting_id)
    names = await active_speaker_names(db, meeting_id)
    return TranscriptOut(
        meeting_id=meeting.id,
        segments=[
            SegmentOut(
                seq=s.seq,
                start_time=s.start_time,
                end_time=s.end_time,
                speaker_label=s.speaker_label,
                speaker_name=names.get(s.speaker_label, (None, s.speaker_label))[1],
                person_id=s.person_id,
                text=s.text,
            )
            for s in segments
        ],
    )


def _fmt_ts(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


@router.get("/{meeting_id}/transcript")
async def download_transcript(
    meeting_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> PlainTextResponse:
    """原文下载（纯文本，带时间戳与说话人）。"""
    meeting = await _get_meeting_or_404(db, meeting_id, user_id)
    segments = await _load_segments(db, meeting_id)
    names = await active_speaker_names(db, meeting_id)
    lines = [
        f"[{_fmt_ts(s.start_time)}] "
        f"{names.get(s.speaker_label, (None, s.speaker_label))[1]}: {s.text}"
        for s in segments
    ]
    filename = f"transcript-{meeting.id}.txt"
    return PlainTextResponse(
        "\n".join(lines),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{meeting_id}/audio-url", response_model=AudioUrlOut)
async def get_audio_url(
    meeting_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> AudioUrlOut:
    """按需换发短时签名播放 URL（PRD §9.2）；<audio> 标签无法带请求头。"""
    meeting = await _get_meeting_or_404(db, meeting_id, user_id)
    if not meeting.audio_url:
        raise HTTPException(status_code=409, detail="meeting has no audio")
    token, ttl = create_audio_token(meeting.id)
    return AudioUrlOut(url=f"/api/audio/{meeting.id}?token={token}", expires_in=ttl)


@router.post(
    "/{meeting_id}/speaker-bindings",
    response_model=SpeakerBindingOut,
    status_code=201,
)
async def rename_speaker(
    meeting_id: UUID,
    body: SpeakerRenameIn,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> SpeakerBindingOut:
    """Speaker 手动重命名（MVP）：追加式绑定 + person_id 物化。"""
    meeting = await _get_meeting_or_404(db, meeting_id, user_id)
    try:
        binding, person = await bind_speaker(
            db, meeting, body.speaker_label, body.name.strip()
        )
    except UnknownSpeakerLabel as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SpeakerBindingOut(
        id=binding.id,
        speaker_label=binding.speaker_label,
        person_id=person.id,
        person_name=person.name,
        confirmed_by=binding.confirmed_by,
        confidence=binding.confidence,
    )


@router.get("/{meeting_id}/summary", response_model=SummaryOut)
async def get_summary(
    meeting_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> Summary:
    """最新版本的结构化纪要（Summary 版本化，PRD Feature 6）。"""
    await _get_meeting_or_404(db, meeting_id, user_id)
    summary = await db.scalar(
        select(Summary)
        .where(Summary.meeting_id == meeting_id)
        .order_by(Summary.version.desc())
        .limit(1)
    )
    if summary is None:
        raise HTTPException(status_code=404, detail="summary not ready")
    return summary


_DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)


@router.get("/{meeting_id}/export.docx")
async def export_word(
    meeting_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> Response:
    """Word 导出：最新版纪要实时渲染（PRD Feature 6，纯程序步骤）。"""
    meeting = await _get_meeting_or_404(db, meeting_id, user_id)
    summary = await db.scalar(
        select(Summary)
        .where(Summary.meeting_id == meeting_id)
        .order_by(Summary.version.desc())
        .limit(1)
    )
    if summary is None:
        raise HTTPException(status_code=404, detail="summary not ready")
    rows = await db.execute(
        select(ActionItem, TranscriptSegment.start_time, Person.name)
        .outerjoin(
            TranscriptSegment, TranscriptSegment.id == ActionItem.source_segment_id
        )
        .outerjoin(Person, Person.id == TranscriptSegment.person_id)
        .where(ActionItem.meeting_id == meeting_id)
    )
    content = build_context(summary.content_json, list(rows), meeting)
    payload = render_summary_docx(content)
    filename = f"minutes-{meeting_id}-v{summary.version}.docx"
    return Response(
        payload,
        media_type=_DOCX_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
