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
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models import DEFAULT_USER_ID, Meeting, MeetingStatus
from app.schemas.meeting import MeetingOut
from app.services.pipeline import run_pipeline
from app.services.storage import get_audio_storage

router = APIRouter(prefix="/meetings", tags=["meetings"])

ALLOWED_SUFFIXES = {".mp3", ".wav", ".m4a", ".mp4"}  # PRD §2 支持格式


@router.post("", response_model=MeetingOut, status_code=201)
async def create_meeting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    db: AsyncSession = Depends(get_db),
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
        user_id=DEFAULT_USER_ID,
        title=title,
        status=MeetingStatus.uploaded,
        audio_url=audio_url,
    )
    db.add(meeting)
    await db.commit()
    await db.refresh(meeting)
    background_tasks.add_task(run_pipeline, meeting.id)
    return meeting


@router.get("/{meeting_id}", response_model=MeetingOut)
async def get_meeting(
    meeting_id: UUID, db: AsyncSession = Depends(get_db)
) -> Meeting:
    meeting = await db.get(Meeting, meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="meeting not found")
    return meeting


@router.post("/{meeting_id}/retry", response_model=MeetingOut)
async def retry_meeting(
    meeting_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> Meeting:
    """失败后重跑管道；各阶段幂等，已完成的转码产物会被复用。"""
    meeting = await db.get(Meeting, meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="meeting not found")
    if meeting.status != MeetingStatus.failed:
        raise HTTPException(
            status_code=409,
            detail=f"meeting is {meeting.status.value}, only failed meetings can be retried",
        )
    background_tasks.add_task(run_pipeline, meeting.id)
    return meeting
