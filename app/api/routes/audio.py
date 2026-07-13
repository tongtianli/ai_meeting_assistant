"""音频播放：短时签名 URL 校验后回源文件，支持 range 请求（PRD §9.2）。

音频为不可变资产，仅提供读取；换对象存储后此路由改为 302 到预签名 URL。
"""
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import verify_audio_token, verify_file_token
from app.db.session import get_db
from app.models import Meeting
from app.services.storage import get_audio_storage

router = APIRouter(prefix="/audio", tags=["audio"])

_MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".mp4": "video/mp4",
}


@router.get("/file/{token}")
async def stream_signed_file(token: str) -> FileResponse:
    """凭签名 token 回源 data_dir 下的派生文件（云端 ASR 经公网拉取音频）。"""
    rel = verify_file_token(token)
    if rel is None:
        raise HTTPException(status_code=403, detail="invalid or expired file token")
    base = settings.data_dir.resolve()
    path = (base / rel).resolve()
    # token 由我们自己签发，此处防御路径穿越纯属纵深
    if not path.is_relative_to(base) or not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(
        path, media_type=_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    )


@router.get("/{meeting_id}")
async def stream_audio(
    meeting_id: UUID, token: str, db: AsyncSession = Depends(get_db)
) -> FileResponse:
    if not verify_audio_token(token, meeting_id):
        raise HTTPException(status_code=403, detail="invalid or expired audio token")
    meeting = await db.get(Meeting, meeting_id)
    if meeting is None or not meeting.audio_url:
        raise HTTPException(status_code=404, detail="audio not found")
    path = get_audio_storage().resolve(meeting.audio_url)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="audio file missing")
    suffix = Path(path).suffix.lower()
    return FileResponse(
        path, media_type=_MEDIA_TYPES.get(suffix, "application/octet-stream")
    )
