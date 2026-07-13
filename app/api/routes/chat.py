"""AI 会议问答（RAG，PRD Feature 5）：基于转录内容的对话，回答带原文引用。"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import require_user
from app.db.session import get_db
from app.models import ChatMessage, Meeting, MeetingStatus, TranscriptSegment
from app.schemas.chat import AskIn, ChatMessageOut
from app.services.qa import answer_question, resolve_citations

router = APIRouter(prefix="/meetings/{meeting_id}/chat", tags=["chat"])


async def _get_meeting_or_404(
    db: AsyncSession, meeting_id: UUID, user_id: UUID
) -> Meeting:
    meeting = await db.get(Meeting, meeting_id)
    if meeting is None or meeting.user_id != user_id:
        raise HTTPException(status_code=404, detail="meeting not found")
    return meeting


async def _to_out(
    db: AsyncSession, meeting_id: UUID, msg: ChatMessage
) -> ChatMessageOut:
    citations = (
        await resolve_citations(db, meeting_id, msg.cited_segment_ids)
        if msg.role == "assistant"
        else []
    )
    return ChatMessageOut(
        id=msg.id,
        role=msg.role,
        content=msg.content,
        citations=citations,
        created_at=msg.created_at,
    )


@router.get("", response_model=list[ChatMessageOut])
async def list_chat(
    meeting_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> list[ChatMessageOut]:
    await _get_meeting_or_404(db, meeting_id, user_id)
    msgs = await db.scalars(
        select(ChatMessage)
        .where(ChatMessage.meeting_id == meeting_id)
        .order_by(ChatMessage.created_at, ChatMessage.id)
    )
    return [await _to_out(db, meeting_id, m) for m in msgs]


@router.post("", response_model=ChatMessageOut, status_code=201)
async def ask(
    meeting_id: UUID,
    body: AskIn,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> ChatMessageOut:
    meeting = await _get_meeting_or_404(db, meeting_id, user_id)
    if meeting.status != MeetingStatus.done:
        raise HTTPException(
            status_code=409, detail="meeting is not ready for Q&A (not done)"
        )
    has_segments = await db.scalar(
        select(func.count())
        .select_from(TranscriptSegment)
        .where(TranscriptSegment.meeting_id == meeting_id)
    )
    if not has_segments:
        raise HTTPException(status_code=409, detail="meeting has no transcript")
    result = await answer_question(db, meeting, body.question.strip())
    return await _to_out(db, meeting_id, result.assistant)
