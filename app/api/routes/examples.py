"""纪要范例库管理（公司文风第二层，few-shot 学习）。

范例两种来源：
1. POST /summary-examples/from-meeting/{id}：把已生成的纪要渲染成
   公司格式文本采纳为范例（可再手工润色，润色后的版本才是学习目标）
2. POST /summary-examples：直接粘贴既有公司纪要文本

启用中的范例在每次摘要时按更新时间取最新若干条注入 prompt
（条数/字符预算见配置 SUMMARY_EXAMPLES_MAX_*）。
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import require_user
from app.db.session import get_db
from app.models import Meeting, Summary, SummaryExample
from app.schemas.summary_example import (
    MAX_EXAMPLE_CHARS,
    SummaryExampleIn,
    SummaryExampleOut,
    SummaryExampleUpdate,
)
from app.services.summary_text import render_summary_text

router = APIRouter(prefix="/summary-examples", tags=["summary-examples"])


async def _get_example_or_404(
    db: AsyncSession, example_id: UUID, user_id: UUID
) -> SummaryExample:
    example = await db.get(SummaryExample, example_id)
    if example is None or example.user_id != user_id:
        raise HTTPException(status_code=404, detail="example not found")
    return example


@router.get("", response_model=list[SummaryExampleOut])
async def list_examples(
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> list[SummaryExample]:
    rows = await db.scalars(
        select(SummaryExample)
        .where(SummaryExample.user_id == user_id)
        .order_by(SummaryExample.updated_at.desc())
    )
    return list(rows)


@router.post("", response_model=SummaryExampleOut, status_code=201)
async def create_example(
    body: SummaryExampleIn,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> SummaryExample:
    example = SummaryExample(user_id=user_id, title=body.title, content=body.content)
    db.add(example)
    await db.commit()
    await db.refresh(example)
    return example


@router.post(
    "/from-meeting/{meeting_id}",
    response_model=SummaryExampleOut,
    status_code=201,
)
async def create_example_from_meeting(
    meeting_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> SummaryExample:
    """采纳某会议的最新纪要为范例（渲染成公司格式文本入库）。"""
    meeting = await db.get(Meeting, meeting_id)
    if meeting is None or meeting.user_id != user_id:
        raise HTTPException(status_code=404, detail="meeting not found")
    summary = await db.scalar(
        select(Summary)
        .where(Summary.meeting_id == meeting_id)
        .order_by(Summary.version.desc())
        .limit(1)
    )
    if summary is None:
        raise HTTPException(status_code=404, detail="summary not ready")
    content = summary.content_json
    if content.get("_meta", {}).get("degraded"):
        raise HTTPException(
            status_code=409, detail="降级的纯文本纪要不适合作为文风范例"
        )
    text = render_summary_text(content)
    if len(text) > MAX_EXAMPLE_CHARS:
        raise HTTPException(status_code=422, detail="纪要过长，无法作为范例")
    example = SummaryExample(
        user_id=user_id,
        title=content.get("title") or meeting.title,
        content=text,
        source_meeting_id=meeting.id,
    )
    db.add(example)
    await db.commit()
    await db.refresh(example)
    return example


@router.patch("/{example_id}", response_model=SummaryExampleOut)
async def update_example(
    example_id: UUID,
    body: SummaryExampleUpdate,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> SummaryExample:
    example = await _get_example_or_404(db, example_id, user_id)
    if body.title is not None:
        example.title = body.title
    if body.content is not None:
        example.content = body.content
    if body.enabled is not None:
        example.enabled = body.enabled
    await db.commit()
    await db.refresh(example)
    return example


@router.delete("/{example_id}", status_code=204)
async def delete_example(
    example_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> None:
    example = await _get_example_or_404(db, example_id, user_id)
    await db.delete(example)
    await db.commit()
