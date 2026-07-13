"""全局术语表管理（跨会议热词，PRD Feature 1）。

启用中的术语在每次转写时作为 hotwords 注入 ASR（provider 无关，见
app/services/pipeline.py::_stage_transcribe）；数量上限 GLOSSARY_MAX_TERMS。
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import require_user
from app.db.session import get_db
from app.models import GlossaryTerm
from app.schemas.glossary_term import (
    GlossaryTermOut,
    GlossaryTermsBulkIn,
    GlossaryTermUpdate,
)

router = APIRouter(prefix="/glossary", tags=["glossary"])

_MAX_TERM_LEN = 255


async def _get_term_or_404(
    db: AsyncSession, term_id: UUID, user_id: UUID
) -> GlossaryTerm:
    term = await db.get(GlossaryTerm, term_id)
    if term is None or term.user_id != user_id:
        raise HTTPException(status_code=404, detail="term not found")
    return term


@router.get("", response_model=list[GlossaryTermOut])
async def list_terms(
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> list[GlossaryTerm]:
    rows = await db.scalars(
        select(GlossaryTerm)
        .where(GlossaryTerm.user_id == user_id)
        .order_by(GlossaryTerm.updated_at.desc())
    )
    return list(rows)


@router.post("", response_model=list[GlossaryTermOut], status_code=201)
async def add_terms(
    body: GlossaryTermsBulkIn,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> list[GlossaryTerm]:
    """批量新增；去空白、批内去重、跳过已存在的术语，返回本次实际新增项。"""
    # 规范化：strip、截长、去空、批内去重（保序）
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in body.terms:
        t = raw.strip()[:_MAX_TERM_LEN].strip()
        if t and t not in seen:
            seen.add(t)
            normalized.append(t)
    if not normalized:
        raise HTTPException(status_code=422, detail="no valid terms after trimming")

    existing = set(
        await db.scalars(
            select(GlossaryTerm.term).where(
                GlossaryTerm.user_id == user_id, GlossaryTerm.term.in_(normalized)
            )
        )
    )
    created = [
        GlossaryTerm(user_id=user_id, term=t)
        for t in normalized
        if t not in existing
    ]
    db.add_all(created)
    await db.commit()
    for term in created:
        await db.refresh(term)
    return created


@router.patch("/{term_id}", response_model=GlossaryTermOut)
async def update_term(
    term_id: UUID,
    body: GlossaryTermUpdate,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> GlossaryTerm:
    term = await _get_term_or_404(db, term_id, user_id)
    if body.enabled is not None:
        term.enabled = body.enabled
    await db.commit()
    await db.refresh(term)
    return term


@router.delete("/{term_id}", status_code=204)
async def delete_term(
    term_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> None:
    term = await _get_term_or_404(db, term_id, user_id)
    await db.delete(term)
    await db.commit()
