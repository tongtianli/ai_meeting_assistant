"""Person 管理：声纹 ID 登记与人员档案。

声纹注册流程（MVP，云端注册 API 打通前）：
1. 火山控制台「声纹管理」上传 ≥10s 纯净人声样本（16kHz/16bit 单声道 wav），
   命名后得到声纹 ID
2. POST /persons 登记（name + voiceprint_id + 本人同意记录）
3. 之后的转写自动携带 voice_print_list，命中即自动绑定说话人

PIPL：声纹为敏感个人信息。删除 Person 时本地关联立即级联；
云端样本在注册 API 打通前需人工在控制台同步删除（响应中明确提示）。
"""
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import require_user
from app.db.session import get_db
from app.models import Person
from app.schemas.person import PersonDeleteOut, PersonIn, PersonOut, PersonUpdate

router = APIRouter(prefix="/persons", tags=["persons"])


def _consent(note: str | None) -> dict | None:
    if not note:
        return None
    return {
        "note": note,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


async def _get_person_or_404(
    db: AsyncSession, person_id: UUID, user_id: UUID
) -> Person:
    person = await db.get(Person, person_id)
    if person is None or person.user_id != user_id:
        raise HTTPException(status_code=404, detail="person not found")
    return person


@router.get("", response_model=list[PersonOut])
async def list_persons(
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> list[Person]:
    rows = await db.scalars(
        select(Person).where(Person.user_id == user_id).order_by(Person.name)
    )
    return list(rows)


@router.post("", response_model=PersonOut, status_code=201)
async def create_person(
    body: PersonIn,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> Person:
    if body.voiceprint_id and not body.consent_note:
        # PRD §9.4：声纹登记必须留存本人同意记录
        raise HTTPException(
            status_code=422,
            detail="consent_note is required when registering a voiceprint_id",
        )
    person = Person(
        user_id=user_id,
        name=body.name,
        voiceprint_id=body.voiceprint_id,
        consent_record=_consent(body.consent_note),
    )
    db.add(person)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail="voiceprint_id already registered to another person",
        )
    await db.refresh(person)
    return person


@router.patch("/{person_id}", response_model=PersonOut)
async def update_person(
    person_id: UUID,
    body: PersonUpdate,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> Person:
    person = await _get_person_or_404(db, person_id, user_id)
    if body.voiceprint_id and not (body.consent_note or person.consent_record):
        raise HTTPException(
            status_code=422,
            detail="consent_note is required when registering a voiceprint_id",
        )
    if body.name is not None:
        person.name = body.name
    if body.voiceprint_id is not None:
        person.voiceprint_id = body.voiceprint_id or None
    if body.consent_note:
        person.consent_record = _consent(body.consent_note)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail="voiceprint_id already registered to another person",
        )
    await db.refresh(person)
    return person


@router.delete("/{person_id}", response_model=PersonDeleteOut)
async def delete_person(
    person_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(require_user),
) -> PersonDeleteOut:
    person = await _get_person_or_404(db, person_id, user_id)
    had_voiceprint = bool(person.voiceprint_id)
    vp_id = person.voiceprint_id
    # 本地级联：speaker_bindings/voice_samples CASCADE，
    # transcript_segments.person_id 置空由 FK 规则处理
    await db.delete(person)
    await db.commit()
    message = "person deleted"
    if had_voiceprint:
        message = (
            f"person deleted; PIPL 提醒：请在火山控制台「声纹管理」中"
            f"同步删除云端声纹样本 {vp_id}（注册 API 打通后将自动级联）"
        )
    return PersonDeleteOut(
        deleted=True,
        cloud_cleanup_required=had_voiceprint,
        message=message,
    )
