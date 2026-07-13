"""Speaker 绑定：写 SpeakerBinding（追加式、可审计）并物化 person_id 到 segments。

- 绑定为追加事件：新绑定生效时旧绑定记 superseded_by，可回滚、可审计（PRD §5）
- 两条写入路径共用一套机制：人工重命名（confirmed_by=human）与
  云端声纹自动匹配（confirmed_by=auto，带置信度）；人工可随时覆盖自动
"""
import logging
import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import (
    ConfirmedBy,
    Meeting,
    Person,
    SpeakerBinding,
    TranscriptSegment,
)
from app.services.asr.base import ASRSegment

logger = logging.getLogger(__name__)


class UnknownSpeakerLabel(ValueError):
    pass


async def _apply_binding(
    session: AsyncSession,
    meeting: Meeting,
    speaker_label: str,
    person: Person,
    confirmed_by: ConfirmedBy,
    confidence: float | None,
) -> SpeakerBinding:
    """追加绑定 + 物化 person_id；不 commit，由调用方控制事务边界。"""
    previous = await session.scalar(
        select(SpeakerBinding).where(
            SpeakerBinding.meeting_id == meeting.id,
            SpeakerBinding.speaker_label == speaker_label,
            SpeakerBinding.superseded_by.is_(None),
        )
    )
    binding = SpeakerBinding(
        meeting_id=meeting.id,
        speaker_label=speaker_label,
        person_id=person.id,
        confirmed_by=confirmed_by,
        confidence=confidence,
    )
    session.add(binding)
    await session.flush()
    if previous is not None:
        previous.superseded_by = binding.id
    await session.execute(
        update(TranscriptSegment)
        .where(
            TranscriptSegment.meeting_id == meeting.id,
            TranscriptSegment.speaker_label == speaker_label,
        )
        .values(person_id=person.id)
    )
    return binding


async def auto_bind_voiceprints(
    session: AsyncSession, meeting: Meeting, segments: list[ASRSegment]
) -> int:
    """云端声纹命中 → 自动绑定 Person（不 commit，由调用方提交）。

    - 按 voiceprint_id 匹配本地 Person；无匹配则自动建档
      （name 取云端注册名，即命中段的 speaker_label）
    - 置信度低于 voiceprint_auto_bind_min_confidence 的命中忽略
      （score 缺失视为云端已过阈，直接采纳）
    """
    hits: dict[str, tuple[str, float | None]] = {}
    for seg in segments:
        if not seg.voiceprint_id:
            continue
        best = hits.get(seg.speaker_label)
        if best is None or (seg.voiceprint_confidence or 0.0) > (best[1] or 0.0):
            hits[seg.speaker_label] = (seg.voiceprint_id, seg.voiceprint_confidence)

    bound = 0
    for label, (vp_id, score) in hits.items():
        if (
            score is not None
            and score < settings.voiceprint_auto_bind_min_confidence
        ):
            logger.info(
                "voiceprint hit for %r skipped: score %.3f below threshold",
                label, score,
            )
            continue
        person = await session.scalar(
            select(Person).where(
                Person.user_id == meeting.user_id,
                Person.voiceprint_id == vp_id,
            )
        )
        if person is None:
            person = Person(
                user_id=meeting.user_id, name=label, voiceprint_id=vp_id
            )
            session.add(person)
            await session.flush()
        await _apply_binding(
            session, meeting, label, person, ConfirmedBy.auto, score
        )
        bound += 1
        logger.info(
            "voiceprint auto-bound %r -> person %s (score=%s)",
            label, person.name, score,
        )
    return bound


async def bind_speaker(
    session: AsyncSession,
    meeting: Meeting,
    speaker_label: str,
    person_name: str,
) -> tuple[SpeakerBinding, Person]:
    label_count = await session.scalar(
        select(func.count())
        .select_from(TranscriptSegment)
        .where(
            TranscriptSegment.meeting_id == meeting.id,
            TranscriptSegment.speaker_label == speaker_label,
        )
    )
    if not label_count:
        raise UnknownSpeakerLabel(
            f"speaker label {speaker_label!r} not found in meeting transcript"
        )

    person = await session.scalar(
        select(Person).where(
            Person.user_id == meeting.user_id, Person.name == person_name
        )
    )
    if person is None:
        person = Person(user_id=meeting.user_id, name=person_name)
        session.add(person)
        await session.flush()

    # 物化：绑定后 segments.person_id 立即可用（纪要与 TODO 归属用真名）
    binding = await _apply_binding(
        session, meeting, speaker_label, person, ConfirmedBy.human, 1.0
    )
    await session.commit()
    await session.refresh(binding)
    return binding, person


async def active_speaker_names(
    session: AsyncSession, meeting_id: uuid.UUID
) -> dict[str, tuple[uuid.UUID, str]]:
    """当前生效的 speaker_label → (person_id, 真名) 映射。"""
    rows = await session.execute(
        select(SpeakerBinding.speaker_label, Person.id, Person.name)
        .join(Person, Person.id == SpeakerBinding.person_id)
        .where(
            SpeakerBinding.meeting_id == meeting_id,
            SpeakerBinding.superseded_by.is_(None),
        )
    )
    return {label: (pid, name) for label, pid, name in rows}
