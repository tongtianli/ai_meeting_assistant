"""Speaker 重命名：写 SpeakerBinding（追加式、可审计）并物化 person_id 到 segments。

- 绑定为追加事件：新绑定生效时旧绑定记 superseded_by，可回滚、可审计（PRD §5）
- MVP 仅人工确认（confirmed_by=human）；二期声纹自动匹配走同一条写入路径
"""
import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    ConfirmedBy,
    Meeting,
    Person,
    SpeakerBinding,
    TranscriptSegment,
)


class UnknownSpeakerLabel(ValueError):
    pass


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
        confirmed_by=ConfirmedBy.human,
        confidence=1.0,
    )
    session.add(binding)
    await session.flush()
    if previous is not None:
        previous.superseded_by = binding.id

    # 物化：绑定后 segments.person_id 立即可用（纪要与 TODO 归属用真名）
    await session.execute(
        update(TranscriptSegment)
        .where(
            TranscriptSegment.meeting_id == meeting.id,
            TranscriptSegment.speaker_label == speaker_label,
        )
        .values(person_id=person.id)
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
