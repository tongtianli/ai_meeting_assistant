"""Speaker 绑定：写 SpeakerBinding（追加式、可审计）并物化 person_id 到 segments。

- 绑定为追加事件：改绑时旧绑定记 superseded_by，撤销时记 revoked_at
  （审计行保留），可回滚、可审计（PRD §5 / 声纹设计 §4.1）
- 两条写入路径共用一套机制：人工重命名（confirmed_by=human）与
  声纹自动匹配（confirmed_by=auto，带置信度）；人工可随时覆盖自动
- human 绑定＝隐式声纹登记：本会议该 speaker 的 VoiceSample 精确物化归属
  （person_id + assigned_by_binding_id）；auto 绑定不物化——错误的自动识别
  不得进入长期参考样本自我放大（声纹设计 §4.1 防漂移）
"""
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import (
    ConfirmedBy,
    Meeting,
    Person,
    SpeakerBinding,
    SpeakerIdentityDismissal,
    TranscriptSegment,
    VoiceSample,
)
from app.services.asr.base import ASRSegment

logger = logging.getLogger(__name__)


class UnknownSpeakerLabel(ValueError):
    pass


def active_binding_clause():
    """生效绑定条件：未被改绑取代且未被撤销。"""
    return (
        SpeakerBinding.superseded_by.is_(None),
        SpeakerBinding.revoked_at.is_(None),
    )


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
            *active_binding_clause(),
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
    if confirmed_by is ConfirmedBy.human:
        # 隐式声纹登记（§4.1）：只物化当前会议、当前 label 的样本；
        # 改绑时同一 UPDATE 精确转给新 Person，不迁移其名下其他样本
        await session.execute(
            update(VoiceSample)
            .where(
                VoiceSample.source_meeting_id == meeting.id,
                VoiceSample.speaker_label == speaker_label,
            )
            .values(person_id=person.id, assigned_by_binding_id=binding.id)
        )
        # 手动绑定即身份已明确：清掉本会议该 speaker 的「跳过」记录，
        # 之后撤销绑定时确认卡自然重新出现（§5）
        await session.execute(
            delete(SpeakerIdentityDismissal).where(
                SpeakerIdentityDismissal.meeting_id == meeting.id,
                SpeakerIdentityDismissal.speaker_label == speaker_label,
            )
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


async def unbind_speaker(
    session: AsyncSession, meeting: Meeting, speaker_label: str
) -> SpeakerBinding:
    """撤销当前生效绑定（声纹设计 §4.1）：审计行保留（记 revoked_at），
    segments 回到未识别，本绑定物化的样本回无主池（不物理删除）。"""
    binding = await session.scalar(
        select(SpeakerBinding).where(
            SpeakerBinding.meeting_id == meeting.id,
            SpeakerBinding.speaker_label == speaker_label,
            *active_binding_clause(),
        )
    )
    if binding is None:
        raise UnknownSpeakerLabel(
            f"no active binding for speaker label {speaker_label!r}"
        )
    binding.revoked_at = datetime.now(timezone.utc)
    await session.execute(
        update(TranscriptSegment)
        .where(
            TranscriptSegment.meeting_id == meeting.id,
            TranscriptSegment.speaker_label == speaker_label,
        )
        .values(person_id=None)
    )
    # 只释放本绑定物化的样本：auto 绑定从未物化，改绑后旧绑定的样本
    # 已转给新绑定——都不会被这里误伤
    await session.execute(
        update(VoiceSample)
        .where(VoiceSample.assigned_by_binding_id == binding.id)
        .values(person_id=None, assigned_by_binding_id=None)
    )
    await session.commit()
    await session.refresh(binding)
    return binding


async def active_speaker_names(
    session: AsyncSession, meeting_id: uuid.UUID
) -> dict[str, tuple[uuid.UUID, str]]:
    """当前生效的 speaker_label → (person_id, 真名) 映射。"""
    rows = await session.execute(
        select(SpeakerBinding.speaker_label, Person.id, Person.name)
        .join(Person, Person.id == SpeakerBinding.person_id)
        .where(
            SpeakerBinding.meeting_id == meeting_id,
            *active_binding_clause(),
        )
    )
    return {label: (pid, name) for label, pid, name in rows}
