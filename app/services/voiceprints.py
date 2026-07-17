"""本地声纹记忆后端（TECH_DESIGN_VOICEPRINT_MEMORY_V1 §3/§4.2，Phase 1）。

- 匹配：本会议每个未绑 speaker 的 embedding，对同 model_name/model_version
  的长期参考样本做 pgvector cosine 比对，按 Person 聚合产出统一结构
  `VoiceprintMatchResult`（Phase 3 云端后端复用同一结构收口）；
- 长期参考样本（§4.1）：person_id 非空且 assigned_by_binding_id 指向
  confirmed_by=human 的生效绑定——auto 绑定的样本不参与匹配（防漂移）；
- 自动绑定六重门槛（§4.2，决议 5/7）：默认关闭（阈值未经真实数据校准），
  Phase 1 匹配结果仅落日志与内部接口，供确认卡（Phase 2）与校准使用；
- pending 状态动态计算（§5，决议 4）：绑定/dismissal 皆无 → 未知。
"""
import logging
import uuid
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
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
from app.services.speakers import (
    UnknownSpeakerLabel,
    _apply_binding,
    active_binding_clause,
)

logger = logging.getLogger(__name__)

_BACKEND_LOCAL = "local"
# 单次比对拉取的参考样本行数上限：候选按余弦距离升序，Person 数量级很小，
# 50 行足以覆盖 top 候选；参考样本总数门槛另走精确 count
_REFERENCE_FETCH_LIMIT = 50


@dataclass
class VoiceprintMatchResult:
    """跨后端统一的匹配结果（§3，决议 2）；Phase 3 云端后端产同构结果。"""

    backend: str
    speaker_label: str
    person_id: uuid.UUID
    person_name: str
    score: float  # 1 - cosine_distance，越大越像
    model_key: str  # f"{model_name}:{model_version}"，仅同桶可比
    evidence_sample_ids: list[uuid.UUID] = field(default_factory=list)
    reference_sample_count: int = 0  # 该 Person 同桶长期参考样本总数
    query_sample_seconds: float = 0.0  # 本会议该 speaker 最长样本时长


def _model_key(name: str, version: str) -> str:
    return f"{name}:{version}"


def _reference_sample_filter():
    """长期参考样本条件（§4.1）：human 生效绑定物化的样本。"""
    return (
        VoiceSample.person_id.is_not(None),
        SpeakerBinding.id == VoiceSample.assigned_by_binding_id,
        SpeakerBinding.confirmed_by == ConfirmedBy.human,
        *active_binding_clause(),
    )


async def match_local_voiceprints(
    session: AsyncSession, meeting: Meeting
) -> dict[str, list[VoiceprintMatchResult]]:
    """未绑 speaker → 按 score 降序的候选 Person 列表（可为空列表）。

    只读，不产生绑定；自动绑定决策在 identify_speakers 收口。
    """
    samples = (
        await session.scalars(
            select(VoiceSample).where(
                VoiceSample.source_meeting_id == meeting.id,
                VoiceSample.speaker_label.is_not(None),
            )
        )
    ).all()
    if not samples:
        return {}

    bound_labels = set(
        await session.scalars(
            select(SpeakerBinding.speaker_label).where(
                SpeakerBinding.meeting_id == meeting.id,
                *active_binding_clause(),
            )
        )
    )

    # 每个未绑 speaker 取时长最长的样本作查询向量（质量优先，§8 样本质量风险）
    query_samples: dict[str, VoiceSample] = {}
    for s in samples:
        if s.speaker_label in bound_labels:
            continue
        best = query_samples.get(s.speaker_label)
        if best is None or s.duration > best.duration:
            query_samples[s.speaker_label] = s

    results: dict[str, list[VoiceprintMatchResult]] = {}
    for label, query in sorted(query_samples.items()):
        results[label] = await _match_one(session, meeting, label, query)
    return results


async def _match_one(
    session: AsyncSession, meeting: Meeting, label: str, query: VoiceSample
) -> list[VoiceprintMatchResult]:
    distance = VoiceSample.embedding.cosine_distance(query.embedding)
    rows = (
        await session.execute(
            select(VoiceSample.id, VoiceSample.person_id, Person.name, distance)
            .join(Person, Person.id == VoiceSample.person_id)
            .where(
                *_reference_sample_filter(),
                # 同模型分桶（§4.2 门槛 6）：跨模型 embedding 不可比
                VoiceSample.model_name == query.model_name,
                VoiceSample.model_version == query.model_version,
                # 排除本会议自身样本：跨会议识人，不自证
                VoiceSample.source_meeting_id != meeting.id,
                Person.user_id == meeting.user_id,
            )
            .order_by(distance)
            .limit(_REFERENCE_FETCH_LIMIT)
        )
    ).all()
    if not rows:
        return []

    by_person: dict[uuid.UUID, VoiceprintMatchResult] = {}
    for sample_id, person_id, person_name, dist in rows:
        r = by_person.get(person_id)
        if r is None:
            r = VoiceprintMatchResult(
                backend=_BACKEND_LOCAL,
                speaker_label=label,
                person_id=person_id,
                person_name=person_name,
                score=round(1.0 - float(dist), 6),
                model_key=_model_key(query.model_name, query.model_version),
                query_sample_seconds=round(query.duration, 3),
            )
            by_person[person_id] = r
        if len(r.evidence_sample_ids) < 3:
            r.evidence_sample_ids.append(sample_id)

    # 门槛 4 的精确参考样本数（fetch limit 可能截断，单独 count）
    counts = dict(
        (
            await session.execute(
                select(VoiceSample.person_id, func.count())
                .join(
                    SpeakerBinding,
                    SpeakerBinding.id == VoiceSample.assigned_by_binding_id,
                )
                .where(
                    VoiceSample.person_id.in_(by_person.keys()),
                    VoiceSample.person_id.is_not(None),
                    SpeakerBinding.confirmed_by == ConfirmedBy.human,
                    *active_binding_clause(),
                    VoiceSample.model_name == query.model_name,
                    VoiceSample.model_version == query.model_version,
                )
                .group_by(VoiceSample.person_id)
            )
        ).all()
    )
    for person_id, r in by_person.items():
        r.reference_sample_count = counts.get(person_id, 0)
    return sorted(by_person.values(), key=lambda r: r.score, reverse=True)


def _auto_bind_blockers(candidates: list[VoiceprintMatchResult]) -> list[str]:
    """六重门槛（§4.2）：返回未通过项（空列表 = 可自动绑定）。
    门槛 6（同模型）由匹配查询分桶天然保证。"""
    blockers: list[str] = []
    top1 = candidates[0]
    if not settings.voiceprint_auto_bind_enabled:
        blockers.append("disabled")
    if top1.score < settings.voiceprint_auto_bind_threshold:
        blockers.append(f"score {top1.score:.3f} < threshold")
    if len(candidates) > 1:
        margin = top1.score - candidates[1].score
        if margin < settings.voiceprint_auto_bind_min_margin:
            blockers.append(f"margin {margin:.3f} < min_margin")
    if top1.reference_sample_count < settings.voiceprint_min_reference_samples:
        blockers.append(
            f"reference samples {top1.reference_sample_count} < min"
        )
    if top1.query_sample_seconds < settings.voiceprint_min_sample_seconds:
        blockers.append(
            f"query sample {top1.query_sample_seconds:.1f}s < min seconds"
        )
    return blockers


async def identify_speakers(
    session: AsyncSession, meeting: Meeting
) -> dict[str, list[VoiceprintMatchResult]]:
    """转写落库后的本地识人收口（管道调用；不 commit，随阶段事务提交）。

    Phase 1：匹配结果落结构化日志（含三档分类与拦截门槛），
    仅当 VOICEPRINT_AUTO_BIND_ENABLED=true 且全部门槛通过时才自动绑定
    （confirmed_by=auto，样本不入长期参考库，见 speakers._apply_binding）。
    """
    matches = await match_local_voiceprints(session, meeting)
    for label, candidates in matches.items():
        if not candidates:
            logger.info(
                "voiceprint[local] %s %r: no reference match (unknown)",
                meeting.id, label,
            )
            continue
        top1 = candidates[0]
        blockers = _auto_bind_blockers(candidates)
        tier = (
            "auto" if not blockers
            else "candidate" if top1.score >= settings.voiceprint_ask_threshold
            else "unknown"
        )
        logger.info(
            "voiceprint[local] %s %r: top1=%s score=%.3f refs=%d bucket=%s "
            "tier=%s blockers=%s",
            meeting.id, label, top1.person_name, top1.score,
            top1.reference_sample_count, top1.model_key, tier,
            blockers or "-",
        )
        if not blockers:
            person = await session.get(Person, top1.person_id)
            if person is None:  # 并发删除竞态兜底
                continue
            await _apply_binding(
                session, meeting, label, person, ConfirmedBy.auto, top1.score
            )
            logger.info(
                "voiceprint[local] auto-bound %r -> %s (score=%.3f)",
                label, person.name, top1.score,
            )
    return matches


async def dismiss_speaker(
    session: AsyncSession, meeting: Meeting, speaker_label: str
) -> None:
    """确认卡「跳过」（§5）：写 dismissal，幂等；之后手动绑定会清掉本记录。"""
    exists = await session.scalar(
        select(func.count())
        .select_from(TranscriptSegment)
        .where(
            TranscriptSegment.meeting_id == meeting.id,
            TranscriptSegment.speaker_label == speaker_label,
        )
    )
    if not exists:
        raise UnknownSpeakerLabel(
            f"speaker label {speaker_label!r} not found in meeting transcript"
        )
    await session.execute(
        pg_insert(SpeakerIdentityDismissal)
        .values(
            id=uuid.uuid4(), meeting_id=meeting.id, speaker_label=speaker_label
        )
        .on_conflict_do_nothing(
            constraint="uq_speaker_dismissal_meeting_label"
        )
    )
    await session.commit()


async def speaker_identity_status(
    session: AsyncSession, meeting_id: uuid.UUID
) -> dict[str, str]:
    """speaker_label → identified / dismissed / pending（§5，动态计算不持久化）。

    确认卡（Phase 2）只对 pending 的 speaker 出现；全部 identified 时静默。
    """
    labels = set(
        await session.scalars(
            select(TranscriptSegment.speaker_label)
            .where(TranscriptSegment.meeting_id == meeting_id)
            .distinct()
        )
    )
    bound = set(
        await session.scalars(
            select(SpeakerBinding.speaker_label).where(
                SpeakerBinding.meeting_id == meeting_id,
                *active_binding_clause(),
            )
        )
    )
    dismissed = set(
        await session.scalars(
            select(SpeakerIdentityDismissal.speaker_label).where(
                SpeakerIdentityDismissal.meeting_id == meeting_id
            )
        )
    )
    status: dict[str, str] = {}
    for label in sorted(labels):
        if label in bound:
            status[label] = "identified"
        elif label in dismissed:
            status[label] = "dismissed"
        else:
            status[label] = "pending"
    return status
