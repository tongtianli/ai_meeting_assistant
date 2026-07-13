"""声纹自动绑定：命中→绑定既有 Person / 自动建档 / 置信度阈值过滤。"""
import asyncio
import uuid

import pytest
from sqlalchemy import select

from app.core.config import settings
from app.db.session import SessionLocal
from app.models import (
    DEFAULT_USER_ID,
    ConfirmedBy,
    Meeting,
    Person,
    SpeakerBinding,
    TranscriptSegment,
)
from app.services.asr.base import ASRSegment
from app.services.speakers import auto_bind_voiceprints
from tests.conftest import requires_db

pytestmark = requires_db


def _seg(label: str, vp: str | None, score: float | None, seq_base: float) -> ASRSegment:
    return ASRSegment(
        start_time=seq_base,
        end_time=seq_base + 1.0,
        speaker_label=label,
        text="测试语句",
        voiceprint_id=vp,
        voiceprint_confidence=score,
    )


async def _setup_meeting(session, segments: list[ASRSegment]) -> Meeting:
    meeting = Meeting(
        user_id=DEFAULT_USER_ID, title="vp-binding-test", audio_url="local://x.wav"
    )
    session.add(meeting)
    await session.flush()
    session.add_all(
        TranscriptSegment(
            meeting_id=meeting.id,
            seq=i,
            start_time=s.start_time,
            end_time=s.end_time,
            speaker_label=s.speaker_label,
            text=s.text,
        )
        for i, s in enumerate(segments)
    )
    await session.flush()
    return meeting


def test_auto_bind_matches_existing_person_and_creates_unknown(monkeypatch) -> None:
    vp_known = f"vp-{uuid.uuid4().hex[:12]}"
    vp_new = f"vp-{uuid.uuid4().hex[:12]}"

    async def scenario() -> None:
        async with SessionLocal() as session:
            known = Person(
                user_id=DEFAULT_USER_ID, name="王工", voiceprint_id=vp_known
            )
            session.add(known)
            await session.flush()

            segments = [
                _seg("王工", vp_known, 0.92, 0.0),
                _seg("小李", vp_new, 0.88, 1.0),  # 本地无档案 → 自动建档
                _seg("speaker_002", None, None, 2.0),  # 未命中 → 不绑定
            ]
            meeting = await _setup_meeting(session, segments)

            bound = await auto_bind_voiceprints(session, meeting, segments)
            await session.commit()
            assert bound == 2

            try:
                bindings = (
                    await session.scalars(
                        select(SpeakerBinding).where(
                            SpeakerBinding.meeting_id == meeting.id
                        )
                    )
                ).all()
                assert len(bindings) == 2
                assert all(b.confirmed_by == ConfirmedBy.auto for b in bindings)
                by_label = {b.speaker_label: b for b in bindings}
                assert by_label["王工"].person_id == known.id
                assert by_label["王工"].confidence == pytest.approx(0.92)

                # 未知声纹自动建档，name 取云端注册名（即 label）
                new_person = await session.get(
                    Person, by_label["小李"].person_id
                )
                assert new_person.name == "小李"
                assert new_person.voiceprint_id == vp_new

                # person_id 已物化到命中的 segments；未命中的保持为空
                segs = (
                    await session.scalars(
                        select(TranscriptSegment).where(
                            TranscriptSegment.meeting_id == meeting.id
                        )
                    )
                ).all()
                by_seg_label = {s.speaker_label: s for s in segs}
                assert by_seg_label["王工"].person_id == known.id
                assert by_seg_label["speaker_002"].person_id is None
            finally:
                await session.delete(meeting)
                new_p = await session.scalar(
                    select(Person).where(Person.voiceprint_id == vp_new)
                )
                if new_p:
                    await session.delete(new_p)
                await session.delete(known)
                await session.commit()

    asyncio.run(scenario())


def test_auto_bind_respects_confidence_threshold(monkeypatch) -> None:
    monkeypatch.setattr(settings, "voiceprint_auto_bind_min_confidence", 0.6)
    vp = f"vp-{uuid.uuid4().hex[:12]}"

    async def scenario() -> None:
        async with SessionLocal() as session:
            segments = [_seg("低分命中", vp, 0.31, 0.0)]
            meeting = await _setup_meeting(session, segments)
            bound = await auto_bind_voiceprints(session, meeting, segments)
            await session.commit()
            try:
                assert bound == 0
                bindings = (
                    await session.scalars(
                        select(SpeakerBinding).where(
                            SpeakerBinding.meeting_id == meeting.id
                        )
                    )
                ).all()
                assert bindings == []
            finally:
                await session.delete(meeting)
                await session.commit()

    asyncio.run(scenario())
