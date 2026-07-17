"""声纹记忆 Phase 1（TECH_DESIGN_VOICEPRINT_MEMORY_V1 §4/§5）：
样本归属物化/改绑/撤销、同模型分桶匹配、自动绑定门槛、dismissal 动态状态、
mock 端到端跨会议识人。"""
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
    SpeakerIdentityDismissal,
    TranscriptSegment,
    VoiceSample,
)
from app.services.asr.base import ASRSegment
from app.services.speakers import (
    auto_bind_voiceprints,
    bind_speaker,
    unbind_speaker,
)
from app.services.voiceprints import (
    VoiceprintMatchResult,
    _auto_bind_blockers,
    dismiss_speaker,
    identify_speakers,
    match_local_voiceprints,
    speaker_identity_status,
)
from tests.conftest import requires_db

pytestmark = requires_db

_MODEL = ("test-diar", "1")


def _vec(x: float, y: float = 0.0) -> list[float]:
    return [x, y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


async def _mk_meeting(session, title: str, labels: list[str]) -> Meeting:
    meeting = Meeting(
        user_id=DEFAULT_USER_ID, title=title, audio_url="local://x.wav"
    )
    session.add(meeting)
    await session.flush()
    session.add_all(
        TranscriptSegment(
            meeting_id=meeting.id,
            seq=i,
            start_time=float(i * 10),
            end_time=float(i * 10 + 10),
            speaker_label=label,
            text=f"{label} 的发言",
        )
        for i, label in enumerate(labels)
    )
    await session.flush()
    return meeting


async def _mk_sample(
    session,
    meeting: Meeting,
    label: str,
    emb: list[float],
    *,
    model: tuple[str, str] = _MODEL,
    duration: float = 10.0,
    person_id=None,
    binding_id=None,
) -> VoiceSample:
    vs = VoiceSample(
        source_meeting_id=meeting.id,
        speaker_label=label,
        embedding=emb,
        model_name=model[0],
        model_version=model[1],
        sample_start=0.0,
        sample_end=duration,
        person_id=person_id,
        assigned_by_binding_id=binding_id,
    )
    session.add(vs)
    await session.flush()
    return vs


async def _cleanup(session, meetings: list[Meeting], person_names: list[str]):
    for m in meetings:
        obj = await session.get(Meeting, m.id)
        if obj is not None:
            await session.delete(obj)
    for name in person_names:
        p = await session.scalar(
            select(Person).where(
                Person.user_id == DEFAULT_USER_ID, Person.name == name
            )
        )
        if p is not None:
            await session.delete(p)
    await session.commit()


# ---------- §4.1 样本归属：物化 / 改绑 / 撤销 ----------


def test_human_bind_materializes_rebind_transfers_unbind_releases() -> None:
    async def scenario() -> None:
        async with SessionLocal() as session:
            try:
                # 历史会议 M0：spk_x 绑定张三 → 长期参考样本
                m0 = await _mk_meeting(session, "vp-m0", ["spk_x"])
                s0 = await _mk_sample(session, m0, "spk_x", _vec(1.0))
                await bind_speaker(session, m0, "spk_x", "张三")

                # 本会议 M1：spk_a / spk_b 两个说话人各一份样本
                m1 = await _mk_meeting(session, "vp-m1", ["spk_a", "spk_b"])
                sa = await _mk_sample(session, m1, "spk_a", _vec(0.9, 0.1))
                sb = await _mk_sample(session, m1, "spk_b", _vec(0.0, 1.0))

                b1, zhang = await bind_speaker(session, m1, "spk_a", "张三")
                await session.refresh(sa)
                await session.refresh(sb)
                # 只物化当前会议、当前 label 的样本；spk_b 不受影响
                assert sa.person_id == zhang.id
                assert sa.assigned_by_binding_id == b1.id
                assert sb.person_id is None

                # 改绑 spk_a → 李四：样本精确转移到新绑定
                b2, li = await bind_speaker(session, m1, "spk_a", "李四")
                await session.refresh(sa)
                await session.refresh(b1)
                assert sa.person_id == li.id
                assert sa.assigned_by_binding_id == b2.id
                assert b1.superseded_by == b2.id
                # 张三名下其他会议的样本不迁移（§4.1）
                await session.refresh(s0)
                assert s0.person_id == zhang.id

                # 撤销 spk_a：样本回无主池（不删除），审计行保留，segments 清空
                revoked = await unbind_speaker(session, m1, "spk_a")
                assert revoked.id == b2.id
                assert revoked.revoked_at is not None
                await session.refresh(sa)
                assert sa.person_id is None
                assert sa.assigned_by_binding_id is None
                seg = await session.scalar(
                    select(TranscriptSegment).where(
                        TranscriptSegment.meeting_id == m1.id,
                        TranscriptSegment.speaker_label == "spk_a",
                    )
                )
                assert seg.person_id is None
                # 张三在 M0 的参考样本不受 M1 撤销影响
                await session.refresh(s0)
                assert s0.person_id == zhang.id
            finally:
                await _cleanup(session, [m0, m1], ["张三", "李四"])

    asyncio.run(scenario())


def test_auto_binding_does_not_materialize_until_human_confirms() -> None:
    """auto 绑定样本不入长期参考；human 确认后才升级（§4.1 防漂移）。"""
    vp = f"vp-{uuid.uuid4().hex[:12]}"

    async def scenario() -> None:
        async with SessionLocal() as session:
            try:
                m = await _mk_meeting(session, "vp-auto", ["spk_c"])
                sc = await _mk_sample(session, m, "spk_c", _vec(1.0))
                segs = [
                    ASRSegment(
                        start_time=0.0,
                        end_time=10.0,
                        speaker_label="spk_c",
                        text="x",
                        voiceprint_id=vp,
                        voiceprint_confidence=0.9,
                    )
                ]
                bound = await auto_bind_voiceprints(session, m, segs)
                await session.commit()
                assert bound == 1
                await session.refresh(sc)
                assert sc.person_id is None  # auto 不物化样本
                assert sc.assigned_by_binding_id is None

                # 用户确认自动结果（human 覆盖）→ 样本升级为长期参考
                b, person = await bind_speaker(session, m, "spk_c", "spk_c")
                assert b.confirmed_by == ConfirmedBy.human
                await session.refresh(sc)
                assert sc.person_id == person.id
                assert sc.assigned_by_binding_id == b.id
            finally:
                await _cleanup(session, [m], ["spk_c"])

    asyncio.run(scenario())


# ---------- §5 dismissal 动态状态 ----------


def test_dismissal_dynamic_status_lifecycle() -> None:
    async def scenario() -> None:
        async with SessionLocal() as session:
            try:
                m = await _mk_meeting(session, "vp-dismiss", ["spk_a", "spk_b"])
                await session.commit()
                status = await speaker_identity_status(session, m.id)
                assert status == {"spk_a": "pending", "spk_b": "pending"}

                # 跳过 spk_a（幂等）
                await dismiss_speaker(session, m, "spk_a")
                await dismiss_speaker(session, m, "spk_a")
                status = await speaker_identity_status(session, m.id)
                assert status["spk_a"] == "dismissed"
                assert status["spk_b"] == "pending"

                # 手动绑定 → identified，且 dismissal 被清除
                await bind_speaker(session, m, "spk_a", "王五")
                status = await speaker_identity_status(session, m.id)
                assert status["spk_a"] == "identified"
                remaining = (
                    await session.scalars(
                        select(SpeakerIdentityDismissal).where(
                            SpeakerIdentityDismissal.meeting_id == m.id
                        )
                    )
                ).all()
                assert remaining == []

                # 撤销绑定 → dismissal 已清除，确认卡自然重新出现（pending）
                await unbind_speaker(session, m, "spk_a")
                status = await speaker_identity_status(session, m.id)
                assert status["spk_a"] == "pending"
            finally:
                await _cleanup(session, [m], ["王五"])

    asyncio.run(scenario())


# ---------- §4.2 匹配：同模型分桶 + 排序 + 参考样本定义 ----------


def test_match_buckets_ranks_and_ignores_non_reference_samples() -> None:
    async def scenario() -> None:
        async with SessionLocal() as session:
            try:
                # 张三参考样本（human 绑定物化）：与查询向量完全同向
                m0 = await _mk_meeting(session, "vp-ref-a", ["spk_x"])
                await _mk_sample(session, m0, "spk_x", _vec(1.0))
                await bind_speaker(session, m0, "spk_x", "张三")
                # 王五参考样本：cos=0.6
                m0b = await _mk_meeting(session, "vp-ref-b", ["spk_w"])
                await _mk_sample(session, m0b, "spk_w", _vec(0.6, 0.8))
                await bind_speaker(session, m0b, "spk_w", "王五")
                # 赵六：向量相同但模型版本不同 → 不同桶，不可比（门槛 6）
                m0c = await _mk_meeting(session, "vp-ref-c", ["spk_z"])
                await _mk_sample(
                    session, m0c, "spk_z", _vec(1.0), model=("test-diar", "2")
                )
                await bind_speaker(session, m0c, "spk_z", "赵六")
                # 直接塞一个 person_id 非空但无 human 绑定溯源的样本
                # （相当于 auto 产物）→ 不是长期参考，不参与匹配
                fake_person = Person(user_id=DEFAULT_USER_ID, name="幽灵")
                session.add(fake_person)
                await session.flush()
                m0d = await _mk_meeting(session, "vp-ref-d", ["spk_g"])
                await _mk_sample(
                    session, m0d, "spk_g", _vec(1.0), person_id=fake_person.id
                )
                await session.commit()

                # 新会议：speaker_001 未绑，embedding 与张三同向
                m2 = await _mk_meeting(session, "vp-new", ["speaker_001"])
                await _mk_sample(session, m2, "speaker_001", _vec(1.0))
                await session.commit()

                matches = await match_local_voiceprints(session, m2)
                assert set(matches) == {"speaker_001"}
                cands = matches["speaker_001"]
                names = [c.person_name for c in cands]
                assert names[0] == "张三"
                assert cands[0].score == pytest.approx(1.0, abs=1e-4)
                assert cands[0].model_key == "test-diar:1"
                assert cands[0].reference_sample_count == 1
                assert cands[0].evidence_sample_ids
                assert "王五" in names
                wang = cands[names.index("王五")]
                assert wang.score == pytest.approx(0.6, abs=1e-4)
                # 不同模型桶与非参考样本不出现
                assert "赵六" not in names
                assert "幽灵" not in names
            finally:
                await _cleanup(
                    session,
                    [m0, m0b, m0c, m0d, m2],
                    ["张三", "王五", "赵六", "幽灵"],
                )

    asyncio.run(scenario())


# ---------- §4.2 六重门槛（纯函数单测） ----------


def _result(score: float, refs: int = 3, seconds: float = 10.0):
    return VoiceprintMatchResult(
        backend="local",
        speaker_label="spk",
        person_id=uuid.uuid4(),
        person_name="某人",
        score=score,
        model_key="test-diar:1",
        reference_sample_count=refs,
        query_sample_seconds=seconds,
    )


def test_auto_bind_gates(monkeypatch) -> None:
    monkeypatch.setattr(settings, "voiceprint_auto_bind_enabled", True)
    monkeypatch.setattr(settings, "voiceprint_auto_bind_threshold", 0.8)
    monkeypatch.setattr(settings, "voiceprint_auto_bind_min_margin", 0.1)
    monkeypatch.setattr(settings, "voiceprint_min_reference_samples", 2)
    monkeypatch.setattr(settings, "voiceprint_min_sample_seconds", 5.0)

    # 全部门槛通过
    assert _auto_bind_blockers([_result(0.95), _result(0.5)]) == []
    # 默认关闭（决议 5/7）
    monkeypatch.setattr(settings, "voiceprint_auto_bind_enabled", False)
    assert "disabled" in " ".join(_auto_bind_blockers([_result(0.95)]))
    monkeypatch.setattr(settings, "voiceprint_auto_bind_enabled", True)
    # top1 分数不够
    assert any(
        "threshold" in b for b in _auto_bind_blockers([_result(0.7)])
    )
    # 两人声音接近：margin 不足
    assert any(
        "margin" in b
        for b in _auto_bind_blockers([_result(0.95), _result(0.9)])
    )
    # 参考样本数不足
    assert any(
        "reference" in b for b in _auto_bind_blockers([_result(0.95, refs=1)])
    )
    # 查询样本太短
    assert any(
        "seconds" in b
        for b in _auto_bind_blockers([_result(0.95, seconds=2.0)])
    )


# ---------- 端到端（mock ASR 管道跨会议） ----------


def test_pipeline_cross_meeting_identify_and_auto_bind(
    tmp_path, monkeypatch
) -> None:
    """会议 1 重命名 → 会议 2 自动识人：默认只落日志不绑定；
    开启 + 门槛通过后自动绑定（confirmed_by=auto，样本不入长期参考）。"""
    import app.services.pipeline as pipeline_mod
    from app.services.asr import MockASRProvider
    from app.services.pipeline import run_pipeline
    from app.services.storage import get_audio_storage
    from tests.conftest import make_wav

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    # 独占 embedding 桶：其他测试留下的 mock-diarizer 参考样本（同为确定性
    # embedding，score=1.0）会打破 margin 门槛——同模型分桶天然隔离
    bucket = f"test-{uuid.uuid4().hex[:8]}"

    class BucketedASR(MockASRProvider):
        async def transcribe(self, audio_path, hotwords=None, voiceprint_ids=None):
            result = await super().transcribe(audio_path, hotwords, voiceprint_ids)
            for e in result.speaker_embeddings:
                e.model_version = bucket
            return result

    monkeypatch.setattr(pipeline_mod, "get_asr_provider", lambda: BucketedASR())

    async def scenario() -> None:
        async def upload(name: str) -> uuid.UUID:
            src = make_wav(tmp_path / name, seconds=2.0)
            with src.open("rb") as f:
                audio_url = get_audio_storage().save(f, name)
            async with SessionLocal() as session:
                meeting = Meeting(
                    user_id=DEFAULT_USER_ID, title=name, audio_url=audio_url
                )
                session.add(meeting)
                await session.commit()
                return meeting.id

        m1_id = await upload("vp-e2e-1.wav")
        m2_id = await upload("vp-e2e-2.wav")
        m3_id = await upload("vp-e2e-3.wav")
        try:
            await run_pipeline(m1_id)
            async with SessionLocal() as session:
                m1 = await session.get(Meeting, m1_id)
                # 样本已带 speaker_label（管道直写）
                labels = set(
                    await session.scalars(
                        select(VoiceSample.speaker_label).where(
                            VoiceSample.source_meeting_id == m1_id
                        )
                    )
                )
                assert labels == {"speaker_001", "speaker_002"}
                # 会议 1：把 speaker_001 重命名为张三（隐式登记）
                await bind_speaker(session, m1, "speaker_001", "张三")

            # 会议 2：默认自动绑定关闭 → 只匹配不绑定
            await run_pipeline(m2_id)
            async with SessionLocal() as session:
                m2 = await session.get(Meeting, m2_id)
                bindings = (
                    await session.scalars(
                        select(SpeakerBinding).where(
                            SpeakerBinding.meeting_id == m2_id
                        )
                    )
                ).all()
                assert bindings == []  # 决议 5：校准前不自动绑
                # 匹配本身命中（mock embedding 跨会议同 label 稳定）
                matches = await match_local_voiceprints(session, m2)
                assert matches["speaker_001"][0].person_name == "张三"
                assert matches["speaker_001"][0].score == pytest.approx(
                    1.0, abs=1e-4
                )
                status = await speaker_identity_status(session, m2_id)
                assert status == {
                    "speaker_001": "pending",
                    "speaker_002": "pending",
                }

            # 会议 3：开启自动绑定 + 门槛放行（mock 样本短，时长门槛置 0）
            monkeypatch.setattr(settings, "voiceprint_auto_bind_enabled", True)
            monkeypatch.setattr(settings, "voiceprint_min_sample_seconds", 0.0)
            await run_pipeline(m3_id)
            async with SessionLocal() as session:
                bindings = (
                    await session.scalars(
                        select(SpeakerBinding).where(
                            SpeakerBinding.meeting_id == m3_id
                        )
                    )
                ).all()
                by_label = {b.speaker_label: b for b in bindings}
                assert set(by_label) == {"speaker_001"}  # 002 无参考样本
                auto = by_label["speaker_001"]
                assert auto.confirmed_by == ConfirmedBy.auto
                assert auto.confidence == pytest.approx(1.0, abs=1e-4)
                zhang = await session.get(Person, auto.person_id)
                assert zhang.name == "张三"
                # auto 绑定样本不物化（§4.1 防漂移）
                m3_samples = (
                    await session.scalars(
                        select(VoiceSample).where(
                            VoiceSample.source_meeting_id == m3_id
                        )
                    )
                ).all()
                assert all(s.person_id is None for s in m3_samples)
                status = await speaker_identity_status(session, m3_id)
                assert status["speaker_001"] == "identified"
                assert status["speaker_002"] == "pending"
        finally:
            async with SessionLocal() as session:
                for mid in (m1_id, m2_id, m3_id):
                    m = await session.get(Meeting, mid)
                    if m is not None:
                        await session.delete(m)
                p = await session.scalar(
                    select(Person).where(
                        Person.user_id == DEFAULT_USER_ID, Person.name == "张三"
                    )
                )
                if p is not None:
                    await session.delete(p)
                await session.commit()

    asyncio.run(scenario())
