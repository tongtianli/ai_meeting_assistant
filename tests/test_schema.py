"""对模型 metadata 做 PRD §5 关键约束校验（不依赖数据库）。"""

import app.models  # noqa: F401  注册全部模型
from app.db.base import Base

EXPECTED_TABLES = {
    "users",
    "auth_identities",
    "meetings",
    "transcript_segments",
    "speaker_bindings",
    "persons",
    "voice_samples",
    "summaries",
    "action_items",
    "chat_messages",
}


def test_all_prd_tables_present() -> None:
    assert EXPECTED_TABLES <= set(Base.metadata.tables)


def test_timestamps_on_all_tables() -> None:
    for name in EXPECTED_TABLES:
        table = Base.metadata.tables[name]
        assert "created_at" in table.c, name
        assert "updated_at" in table.c, name


def test_voice_sample_stub_fields() -> None:
    vs = Base.metadata.tables["voice_samples"]
    # 暗桩：一期入库时 person_id 为空
    assert vs.c.person_id.nullable
    # 模型版本号必须留存
    assert not vs.c.model_name.nullable
    assert not vs.c.model_version.nullable
    # PRD §5：删除 Person 物理级联删除全部 VoiceSample
    person_fk = next(fk for fk in vs.foreign_keys if fk.column.table.name == "persons")
    assert person_fk.ondelete == "CASCADE"


def test_segment_stub_fields_nullable() -> None:
    seg = Base.metadata.tables["transcript_segments"]
    assert seg.c.person_id.nullable
    assert seg.c.embedding.nullable


def test_speaker_binding_auditable() -> None:
    sb = Base.metadata.tables["speaker_bindings"]
    assert sb.c.superseded_by.nullable
    fk = next(fk for fk in sb.foreign_keys if fk.column.table.name == "speaker_bindings")
    assert fk.column.name == "id"


def test_summary_versioned() -> None:
    s = Base.metadata.tables["summaries"]
    uq = [c for c in s.constraints if c.name == "uq_summaries_meeting_version"]
    assert uq, "summaries 应有 (meeting_id, version) 唯一约束"
