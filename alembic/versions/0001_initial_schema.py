"""initial schema per PRD §5 (all 10 tables incl. phase-2 stub fields)

Revision ID: 0001
Revises:
Create Date: 2026-07-09

"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

# MVP 单默认用户（PRD §6）
DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001"


def _pk() -> sa.Column:
    return sa.Column("id", UUID(as_uuid=True), primary_key=True)


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    ]


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        _pk(),
        sa.Column("name", sa.String(255), nullable=False),
        *_timestamps(),
    )

    op.create_table(
        "auth_identities",
        _pk(),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "type",
            sa.Enum(
                "email",
                "wechat_unionid",
                "phone",
                name="auth_identity_type",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("identifier", sa.String(255), nullable=False),
        sa.Column("credential", sa.String(255)),
        *_timestamps(),
        sa.UniqueConstraint("type", "identifier", name="uq_auth_identities_type_identifier"),
    )
    op.create_index("ix_auth_identities_user_id", "auth_identities", ["user_id"])

    op.create_table(
        "persons",
        _pk(),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("consent_record", JSONB),
        *_timestamps(),
    )
    op.create_index("ix_persons_user_id", "persons", ["user_id"])

    op.create_table(
        "meetings",
        _pk(),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "uploaded",
                "transcoding",
                "transcribing",
                "summarizing",
                "done",
                "failed",
                name="meeting_status",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
            server_default="uploaded",
        ),
        sa.Column("duration", sa.Float),
        sa.Column("audio_url", sa.String(1024)),
        sa.Column("error_message", sa.Text),
        *_timestamps(),
    )
    op.create_index("ix_meetings_user_id", "meetings", ["user_id"])

    op.create_table(
        "transcript_segments",
        _pk(),
        sa.Column(
            "meeting_id",
            UUID(as_uuid=True),
            sa.ForeignKey("meetings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("start_time", sa.Float, nullable=False),
        sa.Column("end_time", sa.Float, nullable=False),
        sa.Column("speaker_label", sa.String(64), nullable=False),
        sa.Column(
            "person_id",
            UUID(as_uuid=True),
            sa.ForeignKey("persons.id", ondelete="SET NULL"),
        ),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embedding", Vector()),
        *_timestamps(),
        sa.UniqueConstraint("meeting_id", "seq", name="uq_transcript_segments_meeting_seq"),
    )
    op.create_index("ix_transcript_segments_meeting_id", "transcript_segments", ["meeting_id"])
    op.create_index("ix_transcript_segments_person_id", "transcript_segments", ["person_id"])

    op.create_table(
        "speaker_bindings",
        _pk(),
        sa.Column(
            "meeting_id",
            UUID(as_uuid=True),
            sa.ForeignKey("meetings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("speaker_label", sa.String(64), nullable=False),
        sa.Column(
            "person_id",
            UUID(as_uuid=True),
            sa.ForeignKey("persons.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "confirmed_by",
            sa.Enum(
                "auto",
                "human",
                name="binding_confirmed_by",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("confidence", sa.Float),
        sa.Column(
            "superseded_by",
            UUID(as_uuid=True),
            sa.ForeignKey("speaker_bindings.id", ondelete="SET NULL"),
        ),
        *_timestamps(),
    )
    op.create_index("ix_speaker_bindings_meeting_id", "speaker_bindings", ["meeting_id"])
    op.create_index("ix_speaker_bindings_person_id", "speaker_bindings", ["person_id"])

    op.create_table(
        "voice_samples",
        _pk(),
        sa.Column(
            "person_id",
            UUID(as_uuid=True),
            sa.ForeignKey("persons.id", ondelete="CASCADE"),
        ),
        sa.Column(
            "source_meeting_id",
            UUID(as_uuid=True),
            sa.ForeignKey("meetings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("sample_start", sa.Float, nullable=False),
        sa.Column("sample_end", sa.Float, nullable=False),
        *_timestamps(),
    )
    op.create_index("ix_voice_samples_person_id", "voice_samples", ["person_id"])
    op.create_index("ix_voice_samples_source_meeting_id", "voice_samples", ["source_meeting_id"])

    op.create_table(
        "summaries",
        _pk(),
        sa.Column(
            "meeting_id",
            UUID(as_uuid=True),
            sa.ForeignKey("meetings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("style_profile_id", sa.String(128)),
        sa.Column("content_json", JSONB, nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("meeting_id", "version", name="uq_summaries_meeting_version"),
    )
    op.create_index("ix_summaries_meeting_id", "summaries", ["meeting_id"])

    op.create_table(
        "action_items",
        _pk(),
        sa.Column(
            "meeting_id",
            UUID(as_uuid=True),
            sa.ForeignKey("meetings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("task", sa.Text, nullable=False),
        sa.Column(
            "owner_person_id",
            UUID(as_uuid=True),
            sa.ForeignKey("persons.id", ondelete="SET NULL"),
        ),
        sa.Column("owner_text", sa.String(255)),
        sa.Column("deadline", sa.String(255)),
        sa.Column(
            "source_segment_id",
            UUID(as_uuid=True),
            sa.ForeignKey("transcript_segments.id", ondelete="SET NULL"),
        ),
        *_timestamps(),
    )
    op.create_index("ix_action_items_meeting_id", "action_items", ["meeting_id"])
    op.create_index("ix_action_items_source_segment_id", "action_items", ["source_segment_id"])

    op.create_table(
        "chat_messages",
        _pk(),
        sa.Column(
            "meeting_id",
            UUID(as_uuid=True),
            sa.ForeignKey("meetings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("cited_segment_ids", JSONB),
        *_timestamps(),
    )
    op.create_index("ix_chat_messages_meeting_id", "chat_messages", ["meeting_id"])

    op.execute(
        "INSERT INTO users (id, name, created_at, updated_at) "
        f"VALUES ('{DEFAULT_USER_ID}', 'Default User', now(), now())"
    )


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("action_items")
    op.drop_table("summaries")
    op.drop_table("voice_samples")
    op.drop_table("speaker_bindings")
    op.drop_table("transcript_segments")
    op.drop_table("meetings")
    op.drop_table("persons")
    op.drop_table("auth_identities")
    op.drop_table("users")
