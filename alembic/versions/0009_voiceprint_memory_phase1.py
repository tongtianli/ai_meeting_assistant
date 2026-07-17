"""声纹记忆 Phase 1（TECH_DESIGN_VOICEPRINT_MEMORY_V1 §5）

- voice_samples 加 speaker_label（存量按 segments 时间重叠 best-effort 回填，
  歧义留空）+ assigned_by_binding_id（归属来源，可审计可回滚）
- 存量归属回填：回填出 speaker_label 且该 (meeting, label) 已有生效的
  human 绑定的样本，直接物化归属——历史会议里已重命名的说话人立即获得
  长期参考样本，无需重跑音频（暗桩兑现）
- speaker_bindings 加 revoked_at：撤销（无后继的解绑）与改绑（superseded_by）
  区分开，审计行保留
- 新表 speaker_identity_dismissals：确认卡「跳过」记录（决议 4）

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-17

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "speaker_bindings",
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "voice_samples", sa.Column("speaker_label", sa.String(64), nullable=True)
    )
    op.add_column(
        "voice_samples",
        sa.Column(
            "assigned_by_binding_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "speaker_bindings.id",
                ondelete="SET NULL",
                name="fk_voice_samples_assigned_by_binding",
            ),
            nullable=True,
        ),
    )

    # 存量 speaker_label 回填：样本时间窗与 segments 重叠且只命中一个
    # speaker_label 时采用；多说话人重叠（歧义）留空
    op.execute(
        """
        UPDATE voice_samples vs
        SET speaker_label = m.label
        FROM (
            SELECT vs2.id AS sample_id,
                   min(ts.speaker_label) AS label
            FROM voice_samples vs2
            JOIN transcript_segments ts
              ON ts.meeting_id = vs2.source_meeting_id
             AND ts.start_time < vs2.sample_end
             AND ts.end_time > vs2.sample_start
            GROUP BY vs2.id
            HAVING count(DISTINCT ts.speaker_label) = 1
        ) m
        WHERE vs.id = m.sample_id
          AND vs.speaker_label IS NULL
        """
    )
    # 存量归属回填：已有生效 human 绑定的 (meeting, label) → 样本直接物化归属
    op.execute(
        """
        UPDATE voice_samples vs
        SET person_id = sb.person_id,
            assigned_by_binding_id = sb.id
        FROM speaker_bindings sb
        WHERE sb.meeting_id = vs.source_meeting_id
          AND sb.speaker_label = vs.speaker_label
          AND sb.superseded_by IS NULL
          AND sb.revoked_at IS NULL
          AND sb.confirmed_by = 'human'
          AND vs.speaker_label IS NOT NULL
          AND vs.person_id IS NULL
        """
    )

    op.create_table(
        "speaker_identity_dismissals",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "meeting_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("meetings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("speaker_label", sa.String(64), nullable=False),
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
        sa.UniqueConstraint(
            "meeting_id", "speaker_label", name="uq_speaker_dismissal_meeting_label"
        ),
    )
    op.create_index(
        "ix_speaker_identity_dismissals_meeting_id",
        "speaker_identity_dismissals",
        ["meeting_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_speaker_identity_dismissals_meeting_id",
        table_name="speaker_identity_dismissals",
    )
    op.drop_table("speaker_identity_dismissals")
    op.drop_column("voice_samples", "assigned_by_binding_id")
    op.drop_column("voice_samples", "speaker_label")
    op.drop_column("speaker_bindings", "revoked_at")
