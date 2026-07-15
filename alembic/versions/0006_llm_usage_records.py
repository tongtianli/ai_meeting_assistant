"""LLM 用量审计表（Tech Design M4 §11，Phase 2）

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-14

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_usage_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        # 审计数据须在会议/用户删除后存活：普通列，无外键
        sa.Column("meeting_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("task_type", sa.String(32), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("fallback_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_type", sa.String(32), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
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
    )
    op.create_index("ix_llm_usage_records_meeting_id", "llm_usage_records", ["meeting_id"])
    op.create_index("ix_llm_usage_records_task_type", "llm_usage_records", ["task_type"])
    op.create_index("ix_llm_usage_records_provider", "llm_usage_records", ["provider"])
    op.create_index("ix_llm_usage_records_created_at", "llm_usage_records", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_llm_usage_records_created_at", table_name="llm_usage_records")
    op.drop_index("ix_llm_usage_records_provider", table_name="llm_usage_records")
    op.drop_index("ix_llm_usage_records_task_type", table_name="llm_usage_records")
    op.drop_index("ix_llm_usage_records_meeting_id", table_name="llm_usage_records")
    op.drop_table("llm_usage_records")
