"""全局术语表（跨会议热词，PRD Feature 1）

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-13

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "glossary_terms",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("term", sa.String(255), nullable=False),
        sa.Column(
            "enabled", sa.Boolean(), nullable=False, server_default=sa.true()
        ),
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
        sa.UniqueConstraint("user_id", "term", name="uq_glossary_terms_user_term"),
    )
    op.create_index("ix_glossary_terms_user_id", "glossary_terms", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_glossary_terms_user_id", table_name="glossary_terms")
    op.drop_table("glossary_terms")
