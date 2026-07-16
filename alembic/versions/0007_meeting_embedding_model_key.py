"""meetings.embedding_model_key：向量模型身份（RAG 设计 §4.1.1-G）

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-16

"""
import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 可空：旧会议留空视为 stale，首次问答触发整场重嵌并回填
    op.add_column(
        "meetings", sa.Column("embedding_model_key", sa.String(128), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("meetings", "embedding_model_key")
