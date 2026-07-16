"""pg_trgm 扩展 + transcript text GIN 索引（RAG 设计 §5.2 硬性前置）

`ILIKE '%token%'` 带前导通配符用不上 B-tree，会全表扫 transcript text——
关键词召回上线前必须建 trigram GIN 索引。

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-16

"""
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_transcript_segments_text_trgm "
        "ON transcript_segments USING gin (text gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_transcript_segments_text_trgm")
    # 扩展保留：可能被其他对象使用，卸载属运维决策
