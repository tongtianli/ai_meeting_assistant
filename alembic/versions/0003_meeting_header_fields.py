"""meetings 增加公司纪要模板抬头字段：地点/主持/记录人/重要程度

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-13

"""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("meetings", sa.Column("location", sa.String(255), nullable=True))
    op.add_column("meetings", sa.Column("host", sa.String(255), nullable=True))
    op.add_column("meetings", sa.Column("recorder", sa.String(255), nullable=True))
    op.add_column("meetings", sa.Column("importance", sa.String(32), nullable=True))


def downgrade() -> None:
    op.drop_column("meetings", "importance")
    op.drop_column("meetings", "recorder")
    op.drop_column("meetings", "host")
    op.drop_column("meetings", "location")
