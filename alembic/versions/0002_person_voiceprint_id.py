"""persons.voiceprint_id — 云端声纹库 ID（Seed ASR 2.0 声纹匹配）

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-12

"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "persons",
        sa.Column("voiceprint_id", sa.String(128), nullable=True),
    )
    op.create_unique_constraint(
        "uq_persons_voiceprint_id", "persons", ["voiceprint_id"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_persons_voiceprint_id", "persons", type_="unique")
    op.drop_column("persons", "voiceprint_id")
