"""Initial schema — all Phase 1 tables.

Revision ID: 001
Revises: None
Create Date: 2026-02-01
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID, TSVECTOR
from pgvector.sqlalchemy import Vector

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "organizations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("org_type", sa.String(64), nullable=False, server_default="default"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("values_json", JSONB, nullable=True),
        sa.Column("bdi_json", JSONB, nullable=True),
    )

    op.create_table(
        "memory_units",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("significance", sa.Integer, nullable=False, server_default="5"),
        sa.Column("memory_layer", sa.String(64), nullable=False, server_default="working"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_memory_significance_org", "memory_units", ["significance", "org_id"])

    op.create_table(
        "memory_embeddings",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("memory_id", UUID(as_uuid=True), sa.ForeignKey("memory_units.id", ondelete="CASCADE"), unique=True),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("model", sa.String(128), nullable=False, server_default="all-mpnet-base-v2"),
    )
    op.execute(
        "CREATE INDEX ix_memory_embedding_hnsw ON memory_embeddings "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    op.create_table(
        "memory_lexical",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("memory_id", UUID(as_uuid=True), sa.ForeignKey("memory_units.id", ondelete="CASCADE"), unique=True),
        sa.Column("tsv", TSVECTOR, nullable=False),
    )
    op.create_index("ix_memory_lexical_tsv", "memory_lexical", ["tsv"], postgresql_using="gin")

    op.create_table(
        "memory_metadata",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("memory_id", UUID(as_uuid=True), sa.ForeignKey("memory_units.id", ondelete="CASCADE"), unique=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("entity_type", sa.String(128), nullable=False, server_default="general"),
        sa.Column("significance", sa.Integer, nullable=False, server_default="5"),
        sa.Column("extra", JSONB, nullable=True),
    )

    op.create_table(
        "memory_archives",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content_compressed", sa.LargeBinary, nullable=False),
        sa.Column("entropy_score", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("content_hash", sa.String(64), nullable=False, index=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("is_immutable", sa.Boolean, server_default="true"),
    )

    op.create_table(
        "temporal_memories",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("category", sa.String(128), nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("metadata_json", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_temporal_created_category", "temporal_memories", ["created_at", "category"])

    op.create_table(
        "intent_capsules",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("goal", sa.Text, nullable=False),
        sa.Column("constraints", JSONB, nullable=True),
        sa.Column("signature", sa.String(512), nullable=False, server_default=""),
        sa.Column("signed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean, server_default="true"),
    )

    op.create_table(
        "shield_constraints",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("constraint_type", sa.String(128), nullable=False),
        sa.Column("config", JSONB, nullable=True),
        sa.Column("is_active", sa.Boolean, server_default="true"),
    )


def downgrade() -> None:
    op.drop_table("shield_constraints")
    op.drop_table("intent_capsules")
    op.drop_table("temporal_memories")
    op.drop_table("memory_archives")
    op.drop_table("memory_metadata")
    op.drop_table("memory_lexical")
    op.drop_table("memory_embeddings")
    op.drop_table("memory_units")
    op.drop_table("organizations")
    op.execute("DROP EXTENSION IF EXISTS vector")
