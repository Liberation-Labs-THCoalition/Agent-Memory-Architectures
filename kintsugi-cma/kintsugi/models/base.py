"""Phase 1 SQLAlchemy models — all core tables."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from kintsugi.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    org_type: Mapped[str] = mapped_column(String(64), nullable=False, default="default")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    values_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    bdi_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    memories: Mapped[list["MemoryUnit"]] = relationship(back_populates="organization")


class MemoryUnit(Base):
    __tablename__ = "memory_units"
    __table_args__ = (
        Index("ix_memory_significance_org", "significance", "org_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    significance: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    memory_layer: Mapped[str] = mapped_column(String(64), nullable=False, default="working")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    organization: Mapped["Organization"] = relationship(back_populates="memories")
    embedding: Mapped["MemoryEmbedding | None"] = relationship(back_populates="memory")
    lexical: Mapped["MemoryLexical | None"] = relationship(back_populates="memory")
    metadata_row: Mapped["MemoryMetadata | None"] = relationship(back_populates="memory")


class MemoryEmbedding(Base):
    __tablename__ = "memory_embeddings"
    __table_args__ = (
        Index(
            "ix_memory_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_units.id", ondelete="CASCADE"), unique=True
    )
    embedding = mapped_column(Vector(768), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False, default="all-mpnet-base-v2")

    memory: Mapped["MemoryUnit"] = relationship(back_populates="embedding")


class MemoryLexical(Base):
    __tablename__ = "memory_lexical"
    __table_args__ = (
        Index("ix_memory_lexical_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_units.id", ondelete="CASCADE"), unique=True
    )
    tsv = mapped_column(TSVECTOR, nullable=False)

    memory: Mapped["MemoryUnit"] = relationship(back_populates="lexical")


class MemoryMetadata(Base):
    __tablename__ = "memory_metadata"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_units.id", ondelete="CASCADE"), unique=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    entity_type: Mapped[str] = mapped_column(String(128), nullable=False, default="general")
    significance: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    extra: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    memory: Mapped["MemoryUnit"] = relationship(back_populates="metadata_row")


class MemoryArchive(Base):
    __tablename__ = "memory_archives"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    content_compressed: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    entropy_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    archived_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    is_immutable: Mapped[bool] = mapped_column(Boolean, default=True)


class TemporalMemory(Base):
    __tablename__ = "temporal_memories"
    __table_args__ = (
        Index("ix_temporal_created_category", "created_at", "category"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class IntentCapsule(Base):
    __tablename__ = "intent_capsules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    constraints: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    signature: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    signed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class ShieldConstraint(Base):
    __tablename__ = "shield_constraints"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    constraint_type: Mapped[str] = mapped_column(String(128), nullable=False)
    config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
