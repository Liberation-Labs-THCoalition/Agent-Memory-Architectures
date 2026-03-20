"""Memory data models."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    PROJECT = "project"
    PERSON = "person"
    DECISION = "decision"
    WORKFLOW = "workflow"
    FACT = "fact"
    INTERACTION = "interaction"
    STANDING_INSTRUCTION = "standing_instruction"


class TTLClass(str, Enum):
    PERMANENT = "permanent"
    LONG = "long"
    MEDIUM = "medium"
    SHORT = "short"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    COMPRESSED = "compressed"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"


class EntityType(str, Enum):
    PERSON = "PERSON"
    ORG = "ORG"
    PROJECT = "PROJECT"
    CONCEPT = "CONCEPT"
    TOOL = "TOOL"
    EVENT = "EVENT"
    LOCATION = "LOCATION"
    DATE = "DATE"
    UNKNOWN = "UNKNOWN"


class Entity(BaseModel):
    """An extracted named entity from a memory."""

    name: str
    name_lower: str = ""
    entity_type: EntityType = EntityType.UNKNOWN
    mention_context: str = ""

    def model_post_init(self, __context) -> None:
        if not self.name_lower:
            self.name_lower = self.name.lower()


class Memory(BaseModel):
    """Core memory record."""

    # Identity
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    content_hash: str = ""

    # Classification
    memory_type: MemoryType = MemoryType.FACT
    tags: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)

    # Significance & Decay
    significance: float = Field(default=0.5, ge=0.0, le=1.0)
    ttl_class: TTLClass = TTLClass.MEDIUM
    status: MemoryStatus = MemoryStatus.ACTIVE
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Temporal
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0

    # Notion linkage
    notion_page_id: Optional[str] = None
    notion_database: Optional[str] = None  # Which PARA database it lives in

    # Embedding (local cache only, not stored in Notion)
    embedding: Optional[list[float]] = None

    def model_post_init(self, __context) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]

    @computed_field
    @property
    def entity_names(self) -> list[str]:
        """Convenience accessor for entity names."""
        return [e.name for e in self.entities]

    def refresh(self) -> None:
        """Bump access metrics without modifying content."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    def touch(self) -> None:
        """Mark as modified."""
        self.updated_at = datetime.utcnow()


class Triple(BaseModel):
    """Knowledge graph triple (Phase 3).

    Not stored until Phase 3, but defined here so all phases share one spec.
    """

    subject_entity: str
    predicate: str
    object_entity: str
    source_memory_id: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BootstrapPayload(BaseModel):
    """Curated context returned by memory_bootstrap."""

    standing_instructions: list[Memory] = Field(default_factory=list)
    active_projects: list[Memory] = Field(default_factory=list)
    recent_high_significance: list[Memory] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
