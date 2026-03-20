"""Tests for consolidation engine."""

from datetime import datetime, timedelta

from dispatch_memory.consolidation import Consolidator
from dispatch_memory.models import (
    Entity, EntityType, Memory, MemoryStatus, TTLClass,
)


def test_permanent_no_decay():
    consolidator = Consolidator()
    mem = Memory(
        content="Always do X",
        ttl_class=TTLClass.PERMANENT,
        significance=1.0,
    )
    score = consolidator.compute_decay_score(mem)
    assert score >= 1.0


def test_short_ttl_decays_fast():
    consolidator = Consolidator()
    mem = Memory(
        content="Mentioned once",
        ttl_class=TTLClass.SHORT,
        significance=0.3,
        last_accessed=datetime.utcnow() - timedelta(days=14),
        access_count=1,
    )
    score = consolidator.compute_decay_score(mem)
    # After 14 days with 7-day half-life, should be quite low
    assert score < 0.2


def test_recent_high_significance_stays_active():
    consolidator = Consolidator()
    mem = Memory(
        content="Important recent thing",
        ttl_class=TTLClass.MEDIUM,
        significance=0.9,
        last_accessed=datetime.utcnow(),
        access_count=5,
    )
    score = consolidator.compute_decay_score(mem)
    assert score > 0.5


def test_decay_pass_classifies():
    consolidator = Consolidator(archive_threshold=0.2, forget_threshold=0.1)

    memories = [
        Memory(
            content="Active",
            significance=0.9,
            ttl_class=TTLClass.LONG,
            last_accessed=datetime.utcnow(),
        ),
        Memory(
            content="Should archive",
            significance=0.1,
            ttl_class=TTLClass.SHORT,
            last_accessed=datetime.utcnow() - timedelta(days=30),
        ),
    ]

    result = consolidator.run_decay_pass(memories)
    assert len(result["active"]) >= 1


def test_association_discovery():
    consolidator = Consolidator()
    shared_entity = Entity(name="Project Alpha", entity_type=EntityType.PROJECT)

    memories = [
        Memory(content="A", entities=[shared_entity]),
        Memory(content="B", entities=[shared_entity]),
        Memory(content="C", entities=[Entity(name="Unrelated", entity_type=EntityType.CONCEPT)]),
    ]

    associations = consolidator.discover_associations(memories)
    assert len(associations) >= 1
    assert "project alpha" in associations[0][2]
